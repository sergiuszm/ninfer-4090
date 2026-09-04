#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <spdlog/logger.h>

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <csignal>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

// Fork-local: render one string as a single quoted key=value field value for the structured
// boot lines below (upstream's readable-log rewrite dropped its logging.h helper). Structural
// escaping only; callers decide whether a value is safe to log at all.
std::string quote_log_value(std::string_view value) {
    constexpr std::array<char, 16> hex = {'0', '1', '2', '3', '4', '5', '6', '7',
                                          '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (ch < 0x20 || ch == 0x7f) {
                out += "\\x";
                out.push_back(hex[ch >> 4]);
                out.push_back(hex[ch & 0x0f]);
            } else {
                out.push_back(static_cast<char>(ch));
            }
            break;
        }
    }
    out.push_back('"');
    return out;
}

const char* kv_capacity_mode_name(ninfer::KvCapacityMode mode) noexcept {
    return mode == ninfer::KvCapacityMode::Automatic ? "auto" : "explicit";
}

// Fork-local: upstream logs no pool breakdown. text_kv_bytes and mtp_kv_bytes over the KV
// capacity are the basis of the snapshot size model (18,529 B/token measured), so this fork
// keeps reporting them - restated in upstream's structured style rather than the old
// "state pools: text-kv=4.25 GiB" shape. See baseline/LOG-CONTRACT.md.
void log_state_pools(const std::shared_ptr<spdlog::logger>& logger,
                     const ninfer::serve::GenerationService& service) {
    const ninfer::MemorySummary memory = service.memory_summary();
    logger->info("engine state_pools text_kv_bytes={} mtp_kv_bytes={} dflash_kv_bytes={} "
                 "gdn_state_bytes={} replay_records_bytes={} persistent_arena_bytes={} "
                 "workspace_bytes={}",
                 memory.text_kv_bytes, memory.mtp_kv_bytes, memory.dflash_kv_bytes,
                 memory.gdn_state_bytes, memory.replay_records_bytes,
                 memory.sequence.capacity_bytes, memory.workspace.capacity_bytes);
}

void log_engine_capacity(const std::shared_ptr<spdlog::logger>& logger,
                         const ninfer::serve::GenerationService& service,
                         const ninfer::serve::ServeOptions& options) {
    const ninfer::MemorySummary memory            = service.memory_summary();
    const ninfer::ContextCostSummary context_cost = service.load_summary().context_cost;
    const ninfer::EngineOptions& engine           = service.engine_options();
    const ninfer::ContextCacheOptions& cache      = engine.context_cache;
    logger->info(
        "engine capacity kv_capacity_mode={} kv_capacity_tokens={} kv_page_groups={} "
        "kv_max_page_groups={} runtime_reservation_bytes={} available_after_weights_bytes={} "
        "available_after_startup_bytes={} kv_headroom_bytes={} planned_slack_bytes={} "
        "cuda_graph_allowance_bytes={}",
        kv_capacity_mode_name(memory.kv_capacity_mode), memory.kv_capacity,
        memory.kv_capacity_page_groups, memory.kv_capacity_max_page_groups,
        memory.runtime_reservation_bytes, memory.available_after_weights_bytes,
        memory.available_after_startup_bytes, memory.kv_capacity_headroom_bytes,
        memory.planned_slack_bytes, memory.cuda_graph_allowance_bytes);
    logger->info("engine context_cache enabled={} active_lanes={} device_state_slots={} "
                 "host_state_slots={} host_kv_bytes={} private_continuations={} shared_prefixes={} "
                 "long_anchors_per_continuation={} auto_long_anchors={}",
                 cache.enabled, engine.max_concurrency, *cache.device_state_slots,
                 cache.host_state_slots, cache.host_kv_capacity_bytes,
                 *cache.max_private_continuations, *cache.max_shared_prefixes,
                 *cache.max_long_anchors_per_continuation, service.automatic_private_anchors());
    logger->info(
        "engine context_cost transfer_source={} prefill_source={} hardware_class={} model_id={} "
        "weights_id={}",
        ninfer::context_cost_preset_source_name(context_cost.transfer_source),
        ninfer::context_cost_preset_source_name(context_cost.prefill_source),
        quote_log_value(context_cost.hardware_class),
        quote_log_value(context_cost.model_id),
        quote_log_value(context_cost.weights_id));
    if (options.enable_vision) {
        const ninfer::MediaCacheSummary media = service.media_cache_summary();
        logger->info(
            "engine media preprocess_threads={} cache_capacity_bytes={} live_capacity_bytes={}",
            media.preprocess_threads, media.capacity_bytes, media.live_capacity_bytes);
    }
}

} // namespace

int main(int argc, char** argv) {
    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::serve::serve_usage_text(argv[0]);
        return 0;
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-serve",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Service});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    ninfer::serve::OperationalLog operational_log(logger);
    bool serving = false;

    if (options.deprecated_turn_checkpoints_given) {
        logger->warn("--turn-checkpoints is retired and was ignored; rewrite checkpoints and "
                     "long anchors cover mid-history divergence, sized by "
                     "--max-long-anchors-per-continuation and placed by --auto-long-anchors");
    }
    if (!options.slot_save_path.empty()) {
        std::error_code directory_error;
        std::filesystem::create_directories(options.slot_save_path, directory_error);
        if (directory_error ||
            !std::filesystem::is_directory(options.slot_save_path, directory_error)) {
            logger->error("--slot-save-path is not a usable directory: {}",
                          quote_log_value(options.slot_save_path));
            return 1;
        }
    }

    try {
        ninfer::serve::HttpServer server(options, logger);
        if (!server.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }

        ninfer::serve::GenerationService service(options, startup_log.observer(), logger);
        startup_log.engine_ready(service.load_summary());
        log_engine_capacity(logger, service, options);
        log_state_pools(logger, service);

        using Clock                            = std::chrono::steady_clock;
        const Clock::time_point warmup_started = Clock::now();
        operational_log.warmup_started();
        try {
            service.warmup();
        } catch (const std::exception& exception) {
            const double seconds =
                std::chrono::duration<double>(Clock::now() - warmup_started).count();
            operational_log.warmup_failure(seconds, exception.what());
            return 1;
        }
        operational_log.warmup_complete(
            std::chrono::duration<double>(Clock::now() - warmup_started).count());
        server.attach(service);

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        serving = true;
        operational_log.server_ready(options.host, options.port, server.public_model_id(),
                                     !options.api_key.empty());

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        operational_log.server_stopped();
        return 0;
    } catch (const std::exception& exception) {
        g_server.store(nullptr);
        operational_log.server_failure(serving, exception.what());
        return 1;
    }
}
