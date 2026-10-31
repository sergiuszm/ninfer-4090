#include "ninfer/engine.h"
#include "product/config/options.h"
#include "serve/translate.h"
#include "serve/openai_chat.h"
#include "serve/anthropic_messages.h"
#include "product/logging/configuration.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer::serve;
using ninfer::product::ServeOptions;

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return ninfer::product::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function>
void rejects(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid server configuration was accepted");
}

void test_resources_and_units() {
    const auto defaults = parse({"ninfer-serve", "model.ninfer"});
    const auto& engine  = defaults.launch.engine;
    require(engine.context_cache.enabled && !engine.context_cache.host_capacity_bytes &&
                engine.memory_headroom_bytes == (512ULL << 20),
            "server resource defaults were resolved before model preparation");
    require(defaults.launch.generation.max_tokens == 8192 &&
                !defaults.launch.generation.sampling.seed &&
                !defaults.launch.generation.enable_thinking &&
                defaults.launch.logging.stats_interval_ms == 5000,
            "server defaults changed");
    require(!defaults.model_id_override, "public model alias is unexpectedly set by default");
    for (auto [quantity, bytes] : std::vector<std::pair<std::string, std::uint64_t>>{
             {"0B", 0},
             {"1.5 MiB", 1572864},
             {"0.00000095367431640625MiB", 1},
             {"18446744073709551615B", UINT64_MAX},
             {"17592186044415.99999904632568359375MiB", UINT64_MAX}}) {
        const auto parsed =
            parse({"ninfer-serve", "model.ninfer", "--host-cache-capacity", quantity});
        require(parsed.launch.engine.context_cache.host_capacity_bytes == bytes,
                "capacity lost exact bytes");
    }
    const auto controls =
        parse({"ninfer-serve", "--pending-timeout", "1.25s", "model.ninfer", "--stats-interval",
               "0ms", "--media-cache-capacity", "0B", "--max-request-size", "1.5MiB",
               "--max-concurrency", "4", "--media-preprocess-threads", "auto", "--context-capacity",
               "auto", "--checkpoint-interval", "16384"});
    require(controls.launch.engine.pending_timeout_ms == 1250 &&
                controls.launch.logging.stats_interval_ms == 0 &&
                controls.max_request_bytes == 1572864 &&
                controls.launch.engine.context_cache.checkpoint_interval == 16384 &&
                controls.launch.engine.media_cache_bytes == 0 &&
                controls.launch.engine.context_capacity.mode ==
                    ninfer::ContextCapacityMode::Automatic,
            "unit, zero or automatic controls changed");
    for (const auto& suffix :
         std::vector<std::vector<std::string>>{{"--host-cache-capacity", "18446744073709551616B"},
                                               {"--context-capacity", "0B"},
                                               {"--context-capacity", "0.1B"},
                                               {"--host-cache-capacity", "0.1B"},
                                               {"--host-cache-capacity", "-1MiB"},
                                               {"--pending-timeout", "0ms"},
                                               {"--pending-timeout", "0.1ms"},
                                               {"--pending-timeout", "4294967296ms"},
                                               {"--media-live-capacity", "0B"},
                                               {"--max-concurrency", "9"},
                                               {"--checkpoint-interval", "0"},
                                               {"--port", "65536"},
                                               {"--top-k", "21"},
                                               {"--media-preprocess-threads", "0"}}) {
        std::vector<std::string> args{"ninfer-serve", "model.ninfer"};
        args.insert(args.end(), suffix.begin(), suffix.end());
        rejects([&] { parse(args); });
    }
}

void test_request_defaults() {
    const auto options =
        parse({"ninfer-serve", "model.ninfer", "--temperature", "0", "--top-p", "0.9", "--seed",
               "0", "--thinking-budget", "37", "--no-thinking", "--preserve-thinking"});
    GenerationRequest request;
    request.max_tokens = 1;
    auto semantics     = collect_prompt_options(request, options);
    auto execution     = to_request_options(request, options, true);
    require(
        execution.execution.sampling.temperature == 0 && execution.execution.sampling.seed == 0 &&
            execution.execution.thinking.budget == 37 && !semantics.enable_thinking &&
            semantics.defaults.enable_thinking == false && execution.execution.allow_prefix_reuse,
        "startup defaults did not reach the request");
    request.sampling.temperature = 1.1;
    request.sampling.seed        = 42;
    request.enable_thinking      = true;
    request.preserve_thinking    = false;
    semantics                    = collect_prompt_options(request, options);
    execution                    = to_request_options(request, options, false);
    require(execution.execution.sampling.temperature == 1.1F &&
                execution.execution.sampling.seed == 42 &&
                execution.execution.thinking.budget == 37 &&
                !execution.execution.allow_prefix_reuse && semantics.preserve_thinking == false,
            "explicit request fields did not override startup defaults");
    request.thinking_budget = 9;
    require(to_request_options(request, options, true).execution.thinking.budget == 9,
            "request thinking budget did not override the default");
    const auto large  = parse({"ninfer-serve", "model.ninfer", "--max-tokens", "4294967295"});
    const auto parsed = parse_chat_completion_request(
        RequestJson{{"model", "test"},
                    {"messages", RequestJson::array({{{"role", "user"}, {"content", "x"}}})}},
        {.default_max_tokens = large.launch.generation.max_tokens});
    require(parsed.generation.max_tokens == UINT32_MAX,
            "server output default overflowed at the protocol boundary");
}

void test_effort_tiers_reach_the_frontend() {
    // The serve layer passes every public tier through; the Qwen frontend maps the tiers that
    // the 27B template lacks (tests/models/common/qwen/test_frontend.cpp).
    const auto options  = parse({"ninfer-serve", "model.ninfer"});
    const auto messages = RequestJson::array({{{"role", "user"}, {"content", "x"}}});
    for (const auto& [wire, effort] :
         std::vector<std::pair<std::string, ninfer::ReasoningEffort>>{
             {"max", ninfer::ReasoningEffort::Max},
             {"high", ninfer::ReasoningEffort::High},
             {"minimal", ninfer::ReasoningEffort::Minimal}}) {
        const auto chat = parse_chat_completion_request(
            RequestJson{{"model", "test"}, {"messages", messages}, {"reasoning_effort", wire}}, {});
        require(collect_prompt_options(chat.generation, options).reasoning_effort == effort,
                "Chat reasoning_effort did not reach the frontend unchanged");
        const auto anthropic = parse_anthropic_messages_request(
            RequestJson{{"model", "test"},
                        {"max_tokens", 64},
                        {"messages", messages},
                        {"output_config", {{"effort", wire}}}},
            {});
        require(collect_prompt_options(anthropic.generation, options).reasoning_effort == effort,
                "Messages output_config.effort did not reach the frontend unchanged");
    }
}

void test_thinking_control() {
    require(!parse({"ninfer-serve", "model.ninfer"}).launch.engine.thinking_control,
            "thinking control is set by default");
    const auto custom =
        parse({"ninfer-serve", "model.ninfer", "--thinking-control", "custom guidance"});
    require(custom.launch.engine.thinking_control == "custom guidance",
            "--thinking-control did not preserve its value");
    const auto view = ninfer::product::format_configuration(
        ninfer::product::configuration_view(custom));
    require(view.find("custom guidance") != std::string::npos,
            "startup display lost the thinking control");
    const auto reset = parse({"ninfer-serve", "model.ninfer", "--thinking-control", "custom",
                              "--thinking-control", "model"});
    require(!reset.launch.engine.thinking_control,
            "--thinking-control model did not select the model default");
    rejects([] { (void)parse({"ninfer-serve", "model.ninfer", "--thinking-control", ""}); });
    rejects([] { (void)parse({"ninfer-serve", "model.ninfer", "--thinking-control"}); });

    // The rank wire does not carry a custom control: the Engine refuses it for a multi-rank
    // deployment before it reads the artifact or starts any worker.
    auto engine = custom.launch.engine;
    engine.deployment.emplace();
    engine.deployment->ranks = {{.rank = 0}, {.rank = 1}};
    std::string message;
    try {
        ninfer::Engine refused(engine);
    } catch (const std::invalid_argument& error) { message = error.what(); }
    require(message.find("multi-rank") != std::string::npos,
            "a multi-rank deployment accepted a custom thinking control");
}

// Fork-local (PR #14): --tolerant-tool-calls is opt-in and shown in the startup configuration.
void test_tolerant_tool_calls() {
    const auto shown = [](const ServeOptions& options, bool value) {
        for (const auto& item : ninfer::product::configuration_view(options).items)
            if (item.key == "server.tolerant_tool_calls")
                return std::get<bool>(item.value) == value;
        return false;
    };
    const auto defaults = parse({"ninfer-serve", "model.ninfer"});
    require(!defaults.tolerant_tool_calls && shown(defaults, false),
            "tolerant tool calls are on by default");
    const auto enabled = parse({"ninfer-serve", "model.ninfer", "--tolerant-tool-calls"});
    require(enabled.tolerant_tool_calls && shown(enabled, true),
            "--tolerant-tool-calls was not applied and shown");
    require(!parse({"ninfer-serve", "model.ninfer", "--tolerant-tool-calls",
                    "--no-tolerant-tool-calls"})
                 .tolerant_tool_calls,
            "--no-tolerant-tool-calls did not override an earlier flag");
    rejects([] { (void)parse({"ninfer-serve", "model.ninfer", "--tolerant-tool-calls=maybe"}); });

    GenerationRequest request;
    require(!to_request_options(request, defaults, true).output.tolerant_tool_calls &&
                to_request_options(request, enabled, true).output.tolerant_tool_calls,
            "--tolerant-tool-calls did not reach the request output options");
}

void test_provenance() {
    for (auto args : std::vector<std::vector<std::string>>{
             {"ninfer-serve", "model.ninfer", "--api-key", "private-value"},
             {"ninfer-serve", "--api-key=private-value", "model.ninfer"}}) {
        const auto options = parse(args);
        require(options.api_key == "private-value",
                "authentication value was not retained for the server");
        const auto view = ninfer::product::configuration_view(options);
        for (const auto& argument : view.argv)
            require(argument.find("private-value") == std::string::npos, "argv leaked the API key");
        const auto rendered = ninfer::product::format_configuration(view);
        require(rendered.find("private-value") == std::string::npos &&
                    rendered.find("configured") != std::string::npos,
                "startup display leaked or lost authentication state");
    }
}

// Fork-local (rtx4090-port): each fork KV dtype selects its own storage profile.
void test_fork_kv_dtypes() {
    for (const auto& [name, storage] :
         {std::pair{"rk8v4", ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64},
          std::pair{"rk4v4", ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueGroup64},
          std::pair{"rk4v4-e8", ninfer::KvCacheStorage::RK4V4E8},
          std::pair{"rk2v4-e8", ninfer::KvCacheStorage::RK2V4E8}}) {
        const auto options = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", name});
        require(options.launch.engine.kv_cache == storage,
                "--kv-dtype did not select its fork KV mode");
        require(std::string(ninfer::kv_cache_profile_info(storage).name) == name,
                "fork KV mode name does not round-trip");
    }
}

// Fork-local (catch-up #4 decision 2): requests that omit tool_constraints use auto unless
// --tool-constraints basic restores upstream's default.
void test_tool_constraints() {
    using ninfer::ToolConstraintMode;
    using ninfer::product::ConfigSourceKind;
    const auto shown = [](const ServeOptions& options, std::string_view value,
                          ConfigSourceKind source) {
        for (const auto& item : ninfer::product::configuration_view(options).items)
            if (item.key == "server.tool_constraints")
                return std::get<std::string>(item.value) == value && item.source.kind == source;
        return false;
    };
    const auto defaults = parse({"ninfer-serve", "model.ninfer"});
    require(defaults.tool_constraints == ToolConstraintMode::Automatic &&
                shown(defaults, "auto", ConfigSourceKind::Default),
            "server tool constraints do not default to auto in the startup configuration");
    const auto basic = parse({"ninfer-serve", "model.ninfer", "--tool-constraints", "basic"});
    require(basic.tool_constraints == ToolConstraintMode::Basic &&
                shown(basic, "basic", ConfigSourceKind::CommandLine),
            "--tool-constraints basic was not applied and shown");
    require(parse({"ninfer-serve", "model.ninfer", "--tool-constraints", "basic",
                   "--tool-constraints=auto"})
                    .tool_constraints == ToolConstraintMode::Automatic,
            "--tool-constraints=auto did not override an earlier value");
    for (const auto* value : {"strict", "Auto", "", "none"}) {
        rejects([&] {
            (void)parse({"ninfer-serve", "model.ninfer", "--tool-constraints", value});
        });
    }
    rejects([] { (void)parse({"ninfer-serve", "model.ninfer", "--tool-constraints"}); });

    const auto chat = RequestJson::parse(
        R"({"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"write","parameters":{"type":"object"}}}]})");
    const auto chat_default = parse_chat_completion_request(chat, request_limits(defaults));
    require(chat_default.generation.tool_choice.constraints == ToolConstraintMode::Automatic &&
                !chat_default.generation.constrains_tools(),
            "default server options constrained ordinary Chat tools");
    const auto chat_basic = parse_chat_completion_request(chat, request_limits(basic));
    require(chat_basic.generation.tool_choice.constraints == ToolConstraintMode::Basic &&
                chat_basic.generation.constrains_tools(),
            "--tool-constraints basic did not restore upstream Chat constraints");
    const auto messages = RequestJson::parse(
        R"({"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"write","input_schema":{"type":"object"}}]})");
    const auto messages_default =
        parse_anthropic_messages_request(messages, request_limits(defaults));
    require(messages_default.generation.tool_choice.constraints == ToolConstraintMode::Automatic &&
                !messages_default.generation.constrains_tools(),
            "default server options constrained ordinary Messages tools");
    const auto messages_basic = parse_anthropic_messages_request(messages, request_limits(basic));
    require(messages_basic.generation.tool_choice.constraints == ToolConstraintMode::Basic &&
                messages_basic.generation.constrains_tools(),
            "--tool-constraints basic did not restore upstream Messages constraints");
}

} // namespace

int main() {
    try {
        test_resources_and_units();
        test_request_defaults();
        test_thinking_control();
        test_tolerant_tool_calls();
        test_effort_tiers_reach_the_frontend();
        test_provenance();
        test_fork_kv_dtypes();
        test_tool_constraints();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
