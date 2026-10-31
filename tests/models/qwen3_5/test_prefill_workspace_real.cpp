// Fork-local (PR #15): startup sizing follows the dense-body prefill activation rule. Each lifted
// sizing site of the artifact's text body reserves its INT8 activation staging (prefill_sizing.h),
// and the int8 prefill workspace grows; the decode, verify and MTP rounds keep their a16 sizes,
// because only a prefill pass lifts the groupwise Q4/Q5 policies. The total alone cannot tell
// which site grew: the a16 MLP stage needs more than its int8 staging. Needs a Qwen3.5 artifact
// with a groupwise Q4/Q5 body and a CUDA device for planning; it launches no kernel.

#include "core/device.h"
#include "models/qwen3_5/execution/profile.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/resources.h"
#include "prefill_sizing.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>

namespace {

namespace q35 = ninfer::models::qwen3_5;

constexpr std::int32_t kPrefillChunk = 1024;

ninfer::EngineOptions engine_options(const char* artifact, ninfer::PrefillActivations mode) {
    ninfer::EngineOptions options;
    options.artifact_path            = artifact;
    options.max_context              = 8192;
    options.prefill_chunk            = kPrefillChunk;
    options.max_concurrency          = 2;
    options.prefill_activations      = mode;
    options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 3;
    options.context_cache.enabled    = false;
    return options;
}

std::shared_ptr<const q35::execution::ModelProfile>
describe(const ninfer::EngineOptions& options) {
    ninfer::artifact::Reader reader(options.artifact_path);
    const auto definition = q35::read_definition(reader, ninfer::models::load_options(options));
    return q35::execution::describe_model(*definition);
}

q35::WorkspaceSpec plan_workspace(const char* artifact, ninfer::PrefillActivations mode) {
    const auto options = engine_options(artifact, mode);
    const auto profile = describe(options);
    ninfer::DeviceContext device(options.device);
    return q35::detail::make_sequence_planner_impl(profile, device, options)->minimum->workspace;
}

// Every kind of dense-body site must exist in the artifact and reserve its staging at the planned
// prefill interval.
int check_sizing_sites(const char* artifact) {
    namespace sizing = ninfer::test::qwen3_5_prefill;
    const auto profile = describe(engine_options(artifact, ninfer::PrefillActivations::Int8));
    const q35::execution::BodyActivations prefill(profile->options, true);
    const q35::execution::BodyActivations decode(profile->options, false);
    sizing::LiftedSites lifted;
    int failures =
        sizing::check_text_sizing(profile->text, prefill, decode, 1, kPrefillChunk, lifted);
    std::cout << "lifted sites: attention projection " << lifted.attention_projection
              << ", GDN projection " << lifted.gdn_projection << ", attention output "
              << lifted.attention_output << ", GDN output " << lifted.gdn_output << ", dense MLP "
              << lifted.dense_mlp << '\n';
    if (!lifted.attention_projection || !lifted.gdn_projection || !lifted.attention_output ||
        !lifted.gdn_output || !lifted.dense_mlp) {
        std::cerr << "the artifact lacks a lifted site of some kind\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const auto a16  = plan_workspace(artifact, ninfer::PrefillActivations::A16);
        const auto int8 = plan_workspace(artifact, ninfer::PrefillActivations::Int8);
        std::cout << "text_prefill a16/int8 " << a16.text_prefill << '/' << int8.text_prefill
                  << " ordinary_round " << a16.ordinary_round << '/' << int8.ordinary_round
                  << " mtp_round " << a16.mtp_round << '/' << int8.mtp_round << '\n';
        int failures = check_sizing_sites(artifact);
        if (int8.text_prefill <= a16.text_prefill) {
            std::cerr << "int8 prefill sizing has no INT8 activation staging\n";
            ++failures;
        }
        if (int8.ordinary_round != a16.ordinary_round || int8.mtp_round != a16.mtp_round) {
            std::cerr << "int8 changed the decode, verify or MTP round sizing\n";
            ++failures;
        }
        if (failures == 0) std::cout << "ok\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
