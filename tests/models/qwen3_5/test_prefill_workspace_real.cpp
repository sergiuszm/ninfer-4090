// Fork-local (PR #15): startup sizing follows the dense-body prefill activation rule. With int8,
// the prefill workspace grows by the INT8 activation staging; the decode, verify and MTP rounds
// keep their a16 sizes, because only a prefill pass lifts the groupwise Q4/Q5 policies. Needs a
// Qwen3.5 artifact with a groupwise Q4/Q5 body and a CUDA device for planning; it launches no
// kernel.

#include "core/device.h"
#include "models/qwen3_5/execution/profile.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/resources.h"

#include <cstdlib>
#include <exception>
#include <iostream>

namespace {

namespace q35 = ninfer::models::qwen3_5;

q35::WorkspaceSpec plan_workspace(const char* artifact, ninfer::PrefillActivations mode) {
    ninfer::EngineOptions options;
    options.artifact_path            = artifact;
    options.max_context              = 8192;
    options.prefill_chunk            = 1024;
    options.max_concurrency          = 2;
    options.prefill_activations      = mode;
    options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 3;
    options.context_cache.enabled    = false;
    ninfer::artifact::Reader reader(options.artifact_path);
    const auto definition = q35::read_definition(reader, ninfer::models::load_options(options));
    const auto profile    = q35::execution::describe_model(*definition);
    ninfer::DeviceContext device(options.device);
    return q35::detail::make_sequence_planner_impl(profile, device, options)->minimum->workspace;
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
        int failures = 0;
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
