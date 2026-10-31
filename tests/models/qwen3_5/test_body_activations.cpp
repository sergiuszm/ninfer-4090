// Fork-local (PR #15): the dense-body prefill activation rule of models/qwen3_5/execution.

#include "models/qwen3_5/execution/precision.h"

#include <iostream>
#include <variant>

namespace {

using namespace ninfer;
using models::qwen3_5::execution::BodyActivations;
using ops::LinearPolicy;
namespace execution = models::qwen3_5::execution;

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

Weight weight(QType qtype) {
    Weight result;
    result.qtype = qtype;
    return result;
}

void test_policy(const BodyActivations& prefill, const BodyActivations& unchanged) {
    for (const auto qtype : {QType::Q4_G64_FP16, QType::Q5_G64_FP16}) {
        expect(prefill.policy(qtype, LinearPolicy::A16Only) == LinearPolicy::AllowA8,
               "int8 prefill did not permit A8 for a groupwise A16Only use");
        expect(prefill.policy(qtype, LinearPolicy::AllowA4) == LinearPolicy::AllowA4,
               "int8 prefill changed a declared permission other than A16Only");
        expect(unchanged.policy(qtype, LinearPolicy::A16Only) == LinearPolicy::A16Only,
               "a16 mode or a non-prefill pass changed the declared policy");
    }
    for (const auto qtype : {QType::Q8_G32_FP16, QType::BF16, QType::NVFP4}) {
        expect(prefill.policy(qtype, LinearPolicy::A16Only) == LinearPolicy::A16Only,
               "int8 prefill changed a non-groupwise use");
    }
    expect(prefill.policy(QType::FP8_E4M3FN_ROW_BF16, LinearPolicy::AllowA8) ==
               LinearPolicy::AllowA8,
           "int8 prefill changed a declared FP8 permission");
}

void test_operands(const BodyActivations& prefill, const BodyActivations& unchanged) {
    const ops::LinearProfile q5{QType::Q5_G64_FP16, 5120, 6144};
    expect(prefill(q5).policy == LinearPolicy::AllowA8 &&
               unchanged(q5).policy == LinearPolicy::A16Only,
           "linear profile rule differs from the policy rule");
    const ops::SingleProjectionWeight q4{weight(QType::Q4_G64_FP16)};
    expect(prefill(q4).policy == LinearPolicy::AllowA8 &&
               unchanged(q4).policy == LinearPolicy::A16Only,
           "single weight rule differs from the policy rule");

    const ops::ProjectionProfile pair{{QType::Q4_G64_FP16, 7168, 5120},
                                      ops::LinearProfile{QType::Q5_G64_FP16, 7168, 5120}};
    const auto upgraded = prefill(pair);
    expect(upgraded.first.policy == LinearPolicy::AllowA8 &&
               upgraded.second->policy == LinearPolicy::AllowA8,
           "a groupwise paired profile did not take A8 on both parents");
    expect(unchanged(pair) == pair, "a16 mode or a non-prefill pass changed a paired profile");
    const ops::ProjectionProfile mixed{{QType::BF16, 7168, 5120},
                                       ops::LinearProfile{QType::Q5_G64_FP16, 7168, 5120}};
    expect(prefill(mixed) == mixed, "a pair with a non-groupwise parent changed permission");

    const ops::ProjectionWeights weights =
        ops::PairedProjectionWeights{weight(QType::Q4_G64_FP16), weight(QType::Q5_G64_FP16)};
    expect(std::get<ops::PairedProjectionWeights>(prefill(weights)).policy ==
                   LinearPolicy::AllowA8 &&
               std::get<ops::PairedProjectionWeights>(unchanged(weights)).policy ==
                   LinearPolicy::A16Only,
           "paired weights rule differs from the paired profile rule");
    expect(ops::projection_profile(prefill(weights)) == prefill(ops::projection_profile(weights)),
           "paired weights and their profile disagree after the rule");
}

// The effective mode: int8 only when the rule lifts at least one text-body use.
void test_effective_mode(const BodyActivations& prefill, const BodyActivations& unchanged) {
    const ops::LinearProfile fp8{QType::FP8_E4M3FN_ROW_BF16, 5120, 6144, LinearPolicy::AllowA8};
    const ops::ProjectionProfile fp8_projection{fp8, {}};
    const auto layer = [&](ops::ProjectionProfile projection, ops::LinearProfile output,
                           ops::LinearProfile gate_up, ops::LinearProfile down, bool gdn) {
        execution::BlockProfile block;
        if (gdn)
            block.mixer = execution::GdnProfile{projection, fp8_projection, {}, output};
        else
            block.mixer = execution::AttentionProfile{projection, {}, output};
        block.ffn = execution::DenseProfile{gate_up, down};
        return block;
    };
    const ops::LinearProfile q4{QType::Q4_G64_FP16, 34816, 5120};
    const ops::LinearProfile q5{QType::Q5_G64_FP16, 5120, 17408};
    const ops::ProjectionProfile pair{{QType::Q4_G64_FP16, 7168, 5120},
                                      ops::LinearProfile{QType::Q5_G64_FP16, 7168, 5120}};

    execution::TextProfile groupwise;
    groupwise.layers = {layer(pair, q5, q4, q5, false), layer(pair, q5, q4, q5, true)};
    expect(execution::lifts_text_body(prefill, groupwise),
           "int8 prefill on a groupwise body is not effective");
    expect(!execution::lifts_text_body(unchanged, groupwise),
           "a16 mode or a non-prefill pass reports an effective int8 mode");

    execution::TextProfile fp8_body;
    fp8_body.output_head = q5;
    fp8_body.layers      = {layer(fp8_projection, fp8, fp8, fp8, false),
                            layer(fp8_projection, fp8, fp8, fp8, true)};
    expect(!execution::lifts_text_body(prefill, fp8_body),
           "int8 prefill without a groupwise body use reports an effective int8 mode");

    auto one_use = fp8_body;
    // The FP8 body with one Q5 down projection.
    std::get<execution::DenseProfile>(one_use.layers[1].ffn).down = q5;
    expect(execution::lifts_text_body(prefill, one_use),
           "one groupwise body use does not make int8 prefill effective");

    execution::TextProfile moe;
    moe.layers        = {layer(fp8_projection, fp8, fp8, fp8, false)};
    moe.layers[0].ffn = execution::MoeLayerProfile{};
    expect(!execution::lifts_text_body(prefill, moe),
           "an MoE body without groupwise uses reports an effective int8 mode");
}

} // namespace

int main() {
    models::LoadOptions a16;
    models::LoadOptions int8;
    int8.prefill_activations = PrefillActivations::Int8;
    const BodyActivations prefill(int8, true);
    for (const auto& unchanged :
         {BodyActivations{}, BodyActivations(a16, true), BodyActivations(int8, false)}) {
        test_policy(prefill, unchanged);
        test_operands(prefill, unchanged);
        test_effective_mode(prefill, unchanged);
    }
    std::cout << (failures ? "FAIL" : "ok") << '\n';
    return failures ? 1 : 0;
}
