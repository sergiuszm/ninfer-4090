#pragma once

#include "models/load_options.h"
#include "models/qwen3_5/execution/profile.h"
#include "ninfer/ops/projection_weights.h"

#include <variant>

namespace ninfer::models::qwen3_5::execution {

// Fork-local (PR #15): activation permissions of the dense body in one text pass. The artifact
// declares the groupwise Q4/Q5 uses A16Only. With LoadOptions::prefill_activations == Int8, a
// prefill pass (a SequenceUpdate: prompt prefill and scoring) permits A8 for them, which their
// Ops implement as group-64 INT8 activations. Decode, speculative verification, the MTP and
// draft layers, the output head and Vision keep the declared policies, so graph workspaces do
// not change. Workspace sizing and execution apply the same value.
class BodyActivations {
public:
    // The declared policies.
    BodyActivations() = default;

    BodyActivations(const LoadOptions& options, bool prefill) noexcept
        : int8_(prefill && options.prefill_activations == PrefillActivations::Int8) {}

    [[nodiscard]] ops::LinearPolicy policy(QType qtype, ops::LinearPolicy declared) const noexcept {
        return int8_ && groupwise(qtype) && declared == ops::LinearPolicy::A16Only
                   ? ops::LinearPolicy::AllowA8
                   : declared;
    }

    [[nodiscard]] ops::LinearProfile operator()(ops::LinearProfile profile) const noexcept {
        profile.policy = policy(profile.qtype, profile.policy);
        return profile;
    }

    [[nodiscard]] ops::SingleProjectionWeight
    operator()(ops::SingleProjectionWeight weight) const noexcept {
        weight.policy = policy(weight.weight.qtype, weight.policy);
        return weight;
    }

    // A paired projection has one permission, so both parents must qualify.
    [[nodiscard]] ops::ProjectionProfile operator()(ops::ProjectionProfile profile) const noexcept {
        if (!profile.second) {
            profile.first = (*this)(profile.first);
        } else if (groupwise(profile.first.qtype) && groupwise(profile.second->qtype)) {
            profile.first.policy   = policy(profile.first.qtype, profile.first.policy);
            profile.second->policy = policy(profile.second->qtype, profile.second->policy);
        }
        return profile;
    }

    [[nodiscard]] ops::ProjectionWeights operator()(ops::ProjectionWeights weights) const noexcept {
        if (auto* single = std::get_if<ops::SingleProjectionWeight>(&weights)) {
            *single = (*this)(*single);
        } else {
            auto& pair = std::get<ops::PairedProjectionWeights>(weights);
            if (groupwise(pair.first.qtype) && groupwise(pair.second.qtype))
                pair.policy = policy(pair.first.qtype, pair.policy);
        }
        return weights;
    }

private:
    [[nodiscard]] static bool groupwise(QType qtype) noexcept {
        return qtype == QType::Q4_G64_FP16 || qtype == QType::Q5_G64_FP16;
    }

    bool int8_ = false;
};

// Whether the rule permits A8 for at least one text-body use, that is, whether the effective
// prefill mode is int8. The prefill signature keys the measured cost on this mode, so the int8
// option on an artifact without a groupwise Q4/Q5 body keeps the a16 signature.
[[nodiscard]] inline bool lifts_text_body(const BodyActivations& body, const TextProfile& text) {
    const auto lifts = [&](const auto& profile) { return body(profile) != profile; };
    for (const auto& block : text.layers) {
        const bool mixer = std::visit(
            [&](const auto& m) { return lifts(m.projection) || lifts(m.output); }, block.mixer);
        const auto* dense = std::get_if<DenseProfile>(&block.ffn);
        if (mixer || (dense && (lifts(dense->gate_up) || lifts(dense->down)))) return true;
    }
    return false;
}

} // namespace ninfer::models::qwen3_5::execution
