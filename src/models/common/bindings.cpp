#include "models/common/bindings.h"

#include "artifact/views.h"

#include <algorithm>
#include <cmath>

namespace ninfer::models {

WeightUseId Bindings::use(WeightId id, std::string_view input) const {
    const auto& parameter = at(id);
    for (std::size_t i = 0; i < parameter.uses.size(); ++i) {
        if (parameter.uses[i].input == input) { return {id, i}; }
    }
    throw artifact::ArtifactError(parameter.reference.name + ": unresolved Use " +
                                  std::string(input));
}

WeightId Bindings::parameter(std::string name, artifact::Shape shape,
                             std::vector<std::string> inputs, std::optional<QType> exact_format) {
    if (parameters_.contains(name)) {
        throw artifact::ArtifactError(name + ": duplicate model parameter declaration");
    }
    PendingWeight pending;
    pending.reference = binder.parameter(name, std::move(shape), exact_format);
    // Fork-local (PR #15): the groupwise Q4/Q5 Ops implement an A8 permission as group-64 INT8
    // activations. Only the startup choice LoadOptions::prefill_activations selects them (the
    // Qwen3.5 dense-body prefill rule), never a declared permission, so these uses bind as
    // A16Only on every build.
    const auto& parts    = pending.reference.binding.parts;
    const bool groupwise = std::any_of(parts.begin(), parts.end(), [&](const auto& part) {
        const auto format = binder.reader().geometry(part.object).format;
        return format == QType::Q4_G64_FP16 || format == QType::Q5_G64_FP16;
    });
    for (const auto& input : inputs) {
        const auto& use = binder.use(name, input);
        if (!use.activation_policy) {
            throw artifact::ArtifactError(name + "@" + input + ": missing activation policy");
        }
        WeightUse result;
        result.input = input;
        switch (*use.activation_policy) {
        case artifact::ActivationPolicy::A16Only:
            result.policy = ops::LinearPolicy::A16Only;
            break;
#if defined(NINFER_SM86)
        // Fork-local (rtx4090-port): the A8 (FP8) and A4 (NVFP4) routes load through TMA or use
        // E2M1 block-scale MMA, which sm_89 lacks. Those weights run their A16 routes instead.
        case artifact::ActivationPolicy::AllowA8:
        case artifact::ActivationPolicy::AllowA4:
            result.policy = ops::LinearPolicy::A16Only;
            break;
#else
        case artifact::ActivationPolicy::AllowA8:
            result.policy = ops::LinearPolicy::AllowA8;
            break;
        case artifact::ActivationPolicy::AllowA4:
            result.policy = ops::LinearPolicy::AllowA4;
            break;
#endif
        }
        if (groupwise) result.policy = ops::LinearPolicy::A16Only;
        for (const auto& [role, binding] : use.auxiliaries) {
            if (role != "activation_input_divisor") {
                throw artifact::ArtifactError(name + "@" + input + ": unknown auxiliary " + role);
            }
            const auto value = binder.values(binding, QType::FP32).scalar_f32();
            if (!std::isfinite(value) || value <= 0) {
                throw artifact::ArtifactError(name + "@" + input +
                                              ": activation divisor must be positive finite FP32");
            }
            result.activation_input_divisor = value;
        }
        pending.uses.push_back(std::move(result));
    }
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(
            artifact::object_id(binder.reader().directory().object(part.object)));
    }
    const WeightId id{weights.size()};
    parameters_.emplace(std::move(name), id);
    weights.push_back(std::move(pending));
    return id;
}

WeightId Bindings::direct(std::string name, artifact::Shape shape, QType format) {
    return parameter(std::move(name), std::move(shape), {}, format);
}

std::vector<BoundWeight> resolve_weights(std::vector<PendingWeight>&& pending,
                                         const artifact::MaterializedArtifact& materialized,
                                         std::span<const Device> devices) {
    std::vector<BoundWeight> out;
    out.reserve(pending.size());
    for (std::size_t i = 0; i < pending.size(); ++i) {
        auto& item = pending[i];
        auto view  = artifact::bind_view(item.reference, materialized, devices[i]);
        out.push_back({std::move(item.reference.name), std::move(item.source_objects),
                       std::move(view), std::move(item.uses)});
    }
    return out;
}

} // namespace ninfer::models
