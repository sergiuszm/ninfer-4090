#include "models/qwen3_5/execution/dense_mlp.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

std::size_t dense_mlp_workspace_bytes(const DenseProfile& parameters, std::int32_t first,
                                      std::int32_t last, DenseMlpUse use,
                                      BodyActivations activations) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("Dense MLP: invalid column interval");
    }
    const auto gu   = activations(parameters.gate_up);
    const auto down = activations(parameters.down);
    WorkspaceLayoutBuilder layout;
    if (use == DenseMlpUse::Mtp) {
        (void)layout.alloc(DType::BF16, {gu.n, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(
                ops::linear_workspace_capacity_bytes(gu.qtype, gu.n, gu.k, gu.policy, first, last));
        }
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        (void)layout.alloc(DType::BF16, {down.n, last});
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(down.qtype, down.n, down.k,
                                                                      down.policy, first, last));
    } else {
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, gu.policy, first, last));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, down.policy, first, last));
        }
    }
    return layout.peak_bytes(1);
}

void dense_mlp(const Tensor& hidden, const DenseParameters& parameters, Tensor& residual,
               DenseMlpUse use, BodyActivations activations, MemoryArena& workspace,
               CudaExecutionView execution) {
    auto scope                = workspace.scope();
    const auto columns        = hidden.ne[1];
    const auto& gu            = parameters.gate_up.weight;
    const auto& down          = parameters.down.weight;
    const auto gate_up_policy = activations.policy(gu.qtype, parameters.gate_up.policy);
    const auto down_policy    = activations.policy(down.qtype, parameters.down.policy);
    if (use == DenseMlpUse::Mtp) {
        Tensor gate_up = workspace.alloc(DType::BF16, {gu.n, columns});
        {
            auto call = workspace.scope();
            ops::linear(hidden, gu, gate_up, gate_up_policy, workspace, execution);
        }
        Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
        ops::silu_mul(gate_up.slice(0, 0, gu.n / 2), gate_up.slice(0, gu.n / 2, gu.n / 2),
                      activation, execution);
        Tensor delta = workspace.alloc(DType::BF16, {down.n, columns});
        ops::linear(activation, down, delta, down_policy, workspace, execution);
        ops::residual_add(delta, residual, execution);
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
    {
        auto call = workspace.scope();
        ops::linear_swiglu(hidden, gu, activation, gate_up_policy, workspace, execution);
    }
    ops::linear_add(activation, down, residual, down_policy, workspace, execution);
}

} // namespace ninfer::models::qwen3_5::execution
