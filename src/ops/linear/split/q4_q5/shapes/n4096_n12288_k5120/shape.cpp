#include "ops/linear/split/q4_q5/shapes/n4096_n12288_k5120/launch.h"

namespace ninfer::ops::detail::linear_split::q4_q5_n4096_n12288_k5120 {
namespace {
// A permission for A8 selects group-64 INT8 activations at every token extent, so a prefill
// suffix whose width changes with prefix reuse never switches activation precision.
std::size_t capacity(LinearPolicy policy, std::int32_t, std::int32_t last) {
    return allows_a8(policy) ? int8_proj_workspace_capacity_bytes(5120, last) : 0;
}

void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
            MemoryArena* workspace, cudaStream_t stream) {
    if (allows_a8(weights.policy)) {
        const auto scratch = allocate_int8_proj_workspace(*workspace, x.ne[0], x.ne[1]);
        launch_int8(x, weights.first, *weights.second, outputs, scratch, stream);
        return;
    }
    if (x.ne[1] <= 12)
        launch_small(x, weights.first, *weights.second, outputs, stream);
    else
        launch_matrix(x, weights.first, *weights.second, outputs, stream);
}
} // namespace
} // namespace ninfer::ops::detail::linear_split::q4_q5_n4096_n12288_k5120

namespace ninfer::ops::detail::linear_split {
const SplitShape kQ4Q5SplitN4096N12288K5120{
    4096, 5120, kSplitRows<10240, 6144>, q4_q5_n4096_n12288_k5120::launch,
    q4_q5_n4096_n12288_k5120::capacity, 12288};
} // namespace ninfer::ops::detail::linear_split
