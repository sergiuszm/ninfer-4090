#pragma once
#include "ops/linear/split/dispatch.h"
#include "ops/linear/kernels/int8/int8_proj_launch.h"

namespace ninfer::ops::detail::linear_split::q4_q5_n7168_n7168_k5120 {
void launch_small(const Tensor&, const Weight&, const Weight&, std::span<const Tensor>,
                  cudaStream_t);
void launch_matrix(const Tensor&, const Weight&, const Weight&, std::span<const Tensor>,
                   cudaStream_t);
void launch_int8(const Tensor&, const Weight&, const Weight&, std::span<const Tensor>,
                 Int8ProjWorkspace, cudaStream_t);
} // namespace ninfer::ops::detail::linear_split::q4_q5_n7168_n7168_k5120
