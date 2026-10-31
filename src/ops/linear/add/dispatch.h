#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/linear/kernels/int8/int8_proj_launch.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops::detail {

using LinearAddLaunch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

LinearAddLaunch select_bf16_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens);
LinearAddLaunch select_q4_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens);
LinearAddLaunch select_q5_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens);
LinearAddLaunch select_q8_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens);
LinearAddLaunch select_q5_linear_add_n5120_k6144(std::int32_t tokens);
LinearAddLaunch select_q5_linear_add_n5120_k17408(std::int32_t tokens);
LinearAddLaunch select_q8_linear_add_n2048_k4096(std::int32_t tokens);
LinearAddLaunch select_q8_linear_add_n2048_k6144(std::int32_t tokens);
LinearAddLaunch select_q8_linear_add_n5120_k6144(std::int32_t tokens);
LinearAddLaunch select_q8_linear_add_n5120_k17408(std::int32_t tokens);

// Q5 under an A8 permission: group-64 INT8 activations at every token extent.
void q5_linear_add_int8(const Tensor& x, const Weight& w, Tensor& residual,
                        Int8ProjWorkspace scratch, cudaStream_t stream);

} // namespace ninfer::ops::detail
