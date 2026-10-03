#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Caller-owned staging for the INT8 prefill route: the group-64 quantized
// activation codes and their FP32 group scales. Both live in the Op workspace
// for the duration of one call and are never observable outside it.
struct Q4LinearSwiGluInt8Workspace {
    std::int8_t* codes = nullptr;
    float* scales      = nullptr;
};

// The INT8 GEMM contracts whole 256-element weight super-groups, so K is padded
// to that boundary rather than to the 128 the BF16 routes use.
constexpr std::int32_t q4_linear_swiglu_int8_padded_k(std::int32_t k) noexcept {
    return ((k + 255) / 256) * 256;
}

// The quantized activation staging is O(T), so the route processes tokens in
// bounded tiles and its workspace never scales with the context length. At
// K = 5120 a tile costs 5,440 bytes per token, so the cap holds the workspace at
// 21.25 MiB regardless of how large a prefill chunk the Engine forms.
inline constexpr std::int32_t kQ4LinearSwiGluInt8MaxTokenTile = 4096;

constexpr std::int32_t q4_linear_swiglu_int8_token_tile(std::int32_t cols) noexcept {
    return cols < kQ4LinearSwiGluInt8MaxTokenTile ? cols : kQ4LinearSwiGluInt8MaxTokenTile;
}

void q4_linear_swiglu_int8_folded_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         const Q4LinearSwiGluInt8Workspace& scratch,
                                         cudaStream_t stream);

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(const Tensor& x, const Weight& w,
                                                          Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c40_launch(const Tensor& x, const Weight& w,
                                                         Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c48_launch(const Tensor& x, const Weight& w,
                                                         Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_small_t_tiled_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream);

} // namespace ninfer::ops::detail
