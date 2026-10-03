#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Caller-owned staging for the INT8 prefill route: the group-64 quantized
// activation codes and their FP32 group scales. Both live in the Op workspace
// for the duration of one call and are never observable outside it.
struct Q5LinearAddInt8Workspace {
    std::int8_t* codes = nullptr;
    float* scales      = nullptr;
};

// The INT8 GEMM contracts whole 256-element weight super-groups, so K is padded
// to that boundary rather than to the 128 the BF16 routes use.
constexpr std::int32_t q5_linear_add_int8_padded_k(std::int32_t k) noexcept {
    return ((k + 255) / 256) * 256;
}

// The quantized activation staging is O(T), so the route processes tokens in
// bounded tiles and its workspace never scales with the context length. At
// K = 17408 a tile costs 18,496 bytes per token, holding the workspace at
// 72.25 MiB regardless of the prefill chunk the Engine forms.
inline constexpr std::int32_t kQ5LinearAddInt8MaxTokenTile = 4096;

constexpr std::int32_t q5_linear_add_int8_token_tile(std::int32_t cols) noexcept {
    return cols < kQ5LinearAddInt8MaxTokenTile ? cols : kQ5LinearAddInt8MaxTokenTile;
}

void q5_linear_add_int8_residual_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        const Q5LinearAddInt8Workspace& scratch,
                                        cudaStream_t stream);

void q5_linear_add_gemv_residual_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream);
void q5_linear_add_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);
void q5_linear_add_mma_r64_c16_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c24_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c32_s3_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                         cudaStream_t stream);
void q5_linear_add_mma_r64_c32_s4_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                         cudaStream_t stream);
void q5_linear_add_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);

} // namespace ninfer::ops::detail
