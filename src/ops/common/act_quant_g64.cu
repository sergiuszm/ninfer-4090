// Adapted from tensorninja/ninfer-4090 (Apache-2.0); see docs/ada.md for provenance.

#include "ops/common/act_quant_g64.h"

#include "ops/common/int8_mma.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kActQuantWarpsPerCta = 8;
inline constexpr int kActQuantThreads     = kActQuantWarpsPerCta * 32;

// One warp owns one (token, 64-channel group). Two channels per lane keeps the
// BF16 load a full 128-byte warp transaction and the INT8 store a 64-byte one.
namespace {

__global__ __launch_bounds__(kActQuantThreads) void act_quant_g64_kernel(
    const __nv_bfloat16* __restrict__ x, std::int8_t* __restrict__ xq, float* __restrict__ xs,
    std::int32_t k, std::int32_t cols, std::int32_t padded_k, std::int32_t n_groups) {
    const int warp_id = (static_cast<int>(blockIdx.x) * kActQuantWarpsPerCta) +
                        (static_cast<int>(threadIdx.x) >> 5);
    const int lane    = static_cast<int>(threadIdx.x) & 31;
    if (warp_id >= n_groups * cols) { return; }

    const int col   = warp_id / n_groups;
    const int group = warp_id - col * n_groups;
    const int k0    = group * kInt8ActGroupK + 2 * lane;

    float v0 = 0.0f;
    float v1 = 0.0f;
    if (k0 + 1 < k) {
        const __nv_bfloat162 pair =
            *reinterpret_cast<const __nv_bfloat162*>(&x[static_cast<std::int64_t>(col) * k + k0]);
        v0 = __bfloat162float(pair.x);
        v1 = __bfloat162float(pair.y);
    } else if (k0 < k) {
        v0 = __bfloat162float(x[static_cast<std::int64_t>(col) * k + k0]);
    }

    float m = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off)); }

    const float scale = m > 0.0f ? m / kInt8ActMax : 0.0f;
    const float inv   = m > 0.0f ? kInt8ActMax / m : 0.0f;

    std::int8_t codes[2];
    codes[0] = static_cast<std::int8_t>(int8_quantize_rn(v0 * inv));
    codes[1] = static_cast<std::int8_t>(int8_quantize_rn(v1 * inv));
    *reinterpret_cast<std::uint16_t*>(&xq[static_cast<std::int64_t>(col) * padded_k + k0]) =
        *reinterpret_cast<const std::uint16_t*>(codes);

    if (lane == 0) { xs[static_cast<std::int64_t>(group) * cols + col] = scale; }
}

// Zero-fill the K padding once per launch. Group scales in the padded region are
// zero, so padded groups contribute exactly zero to the contraction.
__global__ __launch_bounds__(256) void act_quant_g64_pad_kernel(std::int8_t* __restrict__ xq,
                                                                float* __restrict__ xs,
                                                                std::int32_t k, std::int32_t cols,
                                                                std::int32_t padded_k,
                                                                std::int32_t n_groups) {
    const std::int64_t pad = static_cast<std::int64_t>(padded_k) - k;
    if (pad <= 0) { return; }
    const std::int64_t total = pad * cols;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t col       = i / pad;
        const std::int64_t off       = i - col * pad;
        xq[col * padded_k + k + off] = 0;
    }
    const int first_pad_group = k / kInt8ActGroupK;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < static_cast<std::int64_t>(n_groups - first_pad_group) * cols;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t g = first_pad_group + i / cols;
        const std::int64_t c = i % cols;
        if (g * kInt8ActGroupK >= k) { xs[g * cols + c] = 0.0f; }
    }
}

} // namespace

void act_quant_g64_launch(const __nv_bfloat16* x, std::int8_t* xq, float* xs, std::int32_t k,
                          std::int32_t cols, std::int32_t padded_k, cudaStream_t stream) {
    const std::int32_t n_groups = padded_k / kInt8ActGroupK;
    const int warps             = n_groups * cols;
    const int blocks            = (warps + kActQuantWarpsPerCta - 1) / kActQuantWarpsPerCta;
    act_quant_g64_kernel<<<blocks, kActQuantThreads, 0, stream>>>(x, xq, xs, k, cols, padded_k,
                                                                  n_groups);
    if (padded_k != k) {
        act_quant_g64_pad_kernel<<<256, 256, 0, stream>>>(xq, xs, k, cols, padded_k, n_groups);
    }
}

} // namespace ninfer::ops::detail
