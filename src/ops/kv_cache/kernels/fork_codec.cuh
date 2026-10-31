#pragma once

// Fork-local packed / rotated mode extensions (rk8v4, rk4v4, rk4v4-e8, rk2v4-e8). K and V are
// optionally rotated per 64-group by an H64 transform before encoding; packed modes store two
// signed 4-bit codes per byte. The E8 lattice codecs live in e8_lattice.cuh and
// e8_root_codec.cuh; this header owns the shared index math and scalar helpers.

#include "ops/common/cuda/memory.cuh"
#include "ops/kv_cache/kernels/layout.cuh"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops {

// Packed int4 codes of one 256-wide head row: 128 code bytes per row in the record addressing of
// layout.cuh. `record` is the byte offset of the code plane, as for the upstream layouts.
__device__ __forceinline__ std::int64_t kv_cache_i4_code_index(std::int64_t record, int kv_head,
                                                               int packed_d, int page_offset) {
    return detail::kv::record_offset<detail::kv::kHeadDim / 2, 1>(record, kv_head, page_offset,
                                                                  packed_d);
}

__device__ __forceinline__ std::int8_t kv_cache_i4_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-7, min(7, q));
    return static_cast<std::int8_t>(q);
}

__device__ __forceinline__ std::uint8_t kv_cache_pack_i4(std::int8_t lo, std::int8_t hi) {
    return static_cast<std::uint8_t>((static_cast<unsigned>(lo) & 0x0fu) |
                                     ((static_cast<unsigned>(hi) & 0x0fu) << 4));
}

__device__ __forceinline__ std::int8_t kv_cache_unpack_i4(std::uint8_t packed, int high) {
    const unsigned nibble = high ? (packed >> 4) : (packed & 0x0fu);
    return static_cast<std::int8_t>(static_cast<int>(nibble ^ 8u) - 8);
}

__device__ __forceinline__ void kv_cache_hadamard64(float& x0, float& x1,
                                                    unsigned mask = 0xffffffffu) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float y0 = __shfl_xor_sync(mask, x0, offset);
        const float y1 = __shfl_xor_sync(mask, x1, offset);
        const bool hi  = (static_cast<int>(threadIdx.x) & offset) != 0;
        x0             = hi ? y0 - x0 : x0 + y0;
        x1             = hi ? y1 - x1 : x1 + y1;
    }
    const float a = x0;
    const float b = x1;
    x0            = (a + b) * 0.125f;
    x1            = (a - b) * 0.125f;
}

template <int QHeads>
__global__ void kv_cache_inverse_rotate_output_kernel(__nv_bfloat16* output, int width,
                                                      int full_width, int column_begin,
                                                      const std::int32_t* valid_columns) {
    const int unit = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (lane >= 32) { return; }
    const int group  = unit % detail::kv::kInt8Groups;
    const int tmp    = unit / detail::kv::kInt8Groups;
    const int q_head = tmp % QHeads;
    const int row    = tmp / QHeads;
    const int batch  = row / width;
    const int token  = row - batch * width;
    const int column = column_begin + token;
    if (token >= width || (valid_columns != nullptr && column >= valid_columns[batch])) { return; }
    const int d0            = group * detail::kv::kInt8Group + lane;
    const int d1            = d0 + 32;
    const std::int64_t base = static_cast<std::int64_t>(detail::kv::kHeadDim) *
                              (q_head + static_cast<std::int64_t>(QHeads) *
                                            (column + static_cast<std::int64_t>(full_width) * batch));
    float x0 = __bfloat162float(output[base + d0]);
    float x1 = __bfloat162float(output[base + d1]);
    kv_cache_hadamard64(x0, x1);
    output[base + d0] = __float2bfloat16(x0);
    output[base + d1] = __float2bfloat16(x1);
}

__device__ __forceinline__ void kv_cache_unpack_i4x16(const std::uint8_t* src8,
                                                      std::int8_t* dst16) {
    const std::uint64_t raw = common::load_vec<std::uint64_t>(src8);
    const auto* bytes       = reinterpret_cast<const std::uint8_t*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst16[2 * i]     = kv_cache_unpack_i4(bytes[i], 0);
        dst16[2 * i + 1] = kv_cache_unpack_i4(bytes[i], 1);
    }
}

} // namespace ninfer::ops
