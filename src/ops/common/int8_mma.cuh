// Adapted from tensorninja/ninfer-4090 (Apache-2.0); see docs/ada.md for provenance.

#pragma once

// INT8 Tensor Core primitives for large-token (prefill) quantized GEMM on sm_89.
//
// Q4G64 codes in [-8,7] and Q5G64 codes in [-16,15] fit INT8 exactly.
// Packed integer conversion adds no weight quantization. Each group contracts
// through mma.sync.m16n8k32.s32.s8.s8.s32 before FP32 rescaling.

#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ void mma_s8_zero(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                            unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                            unsigned b1) {
    c0 = c1 = c2 = c3 = 0;
    mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
}

// Activation quantization group. Matches Q4G64/Q5G64 so one activation scale
// spans exactly the K extent of one weight group.
inline constexpr int kInt8ActGroupK = 64;

// Symmetric INT8 range. -128 is excluded so negation is exact and the decode
// stays symmetric with the weight codes.
inline constexpr float kInt8ActMax = 127.0f;

// Spread four consecutive stored codes into one nibble per byte lane, in K order.
//
// Storage packs code 2j in the low nibble of byte j and code 2j+1 in the high
// nibble, so two consecutive bytes carry exactly the four codes an A-fragment
// register needs. __byte_perm replicates the pair to {b0, b0, b1, b1}; the two
// masked selections then place n0..n3 in byte lanes 0..3.
__device__ __forceinline__ std::uint32_t int8_spread_code_nibbles(std::uint32_t packed_pair) {
    const std::uint32_t d = __byte_perm(packed_pair, 0u, 0x1100u);
    return (d & 0x000F000Fu) | ((d >> 4) & 0x0F000F00u);
}

// Sign-extend a 4-bit code in each byte lane to int8.
// (x & 0x08) * 0x1E == 0xF0 exactly when the code is negative, and 0x08 * 0x1E
// fits one byte, so the multiply never carries between lanes.
__device__ __forceinline__ std::uint32_t int8_sign_extend_4(std::uint32_t x) {
    return x | ((x & 0x08080808u) * 0x1Eu);
}

// Sign-extend a 5-bit code in each byte lane to int8. 0x10 * 0x0E == 0xE0.
__device__ __forceinline__ std::uint32_t int8_sign_extend_5(std::uint32_t x) {
    return x | ((x & 0x10101010u) * 0x0Eu);
}

// Place four consecutive Q5 high bits at bit 4 of byte lanes 0..3.
//
// The high plane is a lane-major bitstream: the high bit of code c is
// high[c >> 3] bit (c & 7). Four consecutive codes therefore occupy four
// consecutive bits of one byte, starting at bit 0 or bit 4. The multiply moves
// bit i to bit 8i + 4; the 16 partial products land on 16 distinct bit
// positions, so nothing carries and the mask keeps only the four wanted bits.
__device__ __forceinline__ std::uint32_t int8_spread_high_bits(std::uint32_t high_byte,
                                                               int bit_offset) {
    const std::uint32_t h = (high_byte >> bit_offset) & 0xFu;
    return (h * 0x02040810u) & 0x10101010u;
}

// Four consecutive Q4G64 codes -> one MMA A-fragment register (4 signed bytes).
__device__ __forceinline__ std::uint32_t q4_codes_to_s8x4(std::uint32_t packed_pair) {
    return int8_sign_extend_4(int8_spread_code_nibbles(packed_pair));
}

// Four consecutive Q5G64 codes -> one MMA A-fragment register (4 signed bytes).
// `bit_offset` is (first_code_index & 7); it is 0 or 4 for every A-fragment.
__device__ __forceinline__ std::uint32_t q5_codes_to_s8x4(std::uint32_t packed_pair,
                                                          std::uint32_t high_byte, int bit_offset) {
    const std::uint32_t nibbles = int8_spread_code_nibbles(packed_pair);
    return int8_sign_extend_5(nibbles | int8_spread_high_bits(high_byte, bit_offset));
}

// Round-to-nearest-even float -> int8 with saturation, matching the quantizer
// oracle. cvt.rni.s8.f32 saturates, so no explicit clamp is needed.
__device__ __forceinline__ int int8_quantize_rn(float v) {
    int r;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(r) : "f"(v));
    return r;
}

} // namespace ninfer::ops::detail
