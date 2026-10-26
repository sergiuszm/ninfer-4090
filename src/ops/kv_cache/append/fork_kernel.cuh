#pragma once

// Fork-local (rtx4090-port): append kernels for the fork KV storages (kv_storage_is_fork_mode).
// Keys and values are rotated per 64-dimension group (kv_cache_hadamard64), not by upstream's
// d256 Hadamard, so upstream Int8Group64 keeps its own kernels in kernel.cuh.

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/e8_lattice.cuh"
#include "ops/kernel/e8_root_codec.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Eight independent quantization units per CTA; one warp owns one
// (token, kv_head, 64-d group), with two dimensions per lane.
template <typename Geometry, bool PackedV, bool RotateK, bool RotateV, bool PackedK,
          bool E8Lattice = false, bool E8Root = false, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_fork_kernel(const __nv_bfloat16* __restrict__ k,
                                     const __nv_bfloat16* __restrict__ v,
                                     const std::int32_t* __restrict__ positions, Metadata metadata,
                                     std::int8_t* __restrict__ cache_k,
                                     std::uint8_t* __restrict__ cache_v,
                                     __half* __restrict__ scale_k, __half* __restrict__ scale_v,
                                     std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads * kKVCacheInt8Groups;
    if (unit >= units) { return; }

    const int group                 = unit % kKVCacheInt8Groups;
    const int tmp                   = unit / kKVCacheInt8Groups;
    const int kv_head               = tmp % Geometry::KVHeads;
    const int token                 = tmp / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int page                        = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    const int page_off              = position & kPagedKVPageMask;
    const int d0                    = group * kKVCacheInt8Group + lane;
    const int d1                    = d0 + 32;

    const std::int64_t src0 = kv_cache_int8_quant_src_index<Geometry>(kv_head, d0, token);
    const std::int64_t src1 = kv_cache_int8_quant_src_index<Geometry>(kv_head, d1, token);
    float k0                = __bfloat162float(k[src0]);
    float k1                = __bfloat162float(k[src1]);
    float v0                = __bfloat162float(v[src0]);
    float v1                = __bfloat162float(v[src1]);
    if constexpr (RotateK) { kv_cache_hadamard64(k0, k1, FullMask); }
    if constexpr (RotateV) { kv_cache_hadamard64(v0, v1, FullMask); }

    float k_abs = fmaxf(fabsf(k0), fabsf(k1));
    float v_abs = fmaxf(fabsf(v0), fabsf(v1));
    k_abs       = warp_max(k_abs, FullMask);
    v_abs       = warp_max(v_abs, FullMask);

    const __half ksh =
        __float2half_rn(k_abs > 0.0f ? k_abs / ((PackedK || E8Root) ? 7.0f : 127.0f) : 0.0f);
    const __half vsh = __float2half_rn(v_abs > 0.0f ? v_abs / (PackedV ? 7.0f : 127.0f) : 0.0f);
    const float ks   = __half2float(ksh);
    const float vs   = __half2float(vsh);
    const float kinv = ks > 0.0f ? 1.0f / ks : 0.0f;
    const float vinv = vs > 0.0f ? 1.0f / vs : 0.0f;
    page             = __shfl_sync(FullMask, page, 0);

    const std::int64_t code_base = kv_cache_int8_quant_code_index<Geometry>(
        page, kv_head, group * kKVCacheInt8Group, page_off);
    if constexpr (E8Root) {
        uint8_t c1_0, c2_0, c1_1, c2_1;
        e8_encode_cylinder_8d_warp(k0, ks, c1_0, c2_0, lane);
        e8_encode_cylinder_8d_warp(k1, ks, c1_1, c2_1, lane);
        if ((lane & 7) == 0) {
            int s0 = (lane / 8);
            int s1 = 4 + (lane / 8);
            const std::int64_t k_base =
                paged_kv_page_head_offset<64, Geometry::KVHeads>(page, kv_head) +
                static_cast<std::int64_t>(page_off) * 64 + group * 16;
            reinterpret_cast<std::uint8_t*>(cache_k)[k_base + s0 * 2 + 0] = c1_0;
            reinterpret_cast<std::uint8_t*>(cache_k)[k_base + s0 * 2 + 1] = c2_0;
            reinterpret_cast<std::uint8_t*>(cache_k)[k_base + s1 * 2 + 0] = c1_1;
            reinterpret_cast<std::uint8_t*>(cache_k)[k_base + s1 * 2 + 1] = c2_1;
        }
        const float v0_hi = __shfl_down_sync(FullMask, v0, 1);
        const float v1_hi = __shfl_down_sync(FullMask, v1, 1);
        if ((lane & 1) == 0) {
            cache_v[kv_cache_i4_code_index<Geometry>(page, kv_head, d0 / 2, page_off)] =
                kv_cache_pack_i4(kv_cache_i4_quant_code(v0, vinv),
                                 kv_cache_i4_quant_code(v0_hi, vinv));
            cache_v[kv_cache_i4_code_index<Geometry>(page, kv_head, d1 / 2, page_off)] =
                kv_cache_pack_i4(kv_cache_i4_quant_code(v1, vinv),
                                 kv_cache_i4_quant_code(v1_hi, vinv));
        }
    } else if constexpr (PackedK) {
        std::int8_t c0 = 0, c1 = 0;
        if constexpr (E8Lattice) {
            float k0_scaled = k0 * kinv;
            float k1_scaled = k1 * kinv;
            e8_project_8d_warp(k0_scaled, k1_scaled, lane);
            // NOTE: same deliberate half-coset approximation as the rk4v4-e8 decode path
            // (see small_t_i8.cuh): the D8+0.5 E8 coset is collapsed by the
            // rintf()+cast below and never reconstructed, since no coset bit exists in the
            // packed i4/int8 codes. The rk4v4 (non-E8) path is unaffected.
            int q0 = static_cast<int>(rintf(k0_scaled));
            int q1 = static_cast<int>(rintf(k1_scaled));
            c0     = static_cast<std::int8_t>(max(-8, min(7, q0)));
            c1     = static_cast<std::int8_t>(max(-8, min(7, q1)));
        } else {
            c0 = kv_cache_i4_quant_code(k0, kinv);
            c1 = kv_cache_i4_quant_code(k1, kinv);
        }
        const std::int8_t c0_hi =
            static_cast<std::int8_t>(__shfl_down_sync(FullMask, static_cast<int>(c0), 1));
        const std::int8_t c1_hi =
            static_cast<std::int8_t>(__shfl_down_sync(FullMask, static_cast<int>(c1), 1));
        const float v0_hi = __shfl_down_sync(FullMask, v0, 1);
        const float v1_hi = __shfl_down_sync(FullMask, v1, 1);
        if ((lane & 1) == 0) {
            reinterpret_cast<std::uint8_t*>(
                cache_k)[kv_cache_i4_code_index<Geometry>(page, kv_head, d0 / 2, page_off)] =
                kv_cache_pack_i4(c0, c0_hi);
            reinterpret_cast<std::uint8_t*>(
                cache_k)[kv_cache_i4_code_index<Geometry>(page, kv_head, d1 / 2, page_off)] =
                kv_cache_pack_i4(c1, c1_hi);
            cache_v[kv_cache_i4_code_index<Geometry>(page, kv_head, d0 / 2, page_off)] =
                kv_cache_pack_i4(kv_cache_i4_quant_code(v0, vinv),
                                 kv_cache_i4_quant_code(v0_hi, vinv));
            cache_v[kv_cache_i4_code_index<Geometry>(page, kv_head, d1 / 2, page_off)] =
                kv_cache_pack_i4(kv_cache_i4_quant_code(v1, vinv),
                                 kv_cache_i4_quant_code(v1_hi, vinv));
        }
    } else {
        cache_k[code_base + lane]      = kv_cache_int8_quant_code(k0, kinv);
        cache_k[code_base + lane + 32] = kv_cache_int8_quant_code(k1, kinv);
        if constexpr (PackedV) {
            const float v0_hi = __shfl_down_sync(FullMask, v0, 1);
            const float v1_hi = __shfl_down_sync(FullMask, v1, 1);
            if ((lane & 1) == 0) {
                cache_v[kv_cache_i4_code_index<Geometry>(page, kv_head, d0 / 2, page_off)] =
                    kv_cache_pack_i4(kv_cache_i4_quant_code(v0, vinv),
                                     kv_cache_i4_quant_code(v0_hi, vinv));
                cache_v[kv_cache_i4_code_index<Geometry>(page, kv_head, d1 / 2, page_off)] =
                    kv_cache_pack_i4(kv_cache_i4_quant_code(v1, vinv),
                                     kv_cache_i4_quant_code(v1_hi, vinv));
            }
        } else {
            auto* cache_v_i8                  = reinterpret_cast<std::int8_t*>(cache_v);
            cache_v_i8[code_base + lane]      = kv_cache_int8_quant_code(v0, vinv);
            cache_v_i8[code_base + lane + 32] = kv_cache_int8_quant_code(v1, vinv);
        }
    }
    if (lane == 0) {
        const std::int64_t scale_off =
            kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, group, page_off);
        scale_k[scale_off] = ksh;
        scale_v[scale_off] = vsh;
    }
}

// Large appends are scheduled in absolute eight-token tiles. Eight divides P=64, so each CTA is
// page-local while an unknown base offset costs at most one empty tail CTA in the launch envelope.
template <typename Geometry, bool PackedV, bool RotateK, bool RotateV, bool PackedK,
          bool E8Lattice = false, bool E8Root = false, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_fork_page_kernel(const __nv_bfloat16* __restrict__ k,
                                          const __nv_bfloat16* __restrict__ v,
                                          const std::int32_t* __restrict__ positions,
                                          Metadata metadata, std::int8_t* __restrict__ cache_k,
                                          std::uint8_t* __restrict__ cache_v,
                                          __half* __restrict__ scale_k,
                                          __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int group             = static_cast<int>(blockIdx.z);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) { return; }

    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;

    const int token  = token_begin + warp;
    const bool valid = token < token_end;
    const int d0     = group * kKVCacheInt8Group + lane;
    const int d1     = d0 + 32;
    float k0 = 0.0f, k1 = 0.0f, v0 = 0.0f, v1 = 0.0f;
    if (valid) {
        const std::int64_t src0 = kv_cache_int8_quant_src_index<Geometry>(kv_head, d0, token);
        const std::int64_t src1 = kv_cache_int8_quant_src_index<Geometry>(kv_head, d1, token);
        k0                      = __bfloat162float(k[src0]);
        k1                      = __bfloat162float(k[src1]);
        v0                      = __bfloat162float(v[src0]);
        v1                      = __bfloat162float(v[src1]);
        if constexpr (RotateK) { kv_cache_hadamard64(k0, k1, FullMask); }
        if constexpr (RotateV) { kv_cache_hadamard64(v0, v1, FullMask); }
    }
    const float k_abs = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
    const float v_abs = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
    const __half ksh =
        __float2half_rn(k_abs > 0.0f ? k_abs / ((PackedK || E8Root) ? 7.0f : 127.0f) : 0.0f);
    const __half vsh = __float2half_rn(v_abs > 0.0f ? v_abs / (PackedV ? 7.0f : 127.0f) : 0.0f);
    const float ks   = __half2float(ksh);
    const float vs   = __half2float(vsh);
    const float kinv = ks > 0.0f ? 1.0f / ks : 0.0f;
    const float vinv = vs > 0.0f ? 1.0f / vs : 0.0f;
    physical_page    = __shfl_sync(FullMask, physical_page, 0);
    if (!valid) { return; }

    const int position = base_position + token;
    const int page_off = position & kPagedKVPageMask;
    const std::int64_t code_base =
        paged_kv_page_head_offset<kKVCacheInt8HeadDim, Geometry::KVHeads>(physical_page, kv_head) +
        static_cast<std::int64_t>(page_off) * kKVCacheInt8HeadDim + group * kKVCacheInt8Group;
    if constexpr (E8Root) {
        uint8_t c1_0, c2_0, c1_1, c2_1;
        e8_encode_cylinder_8d_warp(k0, ks, c1_0, c2_0, lane);
        e8_encode_cylinder_8d_warp(k1, ks, c1_1, c2_1, lane);
        if ((lane & 7) == 0) {
            int s0 = (lane / 8);
            int s1 = 4 + (lane / 8);
            const std::int64_t k_base =
                paged_kv_page_head_offset<64, Geometry::KVHeads>(physical_page, kv_head) +
                static_cast<std::int64_t>(page_off) * 64 + group * 16;
            reinterpret_cast<std::uint8_t*>(cache_k)[k_base + s0 * 2 + 0] = c1_0;
            reinterpret_cast<std::uint8_t*>(cache_k)[k_base + s0 * 2 + 1] = c2_0;
            reinterpret_cast<std::uint8_t*>(cache_k)[k_base + s1 * 2 + 0] = c1_1;
            reinterpret_cast<std::uint8_t*>(cache_k)[k_base + s1 * 2 + 1] = c2_1;
        }
        const float v0_hi = __shfl_down_sync(FullMask, v0, 1);
        const float v1_hi = __shfl_down_sync(FullMask, v1, 1);
        if ((lane & 1) == 0) {
            cache_v[kv_cache_i4_code_index<Geometry>(physical_page, kv_head, d0 / 2, page_off)] =
                kv_cache_pack_i4(kv_cache_i4_quant_code(v0, vinv),
                                 kv_cache_i4_quant_code(v0_hi, vinv));
            cache_v[kv_cache_i4_code_index<Geometry>(physical_page, kv_head, d1 / 2, page_off)] =
                kv_cache_pack_i4(kv_cache_i4_quant_code(v1, vinv),
                                 kv_cache_i4_quant_code(v1_hi, vinv));
        }
    } else if constexpr (PackedK) {
        std::int8_t c0 = 0, c1 = 0;
        if constexpr (E8Lattice) {
            float k0_scaled = k0 * kinv;
            float k1_scaled = k1 * kinv;
            e8_project_8d_warp(k0_scaled, k1_scaled, lane);
            // NOTE: same deliberate half-coset approximation as the rk4v4-e8 decode path
            // (see small_t_i8.cuh): the D8+0.5 E8 coset is collapsed by the
            // rintf()+cast below and never reconstructed, since no coset bit exists in the
            // packed i4/int8 codes. The rk4v4 (non-E8) path is unaffected.
            int q0 = static_cast<int>(rintf(k0_scaled));
            int q1 = static_cast<int>(rintf(k1_scaled));
            c0     = static_cast<std::int8_t>(max(-8, min(7, q0)));
            c1     = static_cast<std::int8_t>(max(-8, min(7, q1)));
        } else {
            c0 = kv_cache_i4_quant_code(k0, kinv);
            c1 = kv_cache_i4_quant_code(k1, kinv);
        }
        const std::int8_t c0_hi =
            static_cast<std::int8_t>(__shfl_down_sync(FullMask, static_cast<int>(c0), 1));
        const std::int8_t c1_hi =
            static_cast<std::int8_t>(__shfl_down_sync(FullMask, static_cast<int>(c1), 1));
        const float v0_hi = __shfl_down_sync(FullMask, v0, 1);
        const float v1_hi = __shfl_down_sync(FullMask, v1, 1);
        if ((lane & 1) == 0) {
            reinterpret_cast<std::uint8_t*>(cache_k)[kv_cache_i4_code_index<Geometry>(
                physical_page, kv_head, d0 / 2, page_off)] = kv_cache_pack_i4(c0, c0_hi);
            reinterpret_cast<std::uint8_t*>(cache_k)[kv_cache_i4_code_index<Geometry>(
                physical_page, kv_head, d1 / 2, page_off)] = kv_cache_pack_i4(c1, c1_hi);
            cache_v[kv_cache_i4_code_index<Geometry>(physical_page, kv_head, d0 / 2, page_off)] =
                kv_cache_pack_i4(kv_cache_i4_quant_code(v0, vinv),
                                 kv_cache_i4_quant_code(v0_hi, vinv));
            cache_v[kv_cache_i4_code_index<Geometry>(physical_page, kv_head, d1 / 2, page_off)] =
                kv_cache_pack_i4(kv_cache_i4_quant_code(v1, vinv),
                                 kv_cache_i4_quant_code(v1_hi, vinv));
        }
    } else {
        cache_k[code_base + lane]      = kv_cache_int8_quant_code(k0, kinv);
        cache_k[code_base + lane + 32] = kv_cache_int8_quant_code(k1, kinv);
        if constexpr (PackedV) {
            const float v0_hi = __shfl_down_sync(FullMask, v0, 1);
            const float v1_hi = __shfl_down_sync(FullMask, v1, 1);
            if ((lane & 1) == 0) {
                cache_v[kv_cache_i4_code_index<Geometry>(physical_page, kv_head, d0 / 2,
                                                         page_off)] =
                    kv_cache_pack_i4(kv_cache_i4_quant_code(v0, vinv),
                                     kv_cache_i4_quant_code(v0_hi, vinv));
                cache_v[kv_cache_i4_code_index<Geometry>(physical_page, kv_head, d1 / 2,
                                                         page_off)] =
                    kv_cache_pack_i4(kv_cache_i4_quant_code(v1, vinv),
                                     kv_cache_i4_quant_code(v1_hi, vinv));
            }
        } else {
            auto* cache_v_i8                  = reinterpret_cast<std::int8_t*>(cache_v);
            cache_v_i8[code_base + lane]      = kv_cache_int8_quant_code(v0, vinv);
            cache_v_i8[code_base + lane + 32] = kv_cache_int8_quant_code(v1, vinv);
        }
    }
    if (lane == 0) {
        const std::int64_t scale_offset =
            paged_kv_page_head_offset<kKVCacheInt8Groups, Geometry::KVHeads>(physical_page,
                                                                             kv_head) +
            static_cast<std::int64_t>(page_off) * kKVCacheInt8Groups + group;
        scale_k[scale_offset] = ksh;
        scale_v[scale_offset] = vsh;
    }
}

} // namespace ninfer::ops
