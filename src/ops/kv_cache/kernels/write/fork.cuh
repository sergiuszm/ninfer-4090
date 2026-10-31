#pragma once

// Fork-local (rtx4090-port): writes for the fork KV storages (kv_storage_is_fork_mode). Keys and
// values are rotated per 64-feature group (kv_cache_hadamard64), not by upstream's D256 Hadamard,
// so upstream Int8Group64 keeps its own writer in int8.cuh.

#include "core/paged_kv_storage.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/quant/integer.cuh"
#include "ops/kv_cache/kernels/e8_lattice.cuh"
#include "ops/kv_cache/kernels/e8_root_codec.cuh"
#include "ops/kv_cache/kernels/fork_codec.cuh"
#include "ops/kv_cache/kernels/layout.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail::kv {

// Compile-time codec of one fork storage, shared by the writers and the causal_cache/e8/ readers.
template <KvCacheStorage Storage>
struct ForkCodec {
    static constexpr KvForkModeFlags flags = kv_fork_mode_flags(Storage);
    static_assert(flags.packed_v && flags.rotate_k && flags.rotate_v,
                  "fork writes rotate both vectors and pack values to INT4");
    static constexpr bool packed_key = flags.packed_k; // INT4 or E8 lattice key codes
    static constexpr bool e8_lattice = flags.e8_lattice;
    static constexpr bool e8_root    = flags.e8_root;
    static constexpr float key_levels = packed_key || e8_root ? 7.0f : 127.0f;
};

// One warp encodes one (token, kv_head, 64-feature group); lane l carries features l and l + 32.
// `record` is valid in lane 0 only and is broadcast after the scale reduction. The standalone
// writers and the fused small-T append share this encoder, so both write identical bytes.
template <typename Geometry, KvCacheStorage Storage>
__device__ __forceinline__ void
write_fork_group(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                 int token, int kv_head, int group, std::int64_t record, int page_off, int lane,
                 std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
                 __half* __restrict__ scale_k, __half* __restrict__ scale_v) {
    using Codec                 = ForkCodec<Storage>;
    constexpr unsigned FullMask = 0xffffffffu;
    const int d0                = group * kInt8Group + lane;
    const int d1                = d0 + 32;
    const std::int64_t src0     = source_offset<Geometry>(kv_head, d0, token);
    const std::int64_t src1     = source_offset<Geometry>(kv_head, d1, token);
    float k0                    = __bfloat162float(k[src0]);
    float k1                    = __bfloat162float(k[src1]);
    float v0                    = __bfloat162float(v[src0]);
    float v1                    = __bfloat162float(v[src1]);
    kv_cache_hadamard64(k0, k1, FullMask);
    kv_cache_hadamard64(v0, v1, FullMask);

    const float k_abs = common::warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
    const float v_abs = common::warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
    const __half ksh  = __float2half_rn(k_abs > 0.0f ? k_abs / Codec::key_levels : 0.0f);
    const __half vsh  = __float2half_rn(v_abs > 0.0f ? v_abs / 7.0f : 0.0f);
    const float ks    = __half2float(ksh);
    const float vs    = __half2float(vsh);
    const float kinv  = ks > 0.0f ? 1.0f / ks : 0.0f;
    const float vinv  = vs > 0.0f ? 1.0f / vs : 0.0f;
    record            = __shfl_sync(FullMask, record, 0);

    if constexpr (Codec::e8_root) {
        std::uint8_t c1_0, c2_0, c1_1, c2_1;
        e8_encode_cylinder_8d_warp(k0, ks, c1_0, c2_0, lane);
        e8_encode_cylinder_8d_warp(k1, ks, c1_1, c2_1, lane);
        if ((lane & 7) == 0) {
            const int s0 = lane / 8;
            const int s1 = 4 + lane / 8;
            const std::int64_t k_base =
                E8RootG64Layout::code_offset(record, kv_head, group * kInt8Group, page_off);
            cache_k[k_base + s0 * 2 + 0] = c1_0;
            cache_k[k_base + s0 * 2 + 1] = c2_0;
            cache_k[k_base + s1 * 2 + 0] = c1_1;
            cache_k[k_base + s1 * 2 + 1] = c2_1;
        }
    } else if constexpr (Codec::packed_key) {
        std::int8_t c0 = 0, c1 = 0;
        if constexpr (Codec::e8_lattice) {
            float k0_scaled = k0 * kinv;
            float k1_scaled = k1 * kinv;
            e8_project_8d_warp(k0_scaled, k1_scaled, lane);
            // NOTE: same deliberate half-coset approximation as the rk4v4-e8 decode path
            // (see small_t_i8.cuh): the D8+0.5 E8 coset is collapsed by the
            // rintf()+cast below and never reconstructed, since no coset bit exists in the
            // packed i4/int8 codes. The rk4v4 (non-E8) path is unaffected.
            const int q0 = static_cast<int>(rintf(k0_scaled));
            const int q1 = static_cast<int>(rintf(k1_scaled));
            c0           = static_cast<std::int8_t>(max(-8, min(7, q0)));
            c1           = static_cast<std::int8_t>(max(-8, min(7, q1)));
        } else {
            c0 = kv_cache_i4_quant_code(k0, kinv);
            c1 = kv_cache_i4_quant_code(k1, kinv);
        }
        const auto c0_hi =
            static_cast<std::int8_t>(__shfl_down_sync(FullMask, static_cast<int>(c0), 1));
        const auto c1_hi =
            static_cast<std::int8_t>(__shfl_down_sync(FullMask, static_cast<int>(c1), 1));
        if ((lane & 1) == 0) {
            cache_k[I4G64Layout::code_offset(record, kv_head, d0, page_off)] =
                kv_cache_pack_i4(c0, c0_hi);
            cache_k[I4G64Layout::code_offset(record, kv_head, d1, page_off)] =
                kv_cache_pack_i4(c1, c1_hi);
        }
    } else {
        auto* codes = reinterpret_cast<std::int8_t*>(cache_k);
        const std::int64_t code_base =
            Int8G64Layout::code_offset(record, kv_head, group * kInt8Group, page_off);
        codes[code_base + lane]      = common::int8_quant_code(k0, kinv);
        codes[code_base + lane + 32] = common::int8_quant_code(k1, kinv);
    }

    const float v0_hi = __shfl_down_sync(FullMask, v0, 1);
    const float v1_hi = __shfl_down_sync(FullMask, v1, 1);
    if ((lane & 1) == 0) {
        cache_v[I4G64Layout::code_offset(record, kv_head, d0, page_off)] = kv_cache_pack_i4(
            kv_cache_i4_quant_code(v0, vinv), kv_cache_i4_quant_code(v0_hi, vinv));
        cache_v[I4G64Layout::code_offset(record, kv_head, d1, page_off)] = kv_cache_pack_i4(
            kv_cache_i4_quant_code(v1, vinv), kv_cache_i4_quant_code(v1_hi, vinv));
    }
    if (lane == 0) {
        const std::int64_t scale_off = Int8G64Layout::scale_offset(record, kv_head, group, page_off);
        scale_k[scale_off]           = ksh;
        scale_v[scale_off]           = vsh;
    }
}

// Eight independent (token, kv_head, group) units per CTA. MultiBatch takes the batch row from
// blockIdx.z; a row with no valid tokens reads no positions, table rows or K/V.
template <typename Geometry, KvCacheStorage Storage, typename Metadata, bool MultiBatch = false>
__launch_bounds__(256) __global__
    void write_fork_kernel(const __nv_bfloat16* __restrict__ k,
                           const __nv_bfloat16* __restrict__ v,
                           const std::int32_t* __restrict__ positions, Metadata metadata,
                           std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
                           __half* __restrict__ scale_k, __half* __restrict__ scale_v,
                           std::int32_t width) {
    if constexpr (MultiBatch) {
        const int batch = blockIdx.z;
        metadata.table_rows += batch;
        if (metadata.valid_columns) metadata.valid_columns += batch;
        const auto offset =
            static_cast<std::int64_t>(batch) * width * kHeadDim * Geometry::KVHeads;
        k += offset;
        v += offset;
        positions += static_cast<std::int64_t>(batch) * width;
    }
    constexpr int Warps = 8;
    const int tokens    = metadata.valid_tokens(width);
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int unit      = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units     = tokens * Geometry::KVHeads * kInt8Groups;
    if (unit >= units) return;

    const int group                 = unit % kInt8Groups;
    const int tmp                   = unit / kInt8Groups;
    const int kv_head               = tmp % Geometry::KVHeads;
    const int token                 = tmp / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int64_t* block_table = metadata.block_table();
    const std::int64_t record       = lane == 0 ? lookup_page(block_table, position) : 0;
    write_fork_group<Geometry, Storage>(k, v, token, kv_head, group, record, position & kPageMask,
                                        lane, cache_k, cache_v, scale_k, scale_v);
}

// Large single-row appends are scheduled in absolute eight-token tiles. Eight divides P=64, so each
// CTA is page-local while an unknown base offset costs at most one empty tail CTA in the launch
// envelope. A zero valid count reads no positions or table rows.
template <typename Geometry, KvCacheStorage Storage, typename Metadata>
__launch_bounds__(256) __global__
    void write_fork_page_kernel(const __nv_bfloat16* __restrict__ k,
                                const __nv_bfloat16* __restrict__ v,
                                const std::int32_t* __restrict__ positions, Metadata metadata,
                                std::uint8_t* __restrict__ cache_k,
                                std::uint8_t* __restrict__ cache_v, __half* __restrict__ scale_k,
                                __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    const int tokens            = metadata.valid_tokens(width);
    if (tokens == 0) return;
    const int warp          = static_cast<int>(threadIdx.x) >> 5;
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int kv_head       = static_cast<int>(blockIdx.y);
    const int group         = static_cast<int>(blockIdx.z);
    const int tile_delta    = static_cast<int>(blockIdx.x);
    const int base_position = positions[0];
    const int tile_position = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page  = tile_position >> kPageShift;
    const int token_begin   = max(0, tile_position - base_position);
    const int token_end     = min(tokens, tile_position + TokensPerTile - base_position);
    const int token         = token_begin + warp;
    if (token >= token_end) return;

    const std::int64_t* block_table = metadata.block_table();
    const std::int64_t record       = lane == 0 ? block_table[logical_page] : 0;
    write_fork_group<Geometry, Storage>(k, v, token, kv_head, group, record,
                                        (base_position + token) & kPageMask, lane, cache_k,
                                        cache_v, scale_k, scale_v);
}

} // namespace ninfer::ops::detail::kv
