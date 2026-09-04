#pragma once

// INT8-native GQA prompt kernel for the registered Qwen3.6 head geometries. QK stays INT8 through
// m16n8k32.s8 Tensor Cores; V alone is dequantized with packed FP16 arithmetic while
// producer warps execute QK. Sixteen warps split each 16-row FP16 PV output across
// four 64-dimension slices.
//
// sm_89 runs a retuned schedule: eight paired producer warps (Bc column halves, one
// named-barrier max exchange per tile), byte-permute V dequant, fp16-accumulated PV
// tiles merged into fp32, and the full 128-register budget. See the ColSplit constant.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kernel/e8_lattice.cuh"
#include "ops/kernel/e8_root_codec.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

#include <cstdint>
#include <type_traits>

namespace ninfer::ops {

inline constexpr int kCausalPromptI8Warps      = 16;
inline constexpr int kCausalPromptI8Threads    = kCausalPromptI8Warps * 32;
inline constexpr int kCausalPromptI8Br         = 64;
inline constexpr int kCausalPromptI8Bc         = 64;
inline constexpr int kCausalPromptI8Groups     = kCausalPromptHeadDim / kKVCacheInt8Group;
inline constexpr int kCausalPromptI8DB16       = kCausalPromptHeadDim / 2;
inline constexpr int kCausalPromptI8RowTiles   = kCausalPromptI8Br / 16;
inline constexpr int kCausalPromptI8DConsumers = kCausalPromptI8Warps / kCausalPromptI8RowTiles;

inline constexpr int kCausalPromptI8QBytes = kCausalPromptI8Br * kCausalPromptHeadDim;
inline constexpr int kCausalPromptI8QScaleBytes =
    kCausalPromptI8Br * kCausalPromptI8Groups * static_cast<int>(sizeof(float));
inline constexpr int kCausalPromptI8KBytes = kCausalPromptI8Bc * kCausalPromptHeadDim;
inline constexpr int kCausalPromptI8VBytes = kCausalPromptI8Bc * kCausalPromptHeadDim;
inline constexpr int kCausalPromptI8VStageBytes =
    kCausalPromptI8Bc * kCausalPromptHeadDim * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptI8PBytes =
    kCausalPromptI8Br * kCausalPromptI8Bc * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptI8ScaleBytes =
    2 * kCausalPromptI8Bc * kCausalPromptI8Groups * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptI8StatsBytes =
    2 * kCausalPromptI8Br * static_cast<int>(sizeof(float));
// Block-max and block-sum exchange slots for the paired-producer schedule (two column
// halves per 16-row tile). Allocated on every arch so the launch envelope stays uniform.
inline constexpr int kCausalPromptI8PairStatsBytes =
    2 * 2 * kCausalPromptI8Br * static_cast<int>(sizeof(float));
inline constexpr int kCausalPromptI8SmemBytes =
    kCausalPromptI8QBytes + kCausalPromptI8QScaleBytes + kCausalPromptI8KBytes + kCausalPromptI8VBytes +
    kCausalPromptI8VStageBytes + kCausalPromptI8PBytes + kCausalPromptI8ScaleBytes +
    kCausalPromptI8StatsBytes + kCausalPromptI8PairStatsBytes;

static_assert(kCausalPromptI8Groups == 4);
static_assert(kCausalPromptI8DConsumers == 4);
static_assert(kCausalPromptI8SmemBytes == 93696);

__device__ __forceinline__ int4 causal_prompt_i8_dequant_f16x8(const std::int8_t* codes8,
                                                             __half scale) {
    const int2 raw   = load_vec<int2>(codes8);
    const __half2 s2 = __halves2half2(scale, scale);
    unsigned packed[4];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
    // Ada throttles on the conversion pipe, so build the halves with byte permutes
    // instead: bias each code to unsigned, splice it under exponent 2^10 (0x64xx is
    // 1024 + code), and subtract 1024 + 128. Integer halves are exact, so the result
    // is bit-identical to the I2F path.
    const __half2 magic2 = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
    const unsigned x0    = static_cast<unsigned>(raw.x) ^ 0x80808080u;
    const unsigned x1    = static_cast<unsigned>(raw.y) ^ 0x80808080u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned src   = i < 2 ? x0 : x1;
        const unsigned pair  = __byte_perm(src, 0x64646464u, (i & 1) ? 0x7352u : 0x7150u);
        const __half2 code2  = __hsub2(*reinterpret_cast<const __half2*>(&pair), magic2);
        const __half2 value2 = __hmul2(code2, s2);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
#else
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 code2 =
            __floats2half2_rn(static_cast<float>(c[2 * i]), static_cast<float>(c[2 * i + 1]));
        const __half2 value2 = __hmul2(code2, s2);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
#endif
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}



// 120 registers is the spill-free point on SM120. Ada codegen spills the producer
// score/accumulator state at 120, so give it the full file: 512 threads x 128 = 64K
// registers, and occupancy is capped at one CTA by shared memory either way.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
#define NINFER_CAUSAL_PROMPT_I8_MAXNREG 128
#else
#define NINFER_CAUSAL_PROMPT_I8_MAXNREG 120
#endif

template <typename Geometry, bool PackedV, bool RotateK, bool RotateV, bool PackedK,
          bool E8Root = false, typename Metadata>
__global__ __maxnreg__(NINFER_CAUSAL_PROMPT_I8_MAXNREG) void causal_attention_prompt_i8_kernel(
    const __nv_bfloat16* __restrict__ q, const std::int8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width) {
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int Br            = kCausalPromptI8Br;
    constexpr int Bc            = kCausalPromptI8Bc;
    constexpr int DB16          = kCausalPromptI8DB16;
    constexpr int Groups        = kCausalPromptI8Groups;
    constexpr int GroupKc       = kKVCacheInt8Group / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVNtPerWarp   = D / (kCausalPromptI8DConsumers * 8);
    constexpr int PVKs          = Bc / 16;
// Ada runs one CTA per SM; four producer warps (one per scheduler) cannot hide mma
// latency and leave twelve workers stalled at the phase barrier. Split each 16-row
// score tile across a warp pair (column halves of Bc) there. SM120 keeps 4/12.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
    constexpr int ColSplit = 2;
#else
    constexpr int ColSplit = 1;
#endif
    constexpr int ProducerWarps = kCausalPromptI8RowTiles * ColSplit;
    constexpr int VWorkerWarps  = kCausalPromptI8Warps - ProducerWarps;
    constexpr int WorkerThreads = VWorkerWarps * 32;
    constexpr int QKNtL         = QKNt / ColSplit;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(GroupKc == 2);
    static_assert(PVNtPerWarp == 8);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8 = reinterpret_cast<std::int8_t*>(smem_raw);
    float* q_scale    = reinterpret_cast<float*>(q_i8 + kCausalPromptI8QBytes);
    std::int8_t* k_i8 = reinterpret_cast<std::int8_t*>(reinterpret_cast<unsigned char*>(q_scale) +
                                                       kCausalPromptI8QScaleBytes);
    std::int8_t* v_i8 = k_i8 + kCausalPromptI8KBytes;
    __half* v_f16     = reinterpret_cast<__half*>(v_i8 + kCausalPromptI8VBytes);
    __half* p_s       = reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(v_f16) +
                                                  kCausalPromptI8VStageBytes);
    __half* k_scale_s =
        reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(p_s) + kCausalPromptI8PBytes);
    __half* v_scale_s    = k_scale_s + Bc * Groups;
    float* alpha_s       = reinterpret_cast<float*>(v_scale_s + Bc * Groups);
    float* final_l_s     = alpha_s + Br;
    float* pair_m_s      = final_l_s + Br;
    float* pair_l_s      = pair_m_s + 2 * Br;
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __nv_bfloat16* k_b16 = reinterpret_cast<__nv_bfloat16*>(k_i8);

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        causal_prompt_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                               kCausalPromptI8Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks    = max_query_abs / Bc + 1;

    // Leading key blocks whose every key is visible to every row of this CTA tile
    // ((kb + 1) * Bc - 1 <= base_pos + q0). Those blocks stage and score without
    // causal guards; the boundary blocks after them keep the exact masked path.
    const int n_full_blocks = (q0 + Br <= tokens) ? min(key_blocks, (base_pos + q0 + 1) / Bc) : 0;

    // Quantize Q cooperatively. One warp owns one (row, 64-d group) at a time.
    for (int unit = warp; unit < Br * Groups; unit += kCausalPromptI8Warps) {
        const int row = unit / Groups;
        const int grp = unit - row * Groups;
        const int d0  = grp * kKVCacheInt8Group + lane;
        const int d1  = d0 + 32;
        float x0      = 0.0f;
        float x1      = 0.0f;
        if (row < tile_rows) {
            x0 = __bfloat162float(q[causal_prompt_q_index<Geometry>(q_head, d0, q0 + row)]);
            x1 = __bfloat162float(q[causal_prompt_q_index<Geometry>(q_head, d1, q0 + row)]);
            if constexpr (RotateK) { kv_cache_hadamard64(x0, x1, FullMask); }
        }
        float absmax    = fmaxf(fabsf(x0), fabsf(x1));
        absmax          = warp_max(absmax, FullMask);
        const float qs  = absmax > 0.0f ? absmax / 127.0f : 0.0f;
        const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
        causal_prompt_store_byte_swizzled(q_i8, row, d0, kv_cache_int8_quant_code(x0, inv));
        causal_prompt_store_byte_swizzled(q_i8, row, d1, kv_cache_int8_quant_code(x1, inv));
        if (lane == 0) { q_scale[row * Groups + grp] = qs; }
    }
    __syncthreads();

    auto issue_kv_tile = [&](int tile_k0, auto full_tag) {
        // FullTile folds every causal guard to taken and dead-codes the zero-fill
        // paths; interior blocks stage with unconditional copies.
        constexpr bool FullTile = decltype(full_tag)::value;
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        for (int key_l = tid; key_l < Bc; key_l += kCausalPromptI8Threads) {
            const int key = tile_k0 + key_l;
            __half* kd    = &k_scale_s[key_l * Groups];
            __half* vd    = &v_scale_s[key_l * Groups];
            if (FullTile || key <= max_query_abs) {
                const std::int64_t off =
                    kv_cache_int8_quant_scale_index<Geometry>(physical_page, kv_head, 0, key_l);
                ninfer::ops::cp_async<8>(kd, &cache_k_scale[off]);
                ninfer::ops::cp_async<8>(vd, &cache_v_scale[off]);
            } else {
                store_vec(kd, make_int2(0, 0));
                store_vec(vd, make_int2(0, 0));
            }
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += kCausalPromptI8Threads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            const int key   = tile_k0 + key_l;
            std::int8_t* kd = &k_i8[(key_l * DB16 + causal_prompt_swz(key_l, dc * 8)) * 2];
            std::int8_t* vd = &v_i8[key_l * D + d];
            if (FullTile || key <= max_query_abs) {
                if constexpr (E8Root) {
                    const std::int64_t koff = paged_kv_page_head_offset<64, Geometry::KVHeads>(
                        physical_page, kv_head) + static_cast<std::int64_t>(key_l) * 64 + (d / 4);
                    const uint32_t src4 = *reinterpret_cast<const uint32_t*>(&reinterpret_cast<const std::uint8_t*>(cache_k)[koff]);
                    const uint8_t c1_0 = static_cast<uint8_t>(src4 & 0xFF);
                    const uint8_t c2_0 = static_cast<uint8_t>((src4 >> 8) & 0xFF);
                    const uint8_t c1_1 = static_cast<uint8_t>((src4 >> 16) & 0xFF);
                    const uint8_t c2_1 = static_cast<uint8_t>((src4 >> 24) & 0xFF);
                    int8_t dec8_0[8], dec8_1[8];
                    e8_root_decode_8d_int8(c1_0, c2_0, dec8_0);
                    e8_root_decode_8d_int8(c1_1, c2_1, dec8_1);
                    *reinterpret_cast<uint64_t*>(&kd[0]) = *reinterpret_cast<const uint64_t*>(dec8_0);
                    *reinterpret_cast<uint64_t*>(&kd[8]) = *reinterpret_cast<const uint64_t*>(dec8_1);
                    const std::int64_t voff =
                        kv_cache_i4_code_index<Geometry>(physical_page, kv_head, d / 2, key_l);
                    kv_cache_unpack_i4x16(&cache_v[voff], vd);
                } else if constexpr (PackedK) {
                    const std::int64_t koff =
                        kv_cache_i4_code_index<Geometry>(physical_page, kv_head, d / 2, key_l);
                    kv_cache_unpack_i4x16(&reinterpret_cast<const std::uint8_t*>(cache_k)[koff], kd);
                    const std::int64_t voff =
                        kv_cache_i4_code_index<Geometry>(physical_page, kv_head, d / 2, key_l);
                    kv_cache_unpack_i4x16(&cache_v[voff], vd);
                } else {
                    const std::int64_t off =
                        kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d, key_l);
                    cp_async<16, Cache::cg>(kd, &cache_k[off]);
                    if constexpr (PackedV) {
                        const std::int64_t voff = kv_cache_i4_code_index<Geometry>(
                            physical_page, kv_head, d / 2, key_l);
                        kv_cache_unpack_i4x16(&cache_v[voff], vd);
                    } else {
                        cp_async<16, Cache::cg>(vd,
                                                &reinterpret_cast<const std::int8_t*>(cache_v)[off]);
                    }
                }
            } else {
                store_vec(kd, make_int4(0, 0, 0, 0));
                store_vec(vd, make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    if (n_full_blocks > 0) {
        issue_kv_tile(0, std::true_type{});
    } else {
        issue_kv_tile(0, std::false_type{});
    }
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    // Keeping exactly two group scales live is the spill-free 120-register point on SM120.
    // Groups 2/3 reload per key tile; retaining all four creates an 8-byte stack frame.
    float q_scale_r0[Groups - 2];
    float q_scale_r1[Groups - 2];
    if (warp < ProducerWarps) {
        const int scale_row0 = (warp / ColSplit) * 16 + gid;
        const int scale_row1 = scale_row0 + 8;
#pragma unroll
        for (int grp = 0; grp < Groups - 2; ++grp) {
            float qs0       = lid == 0 ? q_scale[scale_row0 * Groups + grp] : 0.0f;
            float qs1       = lid == 0 ? q_scale[scale_row1 * Groups + grp] : 0.0f;
            q_scale_r0[grp] = __shfl_sync(FullMask, qs0, gid * 4);
            q_scale_r1[grp] = __shfl_sync(FullMask, qs1, gid * 4);
        }
    }

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;
    // One body, two instantiations: FullTile compiles the interior-block path with no
    // causal masking, no softmax zero-selects, and unguarded staging/dequant; the
    // partial instantiation keeps the exact masked path for boundary blocks.
    auto process_key_block = [&](int kb, auto full_tag) {
        constexpr bool FullTile = decltype(full_tag)::value;
        const int k0            = kb * Bc;
        if (warp < ProducerWarps) {
            const int row_base = (warp / ColSplit) * 16;
            const int col_half = warp % ColSplit;
            float score[QKNtL][4];
#pragma unroll
            for (int nt = 0; nt < QKNtL; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            }

// Full unroll interleaves all four groups' A-fragments and overflows the Ada register
// file into local memory; two groups in flight keep the ntl chains independent spill-free.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
#pragma unroll 2
#else
#pragma unroll
#endif
            for (int grp = 0; grp < Groups; ++grp) {
                float qs0;
                float qs1;
                if (grp < Groups - 2) {
                    qs0 = q_scale_r0[grp];
                    qs1 = q_scale_r1[grp];
                } else {
                    const int scale_row0 = row_base + gid;
                    const int scale_row1 = scale_row0 + 8;
                    qs0                  = lid == 0 ? q_scale[scale_row0 * Groups + grp] : 0.0f;
                    qs1                  = lid == 0 ? q_scale[scale_row1 * Groups + grp] : 0.0f;
                    qs0                  = __shfl_sync(FullMask, qs0, gid * 4);
                    qs1                  = __shfl_sync(FullMask, qs1, gid * 4);
                }

                unsigned af[GroupKc][4];
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k    = grp * GroupKc + kk;
                    const int acol = k * 16 + a_coloff;
                    ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                                smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                                 causal_prompt_swz(row_base + a_rowoff, acol)]));
                }

#pragma unroll
                for (int ntl = 0; ntl < QKNtL; ++ntl) {
                    const int nt = col_half * QKNtL + ntl;
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int k    = grp * GroupKc + kk;
                        const int brow = nt * 8 + b_rin;
                        const int bcol = k * 16 + b_koff;
                        unsigned bf[2];
                        ldmatrix_x2(bf[0], bf[1],
                                    smem_addr(&k_b16[brow * DB16 + causal_prompt_swz(brow, bcol)]));
                        mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                               bf[1]);
                    }
                    const int keya = nt * 8 + 2 * lid;
                    const int keyb = keya + 1;
                    float ks0      = 0.0f;
                    float ks1      = 0.0f;
                    if (gid == 0) {
                        ks0 = __half2float(k_scale_s[keya * Groups + grp]);
                        ks1 = __half2float(k_scale_s[keyb * Groups + grp]);
                    }
                    ks0           = __shfl_sync(FullMask, ks0, lid);
                    ks1           = __shfl_sync(FullMask, ks1, lid);
                    score[ntl][0] = __fmaf_rn(qs0 * ks0, static_cast<float>(c0), score[ntl][0]);
                    score[ntl][1] = __fmaf_rn(qs0 * ks1, static_cast<float>(c1), score[ntl][1]);
                    score[ntl][2] = __fmaf_rn(qs1 * ks0, static_cast<float>(c2), score[ntl][2]);
                    score[ntl][3] = __fmaf_rn(qs1 * ks1, static_cast<float>(c3), score[ntl][3]);
                }
            }

            const int row0 = row_base + gid;
            const int row1 = row0 + 8;
            if constexpr (!FullTile) {
                const int qabs0 = row0 < tile_rows ? base_pos + q0 + row0 : -1;
                const int qabs1 = row1 < tile_rows ? base_pos + q0 + row1 : -1;
                // A boundary block can still be fully visible for a tail CTA whose
                // n_full_blocks collapsed to zero; keep the per-block skip.
                const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
                if (!full_score_tile) {
#pragma unroll
                    for (int ntl = 0; ntl < QKNtL; ++ntl) {
                        const int nt   = col_half * QKNtL + ntl;
                        const int key0 = k0 + nt * 8 + 2 * lid;
                        const int key1 = key0 + 1;
                        score[ntl][0]  = key0 <= qabs0 ? score[ntl][0] : -CUDART_INF_F;
                        score[ntl][1]  = key1 <= qabs0 ? score[ntl][1] : -CUDART_INF_F;
                        score[ntl][2]  = key0 <= qabs1 ? score[ntl][2] : -CUDART_INF_F;
                        score[ntl][3]  = key1 <= qabs1 ? score[ntl][3] : -CUDART_INF_F;
                    }
                }
            }
            float bm0 = -CUDART_INF_F;
            float bm1 = -CUDART_INF_F;
#pragma unroll
            for (int ntl = 0; ntl < QKNtL; ++ntl) {
                bm0 = fmaxf(bm0, fmaxf(score[ntl][0], score[ntl][1]));
                bm1 = fmaxf(bm1, fmaxf(score[ntl][2], score[ntl][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);
            if constexpr (ColSplit == 2) {
                if (lid == 0) {
                    pair_m_s[col_half * Br + row0] = bm0;
                    pair_m_s[col_half * Br + row1] = bm1;
                }
                asm volatile("bar.sync 1, %0;" ::"r"(ProducerWarps * 32) : "memory");
                bm0 = fmaxf(pair_m_s[row0], pair_m_s[Br + row0]);
                bm1 = fmaxf(pair_m_s[row1], pair_m_s[Br + row1]);
            }

            const float nm0        = fmaxf(running_m0, bm0);
            const float nm1        = fmaxf(running_m1, bm1);
            const float nm0_scaled = nm0 * scale_l2;
            const float nm1_scaled = nm1 * scale_l2;
            const float alpha0     = running_m0 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
            const float alpha1     = running_m1 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
            float bl0              = 0.0f;
            float bl1              = 0.0f;
#pragma unroll
            for (int ntl = 0; ntl < QKNtL; ++ntl) {
                const int col0 = (col_half * QKNtL + ntl) * 8 + 2 * lid;
                float p00, p01, p10, p11;
                if constexpr (FullTile) {
                    p00 = exp2_approx(__fmaf_rn(score[ntl][0], scale_l2, -nm0_scaled));
                    p01 = exp2_approx(__fmaf_rn(score[ntl][1], scale_l2, -nm0_scaled));
                    p10 = exp2_approx(__fmaf_rn(score[ntl][2], scale_l2, -nm1_scaled));
                    p11 = exp2_approx(__fmaf_rn(score[ntl][3], scale_l2, -nm1_scaled));
                } else {
                    p00 = score[ntl][0] > -CUDART_INF_F
                              ? exp2_approx(__fmaf_rn(score[ntl][0], scale_l2, -nm0_scaled))
                              : 0.0f;
                    p01 = score[ntl][1] > -CUDART_INF_F
                              ? exp2_approx(__fmaf_rn(score[ntl][1], scale_l2, -nm0_scaled))
                              : 0.0f;
                    p10 = score[ntl][2] > -CUDART_INF_F
                              ? exp2_approx(__fmaf_rn(score[ntl][2], scale_l2, -nm1_scaled))
                              : 0.0f;
                    p11 = score[ntl][3] > -CUDART_INF_F
                              ? exp2_approx(__fmaf_rn(score[ntl][3], scale_l2, -nm1_scaled))
                              : 0.0f;
                }
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                // The swizzle keeps even/odd column pairs adjacent, so store one half2.
                *reinterpret_cast<__half2*>(&p_s[row0 * Bc + causal_prompt_p_swz<Bc>(row0, col0)]) =
                    __floats2half2_rn(p00, p01);
                *reinterpret_cast<__half2*>(&p_s[row1 * Bc + causal_prompt_p_swz<Bc>(row1, col0)]) =
                    __floats2half2_rn(p10, p11);
            }
            bl0 = warp_sum<4>(bl0, FullMask);
            bl1 = warp_sum<4>(bl1, FullMask);
            // With paired producers each half accumulates its own partial row sum; the
            // shared block max makes the alpha sequences identical, so the halves add
            // linearly and are combined once after the key loop.
            running_l0 = __fmaf_rn(running_l0, alpha0, bl0);
            running_l1 = __fmaf_rn(running_l1, alpha1, bl1);
            running_m0 = nm0;
            running_m1 = nm1;
            if (col_half == 0 && lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else if (warp < ProducerWarps + VWorkerWarps) {
            const int worker_tid = tid - ProducerWarps * 32;
#pragma unroll 1
            for (int chunk = worker_tid; chunk < Bc * (D / 8); chunk += WorkerThreads) {
                const int key_l = chunk / (D / 8);
                const int dc    = chunk - key_l * (D / 8);
                const int d     = dc * 8;
                const int key   = k0 + key_l;
                __half* dst     = &v_f16[key_l * D + causal_prompt_swz(key_l, d)];
                if (FullTile || key <= max_query_abs) {
                    const int grp = d >> 6;
                    __half vs     = __float2half_rn(0.0f);
                    if ((lane & 7) == 0) { vs = v_scale_s[key_l * Groups + grp]; }
                    vs = __shfl_sync(FullMask, vs, grp * 8);
                    store_vec(dst, causal_prompt_i8_dequant_f16x8(&v_i8[key_l * D + d], vs));
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) {
            if (kb + 1 < n_full_blocks) {
                issue_kv_tile((kb + 1) * Bc, std::true_type{});
            } else {
                issue_kv_tile((kb + 1) * Bc, std::false_type{});
            }
        }

        const int row_tile = warp % kCausalPromptI8RowTiles;
        const int d_slice  = warp / kCausalPromptI8RowTiles;
        const int row_base = row_tile * 16;
        const float alpha0 = alpha_s[row_base + gid];
        const float alpha1 = alpha_s[row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
        // Consumer Ada runs f32-acc HMMA at half rate. Accumulate the 64-key tile in
        // packed fp16 at full rate and fold it into the fp32 running accumulator once
        // per tile; the per-tile sums are magnitude-bounded by the softmax weights.
        unsigned tacc[PVNtPerWarp][2];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) { tacc[n][0] = tacc[n][1] = 0u; }
#endif
#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned pf[4];
            const int pcol = k * 16 + a_coloff;
            ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                        smem_addr(&p_s[(row_base + a_rowoff) * Bc +
                                       causal_prompt_p_swz<Bc>(row_base + a_rowoff, pcol)]));
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = d_slice * PVNtPerWarp + n;
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_f16[vrow * D + causal_prompt_swz(vrow, vcol)]));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
                mma_f16_f16acc(tacc[n][0], tacc[n][1], pf[0], pf[1], pf[2], pf[3], vf[0], vf[1]);
#else
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                        vf[0], vf[1]);
#endif
            }
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const __half2 lo = *reinterpret_cast<const __half2*>(&tacc[n][0]);
            const __half2 hi = *reinterpret_cast<const __half2*>(&tacc[n][1]);
            acc[n][0] += __half2float(lo.x);
            acc[n][1] += __half2float(lo.y);
            acc[n][2] += __half2float(hi.x);
            acc[n][3] += __half2float(hi.y);
        }
#endif
        if (has_next) { ninfer::ops::cp_wait<0>(); }
        __syncthreads();
    };

    for (int kb = 0; kb < n_full_blocks; ++kb) { process_key_block(kb, std::true_type{}); }
    for (int kb = n_full_blocks; kb < key_blocks; ++kb) { process_key_block(kb, std::false_type{}); }

    if constexpr (ColSplit == 2) {
        if (warp < ProducerWarps && lid == 0) {
            const int row0                            = (warp / ColSplit) * 16 + gid;
            pair_l_s[(warp % ColSplit) * Br + row0]   = running_l0;
            pair_l_s[(warp % ColSplit) * Br + row0 + 8] = running_l1;
        }
        __syncthreads();
        if (warp < ProducerWarps && warp % ColSplit == 0 && lid == 0) {
            const int row0  = (warp / ColSplit) * 16 + gid;
            const int row1  = row0 + 8;
            final_l_s[row0] = pair_l_s[row0] + pair_l_s[Br + row0];
            final_l_s[row1] = pair_l_s[row1] + pair_l_s[Br + row1];
        }
    } else if (warp < ProducerWarps && lid == 0) {
        const int row0  = warp * 16 + gid;
        const int row1  = row0 + 8;
        final_l_s[row0] = running_l0;
        final_l_s[row1] = running_l1;
    }
    __syncthreads();

    const int row_tile = warp % kCausalPromptI8RowTiles;
    const int d_slice  = warp / kCausalPromptI8RowTiles;
    const int row_base = row_tile * 16;
    const int row0     = row_base + gid;
    const int row1     = row0 + 8;
    const float inv_l0 = final_l_s[row0] > 0.0f ? __frcp_rn(final_l_s[row0]) : 0.0f;
    const float inv_l1 = final_l_s[row1] > 0.0f ? __frcp_rn(final_l_s[row1]) : 0.0f;
    __nv_bfloat16* out_row0 =
        row0 < tile_rows ? out + causal_prompt_q_row_offset<Geometry>(q_head, q0 + row0) : nullptr;
    __nv_bfloat16* out_row1 =
        row1 < tile_rows ? out + causal_prompt_q_row_offset<Geometry>(q_head, q0 + row1) : nullptr;
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int d0 = (d_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        if (out_row0 != nullptr) {
            *reinterpret_cast<unsigned*>(&out_row0[d0]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (out_row1 != nullptr) {
            *reinterpret_cast<unsigned*>(&out_row1[d0]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
    causal_prompt_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                           kCausalPromptI8Threads);
}

} // namespace ninfer::ops
