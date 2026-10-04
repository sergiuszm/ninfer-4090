// Adapted from tensorninja/ninfer-4090 and soohl/ninfer (both Apache-2.0):
// the group-64 INT8 activation prefill.

#pragma once

// Folded gate/up Q4G64 x INT8-activation Tensor Core GEMM for prefill token counts.
//
//   out[i, t] = SiLU(gate_up[i, t]) * gate_up[M + i, t],  M = intermediate
//
// This is the fused sibling of int8_rowsplit_gemm.cuh and shares its arithmetic
// contract exactly: Q4G64 codes are consumed as INT8 without a floating-point
// decode, each 64-value quant group contracts in INT32 through
// mma.m16n8k32.s32.s8.s8.s32, and the group result is rescaled once into FP32 by
// the product of the stored weight group scale and the activation group scale.
// Cross-group accumulation is FP32. See that file for the weight super-group
// staging layout.
//
// The fold. A logical block covers kPairRows gate rows followed by their
// kPairRows matching up rows, so one contraction produces both projections into
// one accumulator array and the SiLU epilogue pairs them without a round trip
// through memory. The BF16 route (q4_linear_swiglu_gemm_mma.cuh) does this with a
// single 64-row warp tile, which forces one warp row per block. That is too few
// warps for the INT8 schedule, so the mapping here is generalised: a warp owns
// kWarpPairRows gate rows *and* the kWarpPairRows up rows that match them, and
// blocks may hold several warp rows. Accumulator tile mi < kMmaPairRows is gate,
// mi >= kMmaPairRows is the matching up tile.

#include "ops/common/int8_mma.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int PairRows_, int BlockCols_, int WarpPairRows_, int WarpCols_, int ActStages_,
          int LaunchBoundsMinBlocks_>
struct Q4Int8SwiGluSchedule {
    // Gate rows per block; the block stages twice this many weight rows.
    static constexpr int kPairRows     = PairRows_;
    static constexpr int kBlockRows    = 2 * kPairRows;
    static constexpr int kBlockCols    = BlockCols_;
    static constexpr int kWarpPairRows = WarpPairRows_;
    static constexpr int kWarpCols     = WarpCols_;
    static constexpr int kMinBlocks    = LaunchBoundsMinBlocks_;

    static constexpr int kGroupsPerSuper = 4;
    static constexpr int kSuperK         = kGroupsPerSuper * Q4RowSplitStorage::kGroupK; // 256
    static constexpr int kCodeRowStride  = kGroupsPerSuper * Q4RowSplitStorage::kCodeBytesPerGroup;
    static constexpr int kChunksPerRow   = kCodeRowStride / 16; // 8

    static constexpr int kBlockK  = Q4RowSplitStorage::kGroupK; // 64
    static constexpr int kKBlocks = kBlockK / 32;

    static constexpr int kWeightStages = 2;
    static constexpr int kActStages    = ActStages_;

    static constexpr int kWarpGridRows = kPairRows / kWarpPairRows;
    static constexpr int kWarpGridCols = kBlockCols / kWarpCols;
    static constexpr int kWarps        = kWarpGridRows * kWarpGridCols;
    static constexpr int kThreads      = kWarps * 32;

    // Gate tiles then the matching up tiles.
    static constexpr int kMmaPairRows = kWarpPairRows / 16;
    static constexpr int kMmaRows     = 2 * kMmaPairRows;
    static constexpr int kMmaCols     = kWarpCols / 8;

    static constexpr int kCrBytes = kWeightStages * kBlockRows * kCodeRowStride;
    static constexpr int kSwBytes = kWeightStages * kBlockRows * kGroupsPerSuper * 2;
    static constexpr int kBqBytes = kActStages * kBlockCols * kBlockK;
    static constexpr int kSxBytes = kActStages * kBlockCols * 4;

    static constexpr int kCrOffset    = 0;
    static constexpr int kBqOffset    = kCrOffset + kCrBytes;
    static constexpr int kSxOffset    = kBqOffset + kBqBytes;
    static constexpr int kSwOffset    = kSxOffset + kSxBytes;
    static constexpr int kSharedBytes = kSwOffset + kSwBytes;

    static_assert(kPairRows % kWarpPairRows == 0 && kBlockCols % kWarpCols == 0);
    static_assert(kWarpPairRows % 16 == 0 && kWarpCols % 8 == 0);
    static_assert(kActStages >= 3, "single-barrier pipeline needs at least three activation slots");
    static_assert(kThreads <= 1024 && kWarps >= 1);
    static_assert(kBlockCols % 16 == 0, "activation staging copies 16 columns worth of k at once");
    static_assert(kSharedBytes <= 99 * 1024, "Q4 INT8 SwiGLU exceeds the dynamic shared budget");
};

// clang-format off
template <class Schedule_, bool Full>
__global__ __launch_bounds__(Schedule_::kThreads, Schedule_::kMinBlocks)
void q4_linear_swiglu_int8_gemm_kernel(
    const std::int8_t* __restrict__ xq,
    const float* __restrict__ xs,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    std::int32_t intermediate,
    std::int32_t cols,
    std::int32_t padded_k) {
    // clang-format on
    using S               = Schedule_;
    constexpr int PM      = S::kPairRows;
    constexpr int BM      = S::kBlockRows;
    constexpr int BN      = S::kBlockCols;
    constexpr int BK      = S::kBlockK;
    constexpr int WPR     = S::kWarpPairRows;
    constexpr int WN      = S::kWarpCols;
    constexpr int MTP     = S::kMmaPairRows;
    constexpr int MT      = S::kMmaRows;
    constexpr int NT      = S::kMmaCols;
    constexpr int KB      = S::kKBlocks;
    constexpr int GPS     = S::kGroupsPerSuper;
    constexpr int CRS     = S::kCodeRowStride;
    constexpr int CPR     = S::kChunksPerRow;
    constexpr int WST     = S::kWeightStages;
    constexpr int AST     = S::kActStages;
    constexpr int kGroupK = Q4RowSplitStorage::kGroupK;

    extern __shared__ __align__(16) std::uint8_t smem_raw[];
    auto* Cr = reinterpret_cast<std::uint8_t*>(smem_raw + S::kCrOffset);
    auto* Bq = reinterpret_cast<std::int8_t*>(smem_raw + S::kBqOffset);
    auto* Sx = reinterpret_cast<float*>(smem_raw + S::kSxOffset);
    auto* Sw = reinterpret_cast<std::uint16_t*>(smem_raw + S::kSwOffset);

    const int tid       = static_cast<int>(threadIdx.x);
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int warp_row  = warp / S::kWarpGridCols;
    const int warp_col  = warp % S::kWarpGridCols;
    const int group_row = lane >> 2;
    const int group_col = lane & 3;

    // Visit token tiles within four neighboring gate/up row tiles so nearby CTAs
    // reuse the same weight region before moving to the next output rows.
    const int block      = int(blockIdx.y) * int(gridDim.x) + int(blockIdx.x);
    const int group_span = 4 * int(gridDim.y);
    const int first_row  = (block / group_span) * 4;
    const int group_rows = min(int(gridDim.x) - first_row, 4);
    const int row_block  = first_row + (block % group_span) % group_rows;
    const int col_block  = (block % group_span) / group_rows;
    const int m0         = row_block * PM;
    const int col0       = col_block * BN;
    const int n_groups   = padded_k / kGroupK;
    const int n_supers   = n_groups / GPS;

    // Local staged row -> weight row. Rows [0, PM) are gate, [PM, 2*PM) are the
    // matching up rows one full intermediate block later.
    auto global_row = [&](int row) { return m0 + (row & (PM - 1)) + (row / PM) * intermediate; };
    // Local staged row of accumulator tile mi for this warp.
    auto tile_row = [&](int mi) {
        return (mi < MTP) ? (warp_row * WPR + mi * 16) : (PM + warp_row * WPR + (mi - MTP) * 16);
    };

    auto cr_at = [&](int st, int local_row, int chunk) {
        return &Cr[(st * BM + local_row) * CRS + (chunk ^ (local_row & 7)) * 16];
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
#pragma unroll
            for (int c = 0; c < 4; ++c) { acc[mi][ni][c] = 0.0f; }
        }
    }

    auto stage_weights = [&](int st, int super) {
        const int g0 = super * GPS;
#pragma unroll 1
        for (int item = tid; item < BM * CPR; item += S::kThreads) {
            const int local_row = item / CPR;
            const int chunk     = item - local_row * CPR;
            std::uint8_t* dst   = cr_at(st, local_row, chunk);
            if (Full || m0 + (local_row & (PM - 1)) < intermediate) {
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(local_row)) * n_groups + g0;
                cp_async<16>(dst, &codes[gi * Q4RowSplitStorage::kCodeBytesPerGroup + chunk * 16]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
#pragma unroll 1
        for (int local_row = tid; local_row < BM; local_row += S::kThreads) {
            std::uint16_t v[GPS];
            if (Full || m0 + (local_row & (PM - 1)) < intermediate) {
                const std::uint16_t* src = reinterpret_cast<const std::uint16_t*>(
                    &scales[(static_cast<std::int64_t>(global_row(local_row)) * n_groups + g0) *
                            Q4RowSplitStorage::kScaleBytesPerGroup]);
#pragma unroll
                for (int j = 0; j < GPS; ++j) { v[j] = src[j]; }
            } else {
#pragma unroll
                for (int j = 0; j < GPS; ++j) { v[j] = 0; }
            }
#pragma unroll
            for (int j = 0; j < GPS; ++j) { Sw[(st * BM + local_row) * GPS + j] = v[j]; }
        }
    };

    auto stage_activations = [&](int st, int g) {
        const std::int64_t k0 = static_cast<std::int64_t>(g) * kGroupK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 16); item += S::kThreads) {
            const int local_col = item / (BK / 16);
            const int k16       = item - local_col * (BK / 16);
            const int col       = col0 + local_col;
            std::int8_t* dst    = &Bq[((st * (BK / 16) + k16) * BN + local_col) * 16];
            if (Full || col < cols) {
                cp_async<16>(dst, &xq[static_cast<std::int64_t>(col) * padded_k + k0 + k16 * 16]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
#pragma unroll 1
        for (int local_col = tid; local_col < BN; local_col += S::kThreads) {
            const int col = col0 + local_col;
            Sx[st * BN + local_col] =
                (Full || col < cols) ? xs[static_cast<std::int64_t>(g) * cols + col] : 0.0f;
        }
    };

#pragma unroll
    for (int s = 0; s < AST - 1; ++s) {
        if (s == 0) { stage_weights(0, 0); }
        if (s < n_groups) { stage_activations(s, s); }
        cp_commit();
    }

    for (int super = 0; super < n_supers; ++super) {
        const int wst = super % WST;
#pragma unroll
        for (int j = 0; j < GPS; ++j) {
            const int g   = super * GPS + j;
            const int ast = (AST == GPS) ? j : (g % AST);

            // See ops/common/int8_rowsplit_gemm.cuh: commit index j carries
            // activation group j, so entering iteration g at most AST - 2 of
            // the AST - 1 + g issued commits may still be pending.
            cp_wait<AST - 2>();
            __syncthreads();

            if (j == 0 && super + 1 < n_supers) { stage_weights((super + 1) % WST, super + 1); }
            const int prefetch = g + AST - 1;
            if (prefetch < n_groups) {
                stage_activations((AST == GPS) ? ((j + AST - 1) % AST) : (prefetch % AST),
                                  prefetch);
            }
            cp_commit();

            float ws[MT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int r0 = tile_row(mi) + group_row;
                ws[mi][0]    = __half2float(__ushort_as_half(Sw[(wst * BM + r0) * GPS + j]));
                ws[mi][1]    = __half2float(__ushort_as_half(Sw[(wst * BM + r0 + 8) * GPS + j]));
            }
            float xsv[NT][2];
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int c0 = warp_col * WN + ni * 8 + 2 * group_col;
                xsv[ni][0]   = Sx[ast * BN + c0];
                xsv[ni][1]   = Sx[ast * BN + c0 + 1];
            }

            unsigned af[MT][KB][4];
            unsigned bf[NT][KB][2];
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) {
                const int chunk = j * (Q4RowSplitStorage::kCodeBytesPerGroup / 16) + kb;
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    const int r0           = tile_row(mi) + group_row;
                    const std::uint8_t* p0 = cr_at(wst, r0, chunk) + group_col * 2;
                    const std::uint8_t* p1 = cr_at(wst, r0 + 8, chunk) + group_col * 2;
                    af[mi][kb][0] = q4_codes_to_s8x4(*reinterpret_cast<const std::uint16_t*>(p0));
                    af[mi][kb][1] = q4_codes_to_s8x4(*reinterpret_cast<const std::uint16_t*>(p1));
                    af[mi][kb][2] =
                        q4_codes_to_s8x4(*reinterpret_cast<const std::uint16_t*>(p0 + 8));
                    af[mi][kb][3] =
                        q4_codes_to_s8x4(*reinterpret_cast<const std::uint16_t*>(p1 + 8));
                }
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const int col   = warp_col * WN + ni * 8 + group_row;
                    const int k16   = kb * 2;
                    const int inner = group_col * 4;
                    bf[ni][kb][0]   = *reinterpret_cast<const unsigned*>(
                        &Bq[((ast * (BK / 16) + k16) * BN + col) * 16 + inner]);
                    bf[ni][kb][1] = *reinterpret_cast<const unsigned*>(
                        &Bq[((ast * (BK / 16) + k16 + 1) * BN + col) * 16 + inner]);
                }
            }

#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    int s0, s1, s2, s3;
                    mma_s8_zero(s0, s1, s2, s3, af[mi][0][0], af[mi][0][1], af[mi][0][2],
                                af[mi][0][3], bf[ni][0][0], bf[ni][0][1]);
#pragma unroll
                    for (int kb = 1; kb < KB; ++kb) {
                        mma_s8(s0, s1, s2, s3, af[mi][kb][0], af[mi][kb][1], af[mi][kb][2],
                               af[mi][kb][3], bf[ni][kb][0], bf[ni][kb][1]);
                    }
                    acc[mi][ni][0] =
                        __fmaf_rn(static_cast<float>(s0) * ws[mi][0], xsv[ni][0], acc[mi][ni][0]);
                    acc[mi][ni][1] =
                        __fmaf_rn(static_cast<float>(s1) * ws[mi][0], xsv[ni][1], acc[mi][ni][1]);
                    acc[mi][ni][2] =
                        __fmaf_rn(static_cast<float>(s2) * ws[mi][1], xsv[ni][0], acc[mi][ni][2]);
                    acc[mi][ni][3] =
                        __fmaf_rn(static_cast<float>(s3) * ws[mi][1], xsv[ni][1], acc[mi][ni][3]);
                }
            }
        }
    }

    // ---- SiLU epilogue -----------------------------------------------------
    // Tile mi holds gate, tile mi + MTP holds the up value for the same output
    // row, so the fusion is a register-to-register pairing.
#pragma unroll
    for (int mi = 0; mi < MTP; ++mi) {
        const int r0 = m0 + warp_row * WPR + mi * 16 + group_row;
        const int r1 = r0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int c0 = col0 + warp_col * WN + ni * 8 + 2 * group_col;
            const int c1 = c0 + 1;
            auto store   = [&](int col, int row, float gv, float uv) {
                out[static_cast<std::int64_t>(col) * intermediate + row] =
                    __float2bfloat16_rn(silu(gv) * uv);
            };
            if constexpr (Full) {
                store(c0, r0, acc[mi][ni][0], acc[mi + MTP][ni][0]);
                store(c1, r0, acc[mi][ni][1], acc[mi + MTP][ni][1]);
                store(c0, r1, acc[mi][ni][2], acc[mi + MTP][ni][2]);
                store(c1, r1, acc[mi][ni][3], acc[mi + MTP][ni][3]);
            } else {
                if (r0 < intermediate) {
                    if (c0 < cols) { store(c0, r0, acc[mi][ni][0], acc[mi + MTP][ni][0]); }
                    if (c1 < cols) { store(c1, r0, acc[mi][ni][1], acc[mi + MTP][ni][1]); }
                }
                if (r1 < intermediate) {
                    if (c0 < cols) { store(c0, r1, acc[mi][ni][2], acc[mi + MTP][ni][2]); }
                    if (c1 < cols) { store(c1, r1, acc[mi][ni][3], acc[mi + MTP][ni][3]); }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
