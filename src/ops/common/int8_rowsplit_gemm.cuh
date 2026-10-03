// Adapted from tensorninja/ninfer-4090 (Apache-2.0); see docs/ada.md for provenance.

#pragma once

// Row-split groupwise-int x INT8-activation Tensor Core GEMM for prefill token
// counts, over the Q4G64 and Q5G64 codecs and the store / residual-add
// epilogues.
//
//   Store:       out[Rows, Cols]  = W[Rows, K] * x[K, Cols]
//   AddResidual: out[Rows, Cols] += W[Rows, K] * x[K, Cols]
//
// `rows` is the job's row count and bounds the computation; `out_row_stride` is
// the leading dimension of the destination tensor. They differ when a job writes
// a row slice of a larger output, as the fused projection pairs do: the caller
// passes `out` already advanced to the slice's first row.
//
// The weight codes are consumed as INT8 without any floating-point decode: a
// Q4G64 code is an exact integer in [-8, 7] and a Q5G64 code an exact integer in
// [-16, 15]. Each 64-value quant group contracts in INT32 through
// mma.m16n8k32.s32.s8.s8.s32, then rescales once into FP32 by the product of the
// stored weight group scale and the activation group scale. Cross-group
// accumulation stays FP32, matching the BF16 routes.
//
// Activations arrive pre-quantized as INT8 codes plus one FP32 scale per
// (64-channel group, token) from act_quant_g64. The group extent matches the
// weight codecs exactly, so one activation scale spans one weight group and the
// rescale is a single product.
//
// Weight and activation K-tiling are deliberately decoupled.
//
// Both codecs store codes as codes[row][group][32]. For a fixed group,
// consecutive rows are 32 * n_groups bytes apart, so a warp staging one group
// touches 32 scattered 16-byte spans and issues 16 memory requests. Staging a
// whole 256-element **super-group** instead makes each row's 128 code bytes
// contiguous: eight threads cover one row, a warp covers four rows, and the warp
// issues four requests. One staged weight tile feeds four consecutive quant
// groups, reducing both scattered memory requests and staging barriers.
//
// Activations stay tiled at one quant group. They are re-read by every row
// block, so they are L2-resident and latency-tolerant, and a 256-wide activation
// tile would not fit in shared memory alongside the weight tile.
//
// Q5 additionally carries a high-bit plane of 8 bytes per group, staged at
// 32 bytes per row per super-group with aligned 32-bit fragment loads.
//
// Shared-memory layouts:
//   Cr[row][8][16]     - 128 code bytes per row per super-group, 16-byte chunks
//                        XOR-swizzled by (row & 7) so the eight A rows of a warp
//                        land in eight distinct bank groups at zero padding cost
//   Hr[row][48]        - Q5 only: 32 high bytes per row per super-group, padded
//                        to 48 so eight rows land in eight distinct banks
//   Sw[row][4]         - the four FP16 weight group scales of the super-group
//   Bq[k16][col][16]   - lane L of an n-tile reads word (base + L)
//   Sx[col]            - one FP32 activation group scale per token

#include "ops/common/int8_mma.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class Int8GemmCodec { Q4, Q5 };
enum class Int8GemmEpilogue { Store, AddResidual };

// Both codecs share the group extent, code plane width, and scale width; they
// differ only in whether a high-bit plane exists.
struct Int8RowSplitStorage {
    static constexpr int kGroupK             = 64;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kHighBytesPerGroup  = 8;
    static constexpr int kScaleBytesPerGroup = 2;
};

template <Int8GemmCodec Codec_, int BlockRows_, int BlockCols_, int WarpRows_, int WarpCols_,
          int ActStages_, int LaunchBoundsMinBlocks_>
struct Int8RowSplitGemmSchedule {
    static constexpr Int8GemmCodec kCodec = Codec_;
    static constexpr bool kHasHighPlane   = Codec_ == Int8GemmCodec::Q5;

    static constexpr int kBlockRows = BlockRows_;
    static constexpr int kBlockCols = BlockCols_;
    static constexpr int kWarpRows  = WarpRows_;
    static constexpr int kWarpCols  = WarpCols_;
    static constexpr int kMinBlocks = LaunchBoundsMinBlocks_;

    // Weight tile: one 128-byte code line per row = four quant groups.
    static constexpr int kGroupsPerSuper = 4;
    static constexpr int kSuperK         = kGroupsPerSuper * Int8RowSplitStorage::kGroupK; // 256
    static constexpr int kCodeRowStride = kGroupsPerSuper * Int8RowSplitStorage::kCodeBytesPerGroup;
    static constexpr int kChunksPerRow  = kCodeRowStride / 16; // 8

    static constexpr int kHighBytes =
        kHasHighPlane ? kGroupsPerSuper * Int8RowSplitStorage::kHighBytesPerGroup : 0;
    static constexpr int kHighRowStride = kHasHighPlane ? 48 : 0;
    static constexpr int kHighChunks    = kHighBytes / 16;

    // Activation tile: one quant group.
    static constexpr int kBlockK  = Int8RowSplitStorage::kGroupK; // 64
    static constexpr int kKBlocks = kBlockK / 32;                 // m16n8k32 steps per group

    // Pipeline depth. Weights are double-buffered at super-group granularity,
    // which is four group-times of lookahead on the DRAM-critical stream.
    //
    // Three or more activation buffers allow one barrier per group instead of
    // two: the prefetch for group g + AST - 1 targets slot (g - 1) % AST, which
    // the previous iteration read and which this iteration's single barrier has
    // already retired. A trailing "everyone finished reading" barrier is then
    // unnecessary.
    static constexpr int kWeightStages = 2;
    static constexpr int kActStages    = ActStages_;

    static constexpr int kWarpGridRows = kBlockRows / kWarpRows;
    static constexpr int kWarpGridCols = kBlockCols / kWarpCols;
    static constexpr int kWarps        = kWarpGridRows * kWarpGridCols;
    static constexpr int kThreads      = kWarps * 32;
    static constexpr int kMmaRows      = kWarpRows / 16;
    static constexpr int kMmaCols      = kWarpCols / 8;

    static constexpr int kCrBytes = kWeightStages * kBlockRows * kCodeRowStride;
    static constexpr int kHrBytes = kWeightStages * kBlockRows * kHighRowStride;
    static constexpr int kSwBytes = kWeightStages * kBlockRows * kGroupsPerSuper * 2;
    static constexpr int kBqBytes = kActStages * kBlockCols * kBlockK;
    static constexpr int kSxBytes = kActStages * kBlockCols * 4;

    static constexpr int kCrOffset    = 0;
    static constexpr int kHrOffset    = kCrOffset + kCrBytes;
    static constexpr int kBqOffset    = kHrOffset + kHrBytes;
    static constexpr int kSxOffset    = kBqOffset + kBqBytes;
    static constexpr int kSwOffset    = kSxOffset + kSxBytes;
    static constexpr int kSharedBytes = kSwOffset + kSwBytes;

    static_assert(kBlockRows % kWarpRows == 0 && kBlockCols % kWarpCols == 0);
    static_assert(kWarpRows % 16 == 0 && kWarpCols % 8 == 0);
    static_assert(kActStages >= 3, "single-barrier pipeline needs at least three activation slots");
    static_assert(kThreads <= 1024 && kWarps >= 1);
    static_assert(kBlockCols % 16 == 0, "activation staging copies 16 columns worth of k at once");
    // Ada allows 100 KiB of dynamic shared memory per block; the driver reserves 1 KiB.
    static_assert(kSharedBytes <= 99 * 1024, "INT8 GEMM exceeds the dynamic shared budget");
};

// clang-format off
template <class Schedule_, bool Full, Int8GemmEpilogue Epilogue>
__global__ __launch_bounds__(Schedule_::kThreads, Schedule_::kMinBlocks)
void int8_rowsplit_gemm_kernel(
    const std::int8_t* __restrict__ xq,
    const float* __restrict__ xs,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high,
    const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    std::int32_t rows,
    std::int32_t cols,
    std::int32_t padded_k,
    std::int32_t out_row_stride) {
    // clang-format on
    using S               = Schedule_;
    constexpr int BM      = S::kBlockRows;
    constexpr int BN      = S::kBlockCols;
    constexpr int BK      = S::kBlockK;
    constexpr int WM      = S::kWarpRows;
    constexpr int WN      = S::kWarpCols;
    constexpr int MT      = S::kMmaRows;
    constexpr int NT      = S::kMmaCols;
    constexpr int KB      = S::kKBlocks;
    constexpr int GPS     = S::kGroupsPerSuper;
    constexpr int CRS     = S::kCodeRowStride;
    constexpr int CPR     = S::kChunksPerRow;
    constexpr int HRS     = S::kHighRowStride;
    constexpr int HCH     = S::kHighChunks;
    constexpr int WST     = S::kWeightStages;
    constexpr int AST     = S::kActStages;
    constexpr bool kHigh  = S::kHasHighPlane;
    constexpr int kGroupK = Int8RowSplitStorage::kGroupK;

    extern __shared__ __align__(16) std::uint8_t smem_raw[];
    auto* Cr = reinterpret_cast<std::uint8_t*>(smem_raw + S::kCrOffset);
    auto* Hr = reinterpret_cast<std::uint8_t*>(smem_raw + S::kHrOffset);
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
    // Q5 only: combined byte-select and bit-offset into the packed high word.
    // See the fragment loop; `+16` selects the byte for code c0 + 16.
    const int shift_lo = group_col * 4;
    const int shift_hi = shift_lo + 16;

    const int row0     = static_cast<int>(blockIdx.x) * BM;
    const int col0     = static_cast<int>(blockIdx.y) * BN;
    const int n_groups = padded_k / kGroupK;
    const int n_supers = n_groups / GPS;

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
            const int row       = row0 + local_row;
            std::uint8_t* dst   = cr_at(st, local_row, chunk);
            if (Full || row < rows) {
                const std::int64_t gi = static_cast<std::int64_t>(row) * n_groups + g0;
                cp_async<16>(dst,
                             &codes[gi * Int8RowSplitStorage::kCodeBytesPerGroup + chunk * 16]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
        if constexpr (kHigh) {
#pragma unroll 1
            for (int item = tid; item < BM * HCH; item += S::kThreads) {
                const int local_row = item / HCH;
                const int chunk     = item - local_row * HCH;
                const int row       = row0 + local_row;
                std::uint8_t* dst   = &Hr[(st * BM + local_row) * HRS + chunk * 16];
                if (Full || row < rows) {
                    const std::int64_t gi = static_cast<std::int64_t>(row) * n_groups + g0;
                    cp_async<16>(dst,
                                 &high[gi * Int8RowSplitStorage::kHighBytesPerGroup + chunk * 16]);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
#pragma unroll 1
        for (int local_row = tid; local_row < BM; local_row += S::kThreads) {
            const int row = row0 + local_row;
            std::uint16_t v[GPS];
            if (Full || row < rows) {
                const std::uint16_t* src = reinterpret_cast<const std::uint16_t*>(
                    &scales[(static_cast<std::int64_t>(row) * n_groups + g0) *
                            Int8RowSplitStorage::kScaleBytesPerGroup]);
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

    // Prologue: one commit per group, the first also carrying super-group 0.
#pragma unroll
    for (int s = 0; s < AST - 1; ++s) {
        if (s == 0) { stage_weights(0, 0); }
        if (s < n_groups) { stage_activations(s, s); }
        cp_commit();
    }

    // Iterating super-groups with the four member groups unrolled turns every
    // shared-memory index into a compile-time constant.
    for (int super = 0; super < n_supers; ++super) {
        const int wst = super % WST;
#pragma unroll
        for (int j = 0; j < GPS; ++j) {
            const int g   = super * GPS + j;
            const int ast = (AST == GPS) ? j : (g % AST);

            // Commit index j carries activation group j: the prologue issues
            // AST - 1 commits for groups 0..AST-2, and each iteration adds the
            // one for g + AST - 1. Entering iteration g, AST - 1 + g commits
            // exist and groups 0..g must have landed, so at most AST - 2 may
            // still be pending. Waiting on AST - 1 would leave group g's own
            // copy in flight. With AST slots the prefetch distance is capped at
            // AST - 1, so AST - 2 is also the deepest legal overlap; the group
            // retired here was issued a full iteration ago.
            cp_wait<AST - 2>();
            __syncthreads();

            // Past the barrier every warp has finished reading both the previous
            // group's activation slot and the previous super-group's weight
            // tile, so both are free to overwrite. Issuing the prefetches here -
            // before the MMAs rather than after them - is what removes the
            // second barrier.
            if (j == 0 && super + 1 < n_supers) { stage_weights((super + 1) % WST, super + 1); }
            const int prefetch = g + AST - 1;
            if (prefetch < n_groups) {
                stage_activations((AST == GPS) ? ((j + AST - 1) % AST) : (prefetch % AST),
                                  prefetch);
            }
            cp_commit();

            // Per-group rescale factors: outer product of the weight group scale
            // over the lane's two rows and the activation group scale over its
            // two columns. Computed once per group and reused by every
            // accumulator.
            float ws[MT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int r0 = warp_row * WM + mi * 16 + group_row;
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

            // All fragments of the group are loaded first, then each (mi, ni)
            // tile contracts over both k-steps and is rescaled immediately.
            // Keeping the INT32 accumulator scoped to one tile holds four s32
            // registers live instead of MT * NT * 4, which is what lets the warp
            // tile grow without spilling.
            unsigned af[MT][KB][4];
            unsigned bf[NT][KB][2];
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) {
                // Byte offset j * 32 + kb * 16 inside the row is exactly chunk
                // (j * 2 + kb) of the staged 128-byte line.
                const int chunk = j * (Int8RowSplitStorage::kCodeBytesPerGroup / 16) + kb;
                // Q5: the four A-fragment registers of this (mi, kb) need codes
                // c0 and c0 + 16 for two rows, where
                // c0 = j*64 + kb*32 + group_col*4. The high plane is a
                // lane-major bitstream, so their high bits live at bytes
                // base + group_col/2 and base + group_col/2 + 2 for
                // base = j*8 + kb*4. Those four bytes are one aligned 32-bit
                // word, so a single LDS.u32 replaces four byte loads and the
                // byte select folds into the shift: 8*(group_col/2) +
                // 4*(group_col&1) is exactly 4*group_col.
                const int high_word = j * 8 + kb * 4;
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    const int r0           = warp_row * WM + mi * 16 + group_row;
                    const std::uint8_t* p0 = cr_at(wst, r0, chunk) + group_col * 2;
                    const std::uint8_t* p1 = cr_at(wst, r0 + 8, chunk) + group_col * 2;
                    if constexpr (kHigh) {
                        const std::uint32_t h0 = *reinterpret_cast<const std::uint32_t*>(
                            &Hr[(wst * BM + r0) * HRS + high_word]);
                        const std::uint32_t h1 = *reinterpret_cast<const std::uint32_t*>(
                            &Hr[(wst * BM + r0 + 8) * HRS + high_word]);
                        af[mi][kb][0] = q5_codes_to_s8x4(
                            *reinterpret_cast<const std::uint16_t*>(p0), h0, shift_lo);
                        af[mi][kb][1] = q5_codes_to_s8x4(
                            *reinterpret_cast<const std::uint16_t*>(p1), h1, shift_lo);
                        af[mi][kb][2] = q5_codes_to_s8x4(
                            *reinterpret_cast<const std::uint16_t*>(p0 + 8), h0, shift_hi);
                        af[mi][kb][3] = q5_codes_to_s8x4(
                            *reinterpret_cast<const std::uint16_t*>(p1 + 8), h1, shift_hi);
                    } else {
                        af[mi][kb][0] =
                            q4_codes_to_s8x4(*reinterpret_cast<const std::uint16_t*>(p0));
                        af[mi][kb][1] =
                            q4_codes_to_s8x4(*reinterpret_cast<const std::uint16_t*>(p1));
                        af[mi][kb][2] =
                            q4_codes_to_s8x4(*reinterpret_cast<const std::uint16_t*>(p0 + 8));
                        af[mi][kb][3] =
                            q4_codes_to_s8x4(*reinterpret_cast<const std::uint16_t*>(p1 + 8));
                    }
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

    // ---- epilogue ----------------------------------------------------------
    // AddResidual reads and writes `out` in place. The projection stays in FP32
    // until it is added, so it carries one fewer rounding than a route that
    // stages the projection through BF16 first.
    auto emit = [&](int col, int row, float value) {
        const std::int64_t index = static_cast<std::int64_t>(col) * out_row_stride + row;
        if constexpr (Epilogue == Int8GemmEpilogue::AddResidual) {
            out[index] = __float2bfloat16_rn(__bfloat162float(out[index]) + value);
        } else {
            out[index] = __float2bfloat16_rn(value);
        }
    };

#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int r0 = row0 + warp_row * WM + mi * 16 + group_row;
        const int r1 = r0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int c0   = col0 + warp_col * WN + ni * 8 + 2 * group_col;
            const int c1   = c0 + 1;
            const float* v = acc[mi][ni];
            if constexpr (Full) {
                emit(c0, r0, v[0]);
                emit(c1, r0, v[1]);
                emit(c0, r1, v[2]);
                emit(c1, r1, v[3]);
            } else {
                if (r0 < rows) {
                    if (c0 < cols) { emit(c0, r0, v[0]); }
                    if (c1 < cols) { emit(c1, r0, v[1]); }
                }
                if (r1 < rows) {
                    if (c0 < cols) { emit(c0, r1, v[2]); }
                    if (c1 < cols) { emit(c1, r1, v[3]); }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
