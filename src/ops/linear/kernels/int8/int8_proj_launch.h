// Adapted from tensorninja/ninfer-4090 and soohl/ninfer (both Apache-2.0):
// the group-64 INT8 activation prefill (fork PR #15, see docs/maintainer/port-ledger.md).

#pragma once

// Shared INT8 prefill staging and the job launcher for the paired Q4/Q5 projections.
//
// Every INT8 route quantizes its BF16 activation once per token tile into caller-owned
// workspace (act_quant_g64) and contracts whole 256-element weight super-groups. The staging
// geometry below is the one authority for that workspace: the routes' capacity queries and
// their execution allocate it through the same helper.
//
// The paired linear_split shapes (attention and GDN input projections) are several
// independent row-split GEMMs over one shared activation: the Q4 parent supplies some output
// row ranges and the Q5 parent the rest, and every job contracts the same K. The activation
// is therefore quantized once per tile and every job reads it.
//
// A job names a contiguous weight row range and where it lands. `codes`, `high`, and
// `scales` are already advanced to the job's first weight row, and `out` to its first output
// row; `rows` bounds the computation while `out_row_stride` is the destination tensor's
// leading dimension, which differs when a job writes a slice of a larger packed output.

#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

struct Int8ProjJob {
    const std::uint8_t* codes   = nullptr;
    const std::uint8_t* high    = nullptr; // Q5 only
    const std::uint8_t* scales  = nullptr;
    __nv_bfloat16* out          = nullptr;
    std::int32_t rows           = 0;
    std::int32_t out_row_stride = 0;
    bool q5                     = false;
};

// Caller-owned staging for the quantized activation, shared by every job.
struct Int8ProjWorkspace {
    std::int8_t* codes = nullptr;
    float* scales      = nullptr;
};

// The INT8 GEMM contracts whole 256-element weight super-groups.
constexpr std::int32_t int8_proj_padded_k(std::int32_t k) noexcept {
    return ((k + 255) / 256) * 256;
}

// The quantized activation staging is O(T), so the routes process tokens in bounded tiles
// and their workspace never scales with the context length. At K = 17408 a tile costs
// 18,496 bytes per token, which holds the workspace at 72.25 MiB for any prefill chunk.
inline constexpr std::int32_t kInt8ProjMaxTokenTile = 4096;

constexpr std::int32_t int8_proj_token_tile(std::int32_t cols) noexcept {
    return cols < kInt8ProjMaxTokenTile ? cols : kInt8ProjMaxTokenTile;
}

// Group-64 quantized activation codes plus their FP32 group scales. Groups past K are
// zero-filled by the quantizer and carry a zero scale.
template <class Arena>
Int8ProjWorkspace allocate_int8_proj_workspace(Arena& arena, std::int32_t k, std::int32_t cols) {
    const std::int32_t padded_k = int8_proj_padded_k(k);
    const std::int32_t tile     = int8_proj_token_tile(cols);
    Tensor codes                = arena.alloc(DType::I8, {padded_k, tile});
    Tensor scales               = arena.alloc(DType::FP32, {padded_k / 64, tile});
    return {static_cast<std::int8_t*>(codes.data), static_cast<float*>(scales.data)};
}

// The staging grows with the token count up to the tile cap, so an interval's high-water
// mark is at its last token count.
inline std::size_t int8_proj_workspace_capacity_bytes(std::int32_t k, std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_int8_proj_workspace(layout, k, last);
    return layout.peak_bytes(1);
}

// Quantizes `x` once per token tile and runs every job against it.
// `x` is BF16 [k, cols]; each job's output is BF16 [*, cols].
void int8_proj_launch(const Tensor& x, const Int8ProjJob* jobs, int job_count,
                      const Int8ProjWorkspace& scratch, cudaStream_t stream);

} // namespace ninfer::ops::detail
