// Adapted from tensorninja/ninfer-4090 (Apache-2.0); see docs/ada.md for provenance.

#pragma once

// Shared INT8 prefill launcher for the fused projection pairs.
//
// attn_input_proj and gdn_input_proj are each several independent row-split
// GEMMs over one shared activation: the Q4 parent supplies some output row
// ranges and the Q5 parent the rest, and every job contracts the same K. The
// activation is therefore quantized once per call and every job reads it.
//
// A job names a contiguous weight row range and where it lands. `codes`, `high`,
// and `scales` are already advanced to the job's first weight row, and `out` to
// its first output row; `rows` bounds the computation while `out_row_stride` is
// the destination tensor's leading dimension, which differs when a job writes a
// slice of a larger packed output.

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

// The quantized activation staging is O(T), so the projections process tokens in
// bounded tiles and their workspace never scales with the context length.
inline constexpr std::int32_t kInt8ProjMaxTokenTile = 4096;

constexpr std::int32_t int8_proj_token_tile(std::int32_t cols) noexcept {
    return cols < kInt8ProjMaxTokenTile ? cols : kInt8ProjMaxTokenTile;
}

// Quantizes `x` once and runs every job against it, in bounded token tiles.
// `x` is BF16 [k, cols]; each job's output is BF16 [*, cols].
void int8_proj_launch(const Tensor& x, const Int8ProjJob* jobs, int job_count,
                      const Int8ProjWorkspace& scratch, cudaStream_t stream);

} // namespace ninfer::ops::detail
