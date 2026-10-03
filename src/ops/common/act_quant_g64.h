// Adapted from tensorninja/ninfer-4090 (Apache-2.0); see docs/ada.md for provenance.

#pragma once

// Group-64 symmetric INT8 activation quantization for prefill GEMM routes.
//
//   x[K, T] (BF16, K contiguous)
//     -> xq[padded_k, T] (INT8)  +  xs[n_groups, T] (FP32, T contiguous)
//
//   s[g, t]  = max(|x[64g .. 64g+63, t]|) / 127
//   xq[k, t] = round_to_nearest_even(x[k, t] / s[g, t])          , s > 0
//   xq[k, t] = 0                                                 , s == 0
//
// The 64-channel group matches Q4G64/Q5G64 exactly, so one activation scale
// spans one weight group and a quantized GEMM rescales with a single product.
// Group-wise scales confine an outlier channel's precision loss to its own 64
// channels instead of the whole token.
//
// This is a private staging step beneath a quantized GEMM Op, not an Op: it has
// no independent caller and its result is never observable outside the
// enclosing contraction. It is shared by every INT8 route, so it is one
// compilation unit rather than a header each route would re-emit.
//
// `padded_k` must be a multiple of 64; the caller pads to whatever boundary its
// GEMM contracts. Groups past K are zero-filled and carry a zero scale, so they
// contribute exactly zero.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void act_quant_g64_launch(const __nv_bfloat16* x, std::int8_t* xq, float* xs, std::int32_t k,
                          std::int32_t cols, std::int32_t padded_k, cudaStream_t stream);

} // namespace ninfer::ops::detail
