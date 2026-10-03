// Adapted from tensorninja/ninfer-4090 (Apache-2.0); see docs/ada.md for provenance.

#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "ops/common/act_quant_g64.h"
#include "ops/common/int8_rowsplit_gemm.cuh"

#include "core/device.h"
#include "ops/common/math.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// A 16x64 warp tile limits accumulator pressure in the residual epilogue.
using Int8ResidualCfg = Int8RowSplitGemmSchedule<Int8GemmCodec::Q5, 64, 128, 16, 64, 3, 1>;
constexpr Int8GemmEpilogue kResidual = Int8GemmEpilogue::AddResidual;

template <class Cfg, bool Full>
void launch_int8(const std::int8_t* xq, const float* xs, const Weight& weight, Tensor& residual_out,
                 std::int32_t tokens, std::int32_t padded_k, cudaStream_t stream) {
    static const bool configured = [] {
        CUDA_CHECK(cudaFuncSetAttribute(int8_rowsplit_gemm_kernel<Cfg, Full, kResidual>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Cfg::kSharedBytes));
        return true;
    }();
    (void)configured;

    const dim3 grid(static_cast<unsigned>(div_up(residual_out.ne[0], Cfg::kBlockRows)),
                    static_cast<unsigned>(div_up(tokens, Cfg::kBlockCols)));
    int8_rowsplit_gemm_kernel<Cfg, Full, kResidual>
        <<<grid, Cfg::kThreads, Cfg::kSharedBytes, stream>>>(
            xq, xs, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(residual_out.data), residual_out.ne[0], tokens, padded_k,
            residual_out.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q5_linear_add_int8_residual_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        const Q5LinearAddInt8Workspace& scratch,
                                        cudaStream_t stream) {
    using Cfg                   = Int8ResidualCfg;
    const std::int32_t k        = x.ne[0];
    const std::int32_t padded_k = q5_linear_add_int8_padded_k(k);

    // The token loop is bounded by the workspace tile, not by the CUDA grid
    // limit: the quantized activation staging is O(tokens) and must not scale
    // with the context length. Each tile is quantized and contracted before the
    // next reuses the same staging buffers. The residual add is in place, so the
    // slices are disjoint and the order across tiles does not matter.
    const std::int32_t tile = q5_linear_add_int8_token_tile(x.ne[1]);
    for (std::int32_t offset = 0; offset < x.ne[1]; offset += tile) {
        const std::int32_t count = std::min(tile, x.ne[1] - offset);
        const Tensor x_slice     = x.slice(1, offset, count);
        Tensor out_slice         = residual_out.slice(1, offset, count);
        act_quant_g64_launch(static_cast<const __nv_bfloat16*>(x_slice.data), scratch.codes,
                             scratch.scales, k, count, padded_k, stream);
        const bool full =
            (count % Cfg::kBlockCols) == 0 && (residual_out.ne[0] % Cfg::kBlockRows) == 0;
        if (full) {
            launch_int8<Cfg, true>(scratch.codes, scratch.scales, w, out_slice, count, padded_k,
                                   stream);
        } else {
            launch_int8<Cfg, false>(scratch.codes, scratch.scales, w, out_slice, count, padded_k,
                                    stream);
        }
    }
}

} // namespace ninfer::ops::detail
