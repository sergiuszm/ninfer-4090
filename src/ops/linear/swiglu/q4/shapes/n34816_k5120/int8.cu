// Adapted from tensorninja/ninfer-4090 and soohl/ninfer (both Apache-2.0):
// the group-64 INT8 activation prefill (fork PR #15, see docs/maintainer/port-ledger.md).

#include "ops/linear/swiglu/q4/launch.h"

#include "ops/linear/kernels/int8/act_quant_g64.h"
#include "ops/linear/swiglu/q4/int8_folded_mma.cuh"

#include "core/device.h"
#include "ops/common/host/launch.h"
#include "ops/common/numeric/math.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Each warp owns matching gate/up rows and reuses decoded weights across tokens.
using Int8FoldedCfg = Q4Int8SwiGluSchedule<64, 256, 16, 128, 3, 1>;
// Short prompts benefit from two smaller resident CTAs rather than a mostly masked tile.
using Int8ShortCfg = Q4Int8SwiGluSchedule<32, 128, 16, 64, 3, 2>;

template <class Cfg, bool Full>
void launch_folded_int8(const std::int8_t* xq, const float* xs, const Weight& weight, Tensor& out,
                        std::int32_t tokens, std::int32_t padded_k, cudaStream_t stream) {
    constexpr auto kernel = q4_linear_swiglu_int8_gemm_kernel<Cfg, Full>;
    (void)common::prepare_shared<Cfg::kSharedBytes, kernel, true>();

    const dim3 grid(static_cast<unsigned>(common::div_up(out.ne[0], Cfg::kPairRows)),
                    static_cast<unsigned>(common::div_up(tokens, Cfg::kBlockCols)));
    kernel<<<grid, Cfg::kThreads, Cfg::kSharedBytes, stream>>>(
        xq, xs, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        out.ne[0], tokens, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Cfg>
void launch_sized(const std::int8_t* xq, const float* xs, const Weight& weight, Tensor& out,
                  std::int32_t tokens, std::int32_t padded_k, cudaStream_t stream) {
    const bool full = tokens % Cfg::kBlockCols == 0 && out.ne[0] % Cfg::kPairRows == 0;
    if (full) {
        launch_folded_int8<Cfg, true>(xq, xs, weight, out, tokens, padded_k, stream);
    } else {
        launch_folded_int8<Cfg, false>(xq, xs, weight, out, tokens, padded_k, stream);
    }
}

} // namespace

void launch_q4_swiglu_int8(const Tensor& x, const Weight& weight, Tensor& out,
                           Int8ProjWorkspace scratch, cudaStream_t stream) {
    const std::int32_t k        = x.ne[0];
    const std::int32_t padded_k = int8_proj_padded_k(k);

    // The token loop is bounded by the workspace tile, not by the CUDA grid
    // limit: the quantized activation staging is O(tokens) and must not scale
    // with the context length. Each tile is quantized and contracted before the
    // next reuses the same staging buffers.
    const std::int32_t tile = int8_proj_token_tile(x.ne[1]);
    for (std::int32_t offset = 0; offset < x.ne[1]; offset += tile) {
        const std::int32_t count = std::min(tile, x.ne[1] - offset);
        const Tensor x_slice     = x.slice(1, offset, count);
        Tensor out_slice         = out.slice(1, offset, count);
        act_quant_g64_launch(static_cast<const __nv_bfloat16*>(x_slice.data), scratch.codes,
                             scratch.scales, k, count, padded_k, stream);
        if (count <= Int8ShortCfg::kBlockCols) {
            launch_sized<Int8ShortCfg>(scratch.codes, scratch.scales, weight, out_slice, count,
                                       padded_k, stream);
        } else {
            launch_sized<Int8FoldedCfg>(scratch.codes, scratch.scales, weight, out_slice, count,
                                        padded_k, stream);
        }
    }
}

} // namespace ninfer::ops::detail
