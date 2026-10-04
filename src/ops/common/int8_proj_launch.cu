// Adapted from tensorninja/ninfer-4090 and soohl/ninfer (both Apache-2.0):
// the group-64 INT8 activation prefill.

#include "ops/common/int8_proj_launch.h"

#include "ops/common/act_quant_g64.h"
#include "ops/common/int8_rowsplit_gemm.cuh"

#include "core/device.h"
#include "ops/common/math.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Use smaller row tiles for the 1024-row K/V jobs to expose more CTAs.
constexpr std::int32_t kWideRowThreshold = 2048;

using Q4Wide   = Int8RowSplitGemmSchedule<Int8GemmCodec::Q4, 128, 128, 32, 64, 3, 1>;
using Q4Narrow = Int8RowSplitGemmSchedule<Int8GemmCodec::Q4, 64, 64, 16, 64, 3, 1>;
using Q5Wide   = Int8RowSplitGemmSchedule<Int8GemmCodec::Q5, 128, 128, 32, 64, 3, 1>;
using Q5Narrow = Int8RowSplitGemmSchedule<Int8GemmCodec::Q5, 64, 64, 16, 64, 3, 1>;

template <class Cfg, bool Full>
void launch_job(const Int8ProjJob& job, const std::int8_t* xq, const float* xs, std::int32_t tokens,
                std::int32_t padded_k, cudaStream_t stream) {
    static const bool configured = [] {
        CUDA_CHECK(
            cudaFuncSetAttribute(int8_rowsplit_gemm_kernel<Cfg, Full, Int8GemmEpilogue::Store>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kSharedBytes));
        return true;
    }();
    (void)configured;

    const dim3 grid(static_cast<unsigned>(div_up(job.rows, Cfg::kBlockRows)),
                    static_cast<unsigned>(div_up(tokens, Cfg::kBlockCols)));
    int8_rowsplit_gemm_kernel<Cfg, Full, Int8GemmEpilogue::Store>
        <<<grid, Cfg::kThreads, Cfg::kSharedBytes, stream>>>(xq, xs, job.codes, job.high,
                                                             job.scales, job.out, job.rows, tokens,
                                                             padded_k, job.out_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

template <class Cfg>
void launch_sized(const Int8ProjJob& job, const std::int8_t* xq, const float* xs,
                  std::int32_t tokens, std::int32_t padded_k, cudaStream_t stream) {
    const bool full = (tokens % Cfg::kBlockCols) == 0 && (job.rows % Cfg::kBlockRows) == 0;
    if (full) {
        launch_job<Cfg, true>(job, xq, xs, tokens, padded_k, stream);
    } else {
        launch_job<Cfg, false>(job, xq, xs, tokens, padded_k, stream);
    }
}

} // namespace

void int8_proj_launch(const Tensor& x, const Int8ProjJob* jobs, int job_count,
                      const Int8ProjWorkspace& scratch, cudaStream_t stream) {
    const std::int32_t k        = x.ne[0];
    const std::int32_t padded_k = int8_proj_padded_k(k);
    const std::int32_t tile     = int8_proj_token_tile(x.ne[1]);

    for (std::int32_t offset = 0; offset < x.ne[1]; offset += tile) {
        const std::int32_t count = std::min(tile, x.ne[1] - offset);
        const Tensor x_slice     = x.slice(1, offset, count);
        act_quant_g64_launch(static_cast<const __nv_bfloat16*>(x_slice.data), scratch.codes,
                             scratch.scales, k, count, padded_k, stream);
        for (int i = 0; i < job_count; ++i) {
            Int8ProjJob job = jobs[i];
            job.out += static_cast<std::int64_t>(offset) * job.out_row_stride;
            const bool wide = job.rows >= kWideRowThreshold;
            if (job.q5) {
                if (wide) {
                    launch_sized<Q5Wide>(job, scratch.codes, scratch.scales, count, padded_k,
                                         stream);
                } else {
                    launch_sized<Q5Narrow>(job, scratch.codes, scratch.scales, count, padded_k,
                                           stream);
                }
            } else {
                if (wide) {
                    launch_sized<Q4Wide>(job, scratch.codes, scratch.scales, count, padded_k,
                                         stream);
                } else {
                    launch_sized<Q4Narrow>(job, scratch.codes, scratch.scales, count, padded_k,
                                           stream);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
