// Adapted from tensorninja/ninfer-4090 and soohl/ninfer (both Apache-2.0):
// the group-64 INT8 activation prefill (fork PR #15, see docs/maintainer/port-ledger.md).

#include "core/weight.h"
#include "ops/linear/split/q4_q5/shapes/n7168_n7168_k5120/launch.h"

#include <cstdint>

namespace ninfer::ops::detail::linear_split::q4_q5_n7168_n7168_k5120 {
namespace {

// Both parents are [7168, 5120]: rows [0, 6144) are the query/gate half and rows
// [6144, 7168) the key/value half. Each half writes its own contiguous output,
// so the four jobs share the quantized activation but nothing else.
constexpr std::int32_t kQueryRows = 6144;
constexpr std::int32_t kKvRows    = 1024;

Int8ProjJob make_job(const Weight& weight, std::int32_t row_begin, std::int32_t row_count,
                     const Tensor& out) {
    const std::int64_t groups = weight.padded_shape[1] / 64;
    const bool q5             = weight.qtype == QType::Q5_G64_FP16;
    return Int8ProjJob{
        static_cast<const std::uint8_t*>(weight.qdata) +
            static_cast<std::int64_t>(row_begin) * groups * 32,
        q5 ? static_cast<const std::uint8_t*>(weight.qhigh) +
                 static_cast<std::int64_t>(row_begin) * groups * 8
           : nullptr,
        static_cast<const std::uint8_t*>(weight.scales) +
            static_cast<std::int64_t>(row_begin) * groups * 2,
        static_cast<__nv_bfloat16*>(out.data),
        row_count,
        out.ne[0],
        q5,
    };
}

} // namespace

void launch_int8(const Tensor& x, const Weight& query_key, const Weight& gate_value,
                 std::span<const Tensor> out, Int8ProjWorkspace scratch, cudaStream_t stream) {
    // Outputs are ordered {query, key, gate, value}.
    const Int8ProjJob jobs[4] = {
        make_job(query_key, 0, kQueryRows, out[0]),
        make_job(gate_value, 0, kQueryRows, out[2]),
        make_job(query_key, kQueryRows, kKvRows, out[1]),
        make_job(gate_value, kQueryRows, kKvRows, out[3]),
    };
    int8_proj_launch(x, jobs, 4, scratch, stream);
}

} // namespace ninfer::ops::detail::linear_split::q4_q5_n7168_n7168_k5120
