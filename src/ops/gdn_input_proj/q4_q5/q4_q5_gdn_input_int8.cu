// Adapted from tensorninja/ninfer-4090 and soohl/ninfer (both Apache-2.0):
// the group-64 INT8 activation prefill.

#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "ops/common/int8_proj_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// The Q4 parent [4096, 5120] is the query/key half and lands at the top of the
// packed qkv output. The Q5 parent [12288, 5120] holds value rows [0, 6144),
// which land in qkv below the query/key half, and output-gate rows
// [6144, 12288), which land in their own z output.
constexpr std::int32_t kValueRows = 6144;

Int8ProjJob make_job(const Weight& weight, std::int32_t weight_row_begin, std::int32_t row_count,
                     Tensor& out, std::int32_t out_row_begin) {
    const std::int64_t groups = weight.padded_shape[1] / weight.group;
    const bool q5             = weight.qtype == QType::Q5G64_F16S;
    return Int8ProjJob{
        static_cast<const std::uint8_t*>(weight.qdata) +
            static_cast<std::int64_t>(weight_row_begin) * groups * 32,
        q5 ? static_cast<const std::uint8_t*>(weight.qhigh) +
                 static_cast<std::int64_t>(weight_row_begin) * groups * 8
           : nullptr,
        static_cast<const std::uint8_t*>(weight.scales) +
            static_cast<std::int64_t>(weight_row_begin) * groups * 2,
        static_cast<__nv_bfloat16*>(out.data) + out_row_begin,
        row_count,
        out.ne[0],
        q5,
    };
}

} // namespace

void q4_q5_gdn_input_int8_launch(const Tensor& x, const Weight& qk_weight,
                                 const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                 const Int8ProjWorkspace& scratch, cudaStream_t stream) {
    const Int8ProjJob jobs[3] = {
        make_job(qk_weight, 0, qk_weight.n, qkv, 0),
        make_job(value_z_weight, 0, kValueRows, qkv, qk_weight.n),
        make_job(value_z_weight, kValueRows, kValueRows, z, 0),
    };
    int8_proj_launch(x, jobs, 3, scratch, stream);
}

} // namespace ninfer::ops::detail
