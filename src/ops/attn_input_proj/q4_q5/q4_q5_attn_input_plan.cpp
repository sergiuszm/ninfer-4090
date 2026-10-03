#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/layout.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Allocator>
Int8ProjWorkspace allocate_int8_workspace(Allocator& allocator, std::int32_t k, std::int32_t cols) {
    const std::int32_t padded_k = int8_proj_padded_k(k);
    const std::int32_t groups   = padded_k / 64;
    const std::int32_t tile     = int8_proj_token_tile(cols);
    Tensor codes                = allocator.alloc(DType::I8, {padded_k, tile});
    Tensor scales               = allocator.alloc(DType::FP32, {groups, tile});
    return {static_cast<std::int8_t*>(codes.data), static_cast<float*>(scales.data)};
}

std::size_t int8_workspace_bytes(std::int32_t k, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_int8_workspace(layout, k, cols);
    return layout.peak_bytes(1);
}

bool supported_shape(const Q4Q5AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 6144 && problem.kv_rows == 1024 &&
           problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        return "attn_input_proj.q4_q5.parent_split_fixed";
    case Q4Q5AttnInputScheduleId::MixedR32C64S3:
        return "attn_input_proj.q4_q5.mixed.r32.c64.s3";
    case Q4Q5AttnInputScheduleId::PairR32C64S3:
        return "attn_input_proj.q4_q5.pair.r32.c64.s3";
    case Q4Q5AttnInputScheduleId::MixedR64C128S2:
        return "attn_input_proj.q4_q5.mixed.r64.c128.s2";
    case Q4Q5AttnInputScheduleId::PairR32C64S4:
        return "attn_input_proj.q4_q5.pair.r32.c64.s4";
    case Q4Q5AttnInputScheduleId::Int8Pairs:
        return "attn_input_proj.q4_q5.int8.pairs";
    }
    return "attn_input_proj.q4_q5.unknown";
}

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem,
                                                LinearPolicy policy) {
    if (!q4_q5_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 attention input: exact problem or column count is not admitted");
    }

    // AllowA8 (prefill) uses group-64 INT8 activations at every token extent, so prefix reuse
    // changing a suffix's width never switches precision. A16 keeps this fork's sm_89 routes.
    if (policy == LinearPolicy::AllowA8) {
        return {Q4Q5AttnInputScheduleId::Int8Pairs,
                int8_workspace_bytes(problem.input_rows, problem.cols)};
    }
    if (problem.cols <= 12) return {Q4Q5AttnInputScheduleId::ParentSplitFixed, 0};
    if (problem.cols <= 64) return {Q4Q5AttnInputScheduleId::MixedR32C64S3, 0};
    if (problem.cols <= 104) return {Q4Q5AttnInputScheduleId::PairR32C64S3, 0};
    if (problem.cols <= 128 || problem.cols >= 193)
        return {Q4Q5AttnInputScheduleId::MixedR64C128S2, 0};
    return {Q4Q5AttnInputScheduleId::PairR32C64S4, 0};
}

std::size_t q4_q5_attn_input_capacity_workspace_bytes(std::int32_t min_tokens,
                                                      std::int32_t max_tokens,
                                                      LinearPolicy policy) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("Q4/Q5 attention input: invalid token interval");
    }
    const Q4Q5AttnInputProblem lo{5120, 6144, 1024, 5120, min_tokens};
    const Q4Q5AttnInputProblem hi{5120, 6144, 1024, 5120, max_tokens};
    (void)q4_q5_attn_input_resolve_plan(lo, policy);
    // The INT8 staging grows with the token count up to the tile cap, so the
    // interval's high-water mark is at max_tokens; every other route is
    // workspace-free.
    return q4_q5_attn_input_resolve_plan(hi, policy).workspace_bytes;
}

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   WorkspaceArena* ws, LinearPolicy policy, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan resolved = q4_q5_attn_input_resolve_plan(problem, policy);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("Q4/Q5 attention input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5AttnInputScheduleId::Int8Pairs: {
        if (ws == nullptr) {
            throw std::invalid_argument("Q4/Q5 attention input: INT8 route requires a workspace");
        }
        auto scratch_scope = ws->scope();
        const Int8ProjWorkspace scratch =
            allocate_int8_workspace(*ws, problem.input_rows, problem.cols);
        q4_q5_attn_input_int8_launch(x, query_key_weight, gate_value_weight, q, gate, k, v, scratch,
                                     stream);
        return;
    }
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        q4_q5_attn_input_small_t_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR32C64S3:
        q4_q5_attn_input_mixed_r32_c64_s3_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                 v, stream);
        return;
    case Q4Q5AttnInputScheduleId::PairR32C64S3:
        q4_q5_attn_input_pair_r32_c64_s3_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                v, stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR64C128S2:
        q4_q5_attn_input_mixed_r64_c128_s2_launch(x, query_key_weight, gate_value_weight, q, gate,
                                                  k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::PairR32C64S4:
        q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    }
    throw std::logic_error("Q4/Q5 attention input: unknown schedule");
}

void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, WorkspaceArena* ws, LinearPolicy policy,
                               cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan plan = q4_q5_attn_input_resolve_plan(problem, policy);
    q4_q5_attn_input_execute_plan(plan, x, query_key_weight, gate_value_weight, q, gate, k, v, ws,
                                  policy, stream);
}

} // namespace ninfer::ops::detail
