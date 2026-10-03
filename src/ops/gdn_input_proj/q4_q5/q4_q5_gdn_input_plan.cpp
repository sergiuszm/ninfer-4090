#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/layout.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4Q5GdnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 2> kA16Routes{{
    {{1, 15}, Q4Q5GdnInputScheduleId::IndependentDirectFixed},
    {{16, kAnyCols}, Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128},
}};

// AllowA8 uses private group-64 activation quantization at every token extent.
// Keeping the compute profile independent of T avoids a precision switch when
// prefix reuse changes the width of a prefill suffix.
constexpr std::array<RouteSpec, 1> kA8Routes{{
    {{1, kAnyCols}, Q4Q5GdnInputScheduleId::Int8Jobs},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes) noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return routes.back().cols.last == kAnyCols &&
           expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(kA16Routes) && catalog_is_closed(kA8Routes),
              "GDN input routes must be exact and closed");

template <class Visit>
auto visit_routes(LinearPolicy policy, Visit&& visit) {
    if (policy == LinearPolicy::AllowA8) { return visit(kA8Routes); }
    return visit(kA16Routes);
}

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

bool supported_shape(const Q4Q5GdnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qk_rows == 4096 && problem.value_z_rows == 12288 &&
           problem.qkv_rows == 10240 && problem.z_rows == 6144 && problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_gdn_input_schedule_name(Q4Q5GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed:
        return "gdn_input_proj.q4_q5.independent_direct_fixed";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c128";
    case Q4Q5GdnInputScheduleId::Int8Jobs:
        return "gdn_input_proj.q4_q5.int8.jobs";
    }
    return "gdn_input_proj.q4_q5.unknown";
}

const char* q4_q5_gdn_input_conv_schedule_name(Q4Q5GdnInputConvScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused:
        return "gdn_input_proj_conv.q4_q5.projection_epilogue_fused";
    case Q4Q5GdnInputConvScheduleId::Materialized:
        return "gdn_input_proj_conv.q4_q5.materialized";
    }
    return "gdn_input_proj_conv.q4_q5.unknown";
}

bool q4_q5_gdn_input_admits(const Q4Q5GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5GdnInputPlan q4_q5_gdn_input_resolve_plan(const Q4Q5GdnInputProblem& problem,
                                              LinearPolicy policy) {
    if (!q4_q5_gdn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input: exact problem or column count is not admitted");
    }

    return visit_routes(policy, [&](const auto& routes) -> Q4Q5GdnInputPlan {
        for (const RouteSpec& route : routes) {
            if (!route.cols.contains(problem.cols)) { continue; }
            if (route.schedule == Q4Q5GdnInputScheduleId::Int8Jobs) {
                return {route.schedule, int8_workspace_bytes(problem.input_rows, problem.cols)};
            }
            return {route.schedule, 0};
        }
        throw std::logic_error("Q4/Q5 GDN input: admitted problem has no covering route");
    });
}

std::size_t q4_q5_gdn_input_capacity_workspace_bytes(std::int32_t min_tokens,
                                                     std::int32_t max_tokens, LinearPolicy policy) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("Q4/Q5 GDN input: invalid token interval");
    }
    const Q4Q5GdnInputProblem lo{5120, 4096, 12288, 10240, 6144, 5120, min_tokens};
    const Q4Q5GdnInputProblem hi{5120, 4096, 12288, 10240, 6144, 5120, max_tokens};
    (void)q4_q5_gdn_input_resolve_plan(lo, policy);
    return q4_q5_gdn_input_resolve_plan(hi, policy).workspace_bytes;
}

Q4Q5GdnInputConvPlan q4_q5_gdn_input_conv_resolve_plan(const Q4Q5GdnInputProblem& problem,
                                                       std::int32_t batch_size) {
    if (!q4_q5_gdn_input_admits(problem) || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input conv: exact problem or column count is not admitted");
    }
    if (batch_size > 1) { return {Q4Q5GdnInputConvScheduleId::Materialized}; }
    switch (problem.cols) {
    case 1:
    case 2:
    case 3:
    case 5:
    case 6:
        return {Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused};
    default:
        return {Q4Q5GdnInputConvScheduleId::Materialized};
    }
}

void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& value_z_weight,
                                  Tensor& qkv, Tensor& z, WorkspaceArena* ws, LinearPolicy policy,
                                  cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan resolved = q4_q5_gdn_input_resolve_plan(problem, policy);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("Q4/Q5 GDN input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5GdnInputScheduleId::Int8Jobs: {
        if (ws == nullptr) {
            throw std::invalid_argument("Q4/Q5 GDN input: INT8 route requires a workspace");
        }
        auto scratch_scope = ws->scope();
        const Int8ProjWorkspace scratch =
            allocate_int8_workspace(*ws, problem.input_rows, problem.cols);
        q4_q5_gdn_input_int8_launch(x, qk_weight, value_z_weight, qkv, z, scratch, stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed: {
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        q4_q5_gdn_input_independent_launch(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        q4_q5_gdn_input_grouped_mma_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    }
    throw std::logic_error("Q4/Q5 GDN input: unknown schedule");
}

void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              WorkspaceArena* ws, LinearPolicy policy, cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan plan = q4_q5_gdn_input_resolve_plan(problem, policy);
    q4_q5_gdn_input_execute_plan(plan, x, qk_weight, value_z_weight, qkv, z, ws, policy, stream);
}

} // namespace ninfer::ops::detail
