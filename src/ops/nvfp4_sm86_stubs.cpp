#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc_launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void reject_nvfp4_a4() {
    throw std::runtime_error("NVFP4 A4 execution requires an sm_120a GPU");
}

[[noreturn]] void reject_nvfp4_kv() {
    throw std::runtime_error("NVFP4 KV-cache storage requires an sm_120a GPU");
}

} // namespace

void launch_nvfp4_w4a4_quantize(const Tensor&, const Weight&, Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_w4a4(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_linear_swiglu_w4a4_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                     cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*,
                                         __nv_bfloat16*, std::int32_t, float, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_linear_add_w4a4_launch(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace,
                                  cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_attn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                  Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_gdn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                 Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

// The NVFP4 KV-cache prompt kernel lives in the sm_120a-only non-RDC archive (setmaxnreg); the
// storage mode itself is rejected at option parsing on this build, so these never run.
void causal_attention_prompt_nvfp4_kernel_launch(const Tensor&, const Tensor&, float,
                                                 const PagedKVLayerView&, Tensor&, cudaStream_t) {
    reject_nvfp4_kv();
}

void causal_attention_prompt_nvfp4_batch_kernel_launch(const Tensor&, const Tensor&, const Tensor&,
                                                       const Tensor&, float,
                                                       const PagedKVBatchLayerView&, Tensor&,
                                                       cudaStream_t) {
    reject_nvfp4_kv();
}

// The small-T decode kernels and the append kernels for both NVFP4-valued modes quantize with
// cvt.e2m1x2, which sm_89 lacks, so their translation units are excluded from this build.
void causal_attention_small_t_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&,
                                           const Tensor&, const Tensor&, const Tensor&, float,
                                           PagedKVBatchLayerView, CausalAttentionExecutionEnvelope,
                                           std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
                                           Tensor&, cudaStream_t) {
    reject_nvfp4_kv();
}

void causal_attention_cached_small_t_nvfp4_launch(const Tensor&, const Tensor&, float,
                                                  const PagedKVLayerView&,
                                                  CausalAttentionExecutionEnvelope, Tensor&,
                                                  Tensor&, Tensor&, Tensor&, cudaStream_t) {
    reject_nvfp4_kv();
}

void causal_attention_small_t_k8v4_launch(const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, const Tensor&, const Tensor&, float,
                                          PagedKVBatchLayerView, CausalAttentionExecutionEnvelope,
                                          std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
                                          Tensor&, cudaStream_t) {
    reject_nvfp4_kv();
}

void causal_attention_cached_small_t_k8v4_launch(const Tensor&, const Tensor&, float,
                                                 const PagedKVLayerView&,
                                                 CausalAttentionExecutionEnvelope, Tensor&,
                                                 Tensor&, Tensor&, Tensor&, cudaStream_t) {
    reject_nvfp4_kv();
}

void kv_cache_append_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                                  cudaStream_t) {
    reject_nvfp4_kv();
}

void kv_cache_append_nvfp4_batch_launch(const Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, const Tensor&, PagedKVBatchLayerView,
                                        cudaStream_t) {
    reject_nvfp4_kv();
}

void kv_cache_append_k8v4_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                                 cudaStream_t) {
    reject_nvfp4_kv();
}

void kv_cache_append_k8v4_batch_launch(const Tensor&, const Tensor&, const Tensor&,
                                       const Tensor&, const Tensor&, PagedKVBatchLayerView,
                                       cudaStream_t) {
    reject_nvfp4_kv();
}

} // namespace ninfer::ops::detail
