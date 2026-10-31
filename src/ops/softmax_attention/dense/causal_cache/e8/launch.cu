// Fork-local (rtx4090-port): causal cached attention over the fork KV storages. The small-T
// kernels append the new keys and values themselves; the prompt route appends through
// kv::launch_fork_write and then reads the whole history from the cache. Launch code from
// rtx4090-port small_t.cu, prompt.cu and causal_softmax_attention.cpp, int8 family only.
#include "ops/softmax_attention/dense/causal_cache/e8/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "core/paged_kv_storage.h"
#include "ops/common/numeric/math.h"
#include "ops/kv_cache/kernels/fork_write.h"
#include "ops/softmax_attention/dense/causal_cache/e8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/e8/small_t_launch.cuh"
#include "ops/softmax_attention/kernels/causal/e8/prompt_i8.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

using e8_launch::E8KvCacheView;
using e8_launch::SmallTInvocation;
using e8_launch::small_t_launch_for;

template <typename CacheInput>
void small_t_dispatch(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                      KvCacheStorage storage, E8KvCacheView<CacheInput::writes_cache> cache,
                      const SmallTInvocation& invocation,
                      CausalAttentionExecutionEnvelope envelope, CausalPartialView partial,
                      Tensor& out, cudaStream_t stream) {
    if (q.ne[1] == e8::CausalD256H24Kv4::QHeads) {
        small_t_launch_for<e8::CausalD256H24Kv4>(q, input, pos, scale, storage, cache, invocation,
                                                 envelope, partial, out, stream);
        return;
    }
    small_t_launch_for<e8::CausalD256H16Kv2>(q, input, pos, scale, storage, cache, invocation,
                                             envelope, partial, out, stream);
}

void small_t_append(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                    const Tensor& valid_columns, const Tensor& table_rows, float scale,
                    PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
                    std::int32_t column_begin, std::int32_t width, CausalPartialView partial,
                    Tensor& out, cudaStream_t stream) {
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    const SmallTInvocation invocation{
        .full_width   = q.ne[2],
        .column_begin = column_begin,
        .width        = width,
        .batch_size   = q.ne[3],
    };
    small_t_dispatch(
        q, input, pos, scale, cache.storage,
        make_quantized_causal_cache_view<E8KvCacheView<true>>(cache, &valid_columns, &table_rows),
        invocation, envelope, partial, out, stream);
}

void small_t_cached(const Tensor& q, const Tensor& pos, float scale,
                    const ConstPagedKVLayerView& cache, CausalAttentionExecutionEnvelope envelope,
                    CausalPartialView partial, Tensor& out, cudaStream_t stream) {
    const SmallTInvocation invocation{
        .full_width   = q.ne[2],
        .column_begin = 0,
        .width        = q.ne[2],
        .batch_size   = 1,
    };
    small_t_dispatch(q, CausalCachedInput{}, pos, scale, cache.storage,
                     make_quantized_causal_cache_view<E8KvCacheView<false>>(
                         single_row_paged_kv_batch_view(cache)),
                     invocation, envelope, partial, out, stream);
}

template <typename Geometry, typename Cache, typename Metadata>
void prompt_attention_for(const Tensor& q, const Tensor& positions, float scale,
                          const Cache& cache, Metadata metadata, Tensor& out,
                          cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const dim3 grid(static_cast<unsigned>(common::div_up(tokens, e8::kCausalPromptI8Br)),
                    static_cast<unsigned>(Geometry::QHeads), 1U);
    const auto launch = [&]<KvCacheStorage Storage>() {
        const auto kernel = e8::causal_attention_prompt_i8_kernel<Geometry, Storage, Metadata>;
        static const cudaError_t attr = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, e8::kCausalPromptI8SmemBytes);
        CUDA_CHECK(attr);
        kernel<<<grid, e8::kCausalPromptI8Threads, e8::kCausalPromptI8SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::uint8_t*>(cache.k_pages.data()),
            static_cast<const std::uint8_t*>(cache.v_pages.data()),
            static_cast<const __half*>(cache.k_scale_pages.data()),
            static_cast<const __half*>(cache.v_scale_pages.data()), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
    };
    switch (cache.storage) {
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        launch.template operator()<KvCacheStorage::RotatedInt8KeyInt4ValueGroup64>();
        break;
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
        launch.template operator()<KvCacheStorage::RotatedInt4KeyInt4ValueGroup64>();
        break;
    case KvCacheStorage::RK4V4E8:
        launch.template operator()<KvCacheStorage::RK4V4E8>();
        break;
    case KvCacheStorage::RK2V4E8:
        launch.template operator()<KvCacheStorage::RK2V4E8>();
        break;
    default:
        throw std::invalid_argument("e8 causal attention: storage is not a fork KV mode");
    }
    CUDA_CHECK(cudaGetLastError());
    kv_cache_inverse_rotate_output_kernel<Geometry::QHeads>
        <<<tokens * Geometry::QHeads * kv::kInt8Groups, 32, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out.data), tokens, tokens, 0, nullptr);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Cache, typename Metadata>
void prompt_dispatch(const Tensor& q, const Tensor& positions, float scale, const Cache& cache,
                     Metadata metadata, Tensor& out, cudaStream_t stream) {
    if (q.ne[1] == e8::CausalD256H24Kv4::QHeads) {
        prompt_attention_for<e8::CausalD256H24Kv4>(q, positions, scale, cache, metadata, out,
                                                   stream);
        return;
    }
    prompt_attention_for<e8::CausalD256H16Kv2>(q, positions, scale, cache, metadata, out, stream);
}

void prompt_append(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   const Tensor& valid_columns, const Tensor& table_rows, float scale,
                   PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream) {
    kv::launch_fork_write(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const kv::BatchMetadata<Masked> metadata{
            static_cast<const std::int64_t*>(cache.k_pages.records.page_offsets.data),
            static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(table_rows.data),
            cache.k_pages.records.page_offsets.ne[0]};
        prompt_dispatch(q, positions, scale, cache, metadata, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

void prompt_cached(const Tensor& q, const Tensor& positions, float scale,
                   const ConstPagedKVLayerView& cache, Tensor& out, cudaStream_t stream) {
    const kv::DirectMetadata metadata{
        static_cast<const std::int64_t*>(cache.k_pages.records.page_offsets.data)};
    prompt_dispatch(q, positions, scale, cache, metadata, out, stream);
}

void require_fork_storage(KvCacheStorage storage, const char* op) {
    if (!kv_storage_is_fork_mode(storage)) {
        throw std::invalid_argument(std::string(op) + ": storage is not a fork KV mode");
    }
}

} // namespace

void e8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, const Tensor& valid, const Tensor& rows,
                            float scale, PagedKVBatchLayerView cache,
                            CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                            Tensor& out, CudaExecutionView execution) {
    require_fork_storage(cache.storage, "e8_kv_append_attention");
    const cudaStream_t stream = execution.stream();
    const std::int32_t width  = q.ne[2];
    const std::int32_t batch  = q.ne[3];
    auto scope                = workspace.scope();
    const E8KvRoute route     = e8_kv_resolve_route(q.ne[1], width, batch, envelope);
    if (route == E8KvRoute::ChunkedSmallT) {
        const std::int32_t per_call = e8_kv_chunk_tokens(q.ne[1], width, batch, envelope);
        for (std::int32_t begin = 0; begin < width; begin += per_call) {
            const std::int32_t count  = std::min(per_call, width - begin);
            auto chunk_scope          = workspace.scope();
            const std::int32_t splits = e8_kv_split_capacity(q.ne[1], count, envelope, batch);
            const auto partial = allocate_causal_partials(workspace, q.ne[1], count, splits, batch);
            small_t_append(q, k, v, positions, valid, rows, scale, cache, envelope, begin, count,
                           partial.view(), out, stream);
        }
        return;
    }
    if (route == E8KvRoute::SmallT) {
        const std::int32_t splits = e8_kv_split_capacity(q.ne[1], width, envelope, batch);
        const auto partial = allocate_causal_partials(workspace, q.ne[1], width, splits, batch);
        small_t_append(q, k, v, positions, valid, rows, scale, cache, envelope, 0, width,
                       partial.view(), out, stream);
        return;
    }
    prompt_append(q, k, v, positions, valid, rows, scale, cache, out, stream);
}

void e8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                            const ConstPagedKVLayerView& cache,
                            CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                            Tensor& out, CudaExecutionView execution) {
    require_fork_storage(cache.storage, "e8_kv_cached_attention");
    const cudaStream_t stream = execution.stream();
    const std::int32_t width  = q.ne[2];
    auto scope                = workspace.scope();
    const E8KvRoute route     = e8_kv_resolve_route(q.ne[1], width, 1, envelope);
    if (route == E8KvRoute::ChunkedSmallT) {
        const std::int32_t per_call = e8_kv_chunk_tokens(q.ne[1], width, 1, envelope);
        for (std::int32_t begin = 0; begin < width; begin += per_call) {
            const std::int32_t count  = std::min(per_call, width - begin);
            auto chunk_scope          = workspace.scope();
            const std::int32_t splits = e8_kv_split_capacity(q.ne[1], count, envelope);
            const auto partial  = allocate_causal_partials(workspace, q.ne[1], count, splits, 1);
            Tensor q_chunk        = q.slice(2, begin, count);
            Tensor position_chunk = positions.slice(0, begin, count);
            Tensor out_chunk      = out.slice(2, begin, count);
            small_t_cached(q_chunk, position_chunk, scale, cache, envelope, partial.view(),
                           out_chunk, stream);
        }
        return;
    }
    if (route == E8KvRoute::SmallT) {
        const std::int32_t splits = e8_kv_split_capacity(q.ne[1], width, envelope);
        const auto partial        = allocate_causal_partials(workspace, q.ne[1], width, splits, 1);
        small_t_cached(q, positions, scale, cache, envelope, partial.view(), out, stream);
        return;
    }
    prompt_cached(q, positions, scale, cache, out, stream);
}

} // namespace ninfer::ops::detail
