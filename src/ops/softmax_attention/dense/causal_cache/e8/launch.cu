// Fork-local (rtx4090-port): causal cached attention over the fork KV storages. The small-T
// kernels append the new keys and values themselves; the prompt route appends through
// kv_cache_append_batch_launch and then reads the whole history from the cache. Launch code from
// rtx4090-port small_t.cu, prompt.cu and causal_softmax_attention.cpp, int8 family only.
#include "ops/softmax_attention/dense/causal_cache/e8/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "core/paged_kv_storage.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/e8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/e8/prompt_i8.cuh"
#include "ops/softmax_attention/dense/causal_cache/e8/small_t_launch.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

using e8_launch::SmallTInvocation;
using e8_launch::small_t_launch_for;

template <typename CacheInput>
void small_t_dispatch(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                      PagedKVBatchLayerView cache, const SmallTInvocation& invocation,
                      CausalAttentionExecutionEnvelope envelope, E8SmallTPartials& partial,
                      Tensor& out, cudaStream_t stream) {
    if (q.ne[1] == e8::CausalD256H24Kv4::QHeads) {
        small_t_launch_for<e8::CausalD256H24Kv4>(q, input, pos, scale, cache, invocation, envelope,
                                                 partial, out, stream);
        return;
    }
    small_t_launch_for<e8::CausalD256H16Kv2>(q, input, pos, scale, cache, invocation, envelope,
                                             partial, out, stream);
}

void small_t_append(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                    const Tensor& valid_columns, const Tensor& table_rows, float scale,
                    PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope,
                    std::int32_t column_begin, std::int32_t width, E8SmallTPartials& partial,
                    Tensor& out, cudaStream_t stream) {
    const e8::CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                      static_cast<const __nv_bfloat16*>(v.data)};
    const SmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
    };
    small_t_dispatch(q, input, pos, scale, cache, invocation, envelope, partial, out, stream);
}

void small_t_cached(const Tensor& q, const Tensor& pos, float scale, const PagedKVLayerView& cache,
                    CausalAttentionExecutionEnvelope envelope, E8SmallTPartials& partial,
                    Tensor& out, cudaStream_t stream) {
    const SmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
    };
    const PagedKVBatchLayerView batch_cache = single_row_paged_kv_batch_view(cache);
    small_t_dispatch(q, e8::CausalCachedInput{}, pos, scale, batch_cache, invocation, envelope,
                     partial, out, stream);
}

template <typename Geometry, typename CacheView, typename Metadata>
void prompt_attention_for(const Tensor& q, const Tensor& positions, float scale,
                          const CacheView& cache, Metadata metadata, Tensor& out,
                          cudaStream_t stream) {
    const auto tokens          = static_cast<std::int32_t>(q.ne[2]);
    const KvForkModeFlags mode = kv_fork_mode_flags(cache.storage);
    const dim3 grid(static_cast<unsigned>(div_up(tokens, e8::kCausalPromptI8Br)),
                    static_cast<unsigned>(Geometry::QHeads), 1U);
    const auto launch = [&]<bool PackedK, bool E8Root>() {
        const auto kernel = e8::causal_attention_prompt_i8_kernel<Geometry, true, true, true,
                                                                  PackedK, E8Root, Metadata>;
        static const cudaError_t attr = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, e8::kCausalPromptI8SmemBytes);
        CUDA_CHECK(attr);
        kernel<<<grid, e8::kCausalPromptI8Threads, e8::kCausalPromptI8SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int8_t*>(cache.k_pages.data),
            static_cast<const std::uint8_t*>(cache.v_pages.data),
            static_cast<const __half*>(cache.k_scale_pages.data),
            static_cast<const __half*>(cache.v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
    };
    if (mode.e8_root) {
        launch.template operator()<false, true>();
    } else if (mode.packed_k) {
        launch.template operator()<true, false>();
    } else {
        launch.template operator()<false, false>();
    }
    CUDA_CHECK(cudaGetLastError());
    if (mode.rotate_v) {
        kv_cache_inverse_rotate_output_kernel<Geometry::QHeads>
            <<<tokens * Geometry::QHeads * kKVCacheInt8Groups, 32, 0, stream>>>(
                static_cast<__nv_bfloat16*>(out.data), tokens, tokens, 0, nullptr);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <typename CacheView, typename Metadata>
void prompt_dispatch(const Tensor& q, const Tensor& positions, float scale, const CacheView& cache,
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
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        prompt_dispatch(q, positions, scale, cache, metadata, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

void prompt_cached(const Tensor& q, const Tensor& positions, float scale,
                   const PagedKVLayerView& cache, Tensor& out, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
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
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, DeviceExecutionView execution) {
    require_fork_storage(cache.storage, "e8_kv_append_attention");
    const cudaStream_t stream = execution.stream;
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
            E8SmallTPartials partial =
                allocate_e8_small_t_partials(workspace, q.ne[1], count, splits, batch);
            small_t_append(q, k, v, positions, valid, rows, scale, cache, envelope, begin, count,
                           partial, out, stream);
        }
        return;
    }
    if (route == E8KvRoute::SmallT) {
        const std::int32_t splits = e8_kv_split_capacity(q.ne[1], width, envelope, batch);
        E8SmallTPartials partial =
            allocate_e8_small_t_partials(workspace, q.ne[1], width, splits, batch);
        small_t_append(q, k, v, positions, valid, rows, scale, cache, envelope, 0, width, partial,
                       out, stream);
        return;
    }
    prompt_append(q, k, v, positions, valid, rows, scale, cache, out, stream);
}

void e8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                            const PagedKVLayerView& cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, DeviceExecutionView execution) {
    require_fork_storage(cache.storage, "e8_kv_cached_attention");
    const cudaStream_t stream = execution.stream;
    const std::int32_t width  = q.ne[2];
    auto scope                = workspace.scope();
    const E8KvRoute route     = e8_kv_resolve_route(q.ne[1], width, 1, envelope);
    if (route == E8KvRoute::ChunkedSmallT) {
        const std::int32_t per_call = e8_kv_chunk_tokens(q.ne[1], width, 1, envelope);
        for (std::int32_t begin = 0; begin < width; begin += per_call) {
            const std::int32_t count  = std::min(per_call, width - begin);
            auto chunk_scope          = workspace.scope();
            const std::int32_t splits = e8_kv_split_capacity(q.ne[1], count, envelope);
            E8SmallTPartials partial =
                allocate_e8_small_t_partials(workspace, q.ne[1], count, splits, 1);
            Tensor q_chunk        = q.slice(2, begin, count);
            Tensor position_chunk = positions.slice(0, begin, count);
            Tensor out_chunk      = out.slice(2, begin, count);
            small_t_cached(q_chunk, position_chunk, scale, cache, envelope, partial, out_chunk,
                           stream);
        }
        return;
    }
    if (route == E8KvRoute::SmallT) {
        const std::int32_t splits = e8_kv_split_capacity(q.ne[1], width, envelope);
        E8SmallTPartials partial =
            allocate_e8_small_t_partials(workspace, q.ne[1], width, splits, 1);
        small_t_cached(q, positions, scale, cache, envelope, partial, out, stream);
        return;
    }
    prompt_cached(q, positions, scale, cache, out, stream);
}

} // namespace ninfer::ops::detail
