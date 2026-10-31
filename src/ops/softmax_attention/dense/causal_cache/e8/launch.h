#pragma once

// Fork-local (rtx4090-port): causal cached attention over the fork KV storages
// (kv_storage_is_fork_mode). Same contract as the upstream per-storage families; any other storage
// is refused.

#include "ninfer/ops/softmax_attention.h"

namespace ninfer::ops::detail {

void e8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, const Tensor& valid, const Tensor& rows,
                            float scale, PagedKVBatchLayerView cache,
                            CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                            Tensor& out, CudaExecutionView execution);

void e8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                            const ConstPagedKVLayerView& cache,
                            CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                            Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops::detail
