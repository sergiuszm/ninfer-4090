#pragma once

// Fork-local (rtx4090-port): causal cached attention over the fork KV storages
// (kv_storage_is_fork_mode). Same contract as the upstream per-storage families.

#include "ninfer/ops/softmax_attention.h"

namespace ninfer::ops::detail {

void e8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, const Tensor& valid, const Tensor& rows,
                            float scale, PagedKVBatchLayerView cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, DeviceExecutionView execution);

void e8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                            const PagedKVLayerView& cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, DeviceExecutionView execution);

} // namespace ninfer::ops::detail
