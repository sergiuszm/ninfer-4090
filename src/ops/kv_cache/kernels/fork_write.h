#pragma once

#include <cuda_runtime_api.h>

#include "core/paged_kv_cache.h"

namespace ninfer::ops::detail::kv {

// Fork-local (rtx4090-port): D256 packed-record writes for the fork KV storages
// (kv_storage_is_fork_mode) from validated BF16 operands. Both vectors are rotated per 64-feature
// group (H64) and encoded with the fork codecs of ninfer/kv_cache_format.h; upstream Int8Group64
// writes stay in write.h. The batch form accepts B 1..8 with device-selected valid prefixes
// (valid_columns may be null for full rows); a row with no valid tokens reads no positions, table
// rows or K/V. Grids depend on the physical T and B only, so a captured launch stays valid.
void launch_fork_write(const Tensor& k, const Tensor& v, const Tensor& positions,
                       PagedKVLayerView cache, cudaStream_t stream);
void launch_fork_write(const Tensor& k, const Tensor& v, const Tensor& positions,
                       const Tensor& valid_columns, const Tensor& table_rows,
                       PagedKVBatchLayerView cache, cudaStream_t stream);

} // namespace ninfer::ops::detail::kv
