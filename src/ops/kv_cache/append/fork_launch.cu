// ninfer::ops::detail - fork KV storage append launch ownership (rtx4090-port).
#include "ops/kv_cache/append/launch.h"

#include "core/device.h"
#include "core/paged_kv_storage.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/fork_kernel.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

template <typename Geometry, bool PackedV, bool RotateK, bool RotateV, bool PackedK, bool E8Lattice,
          bool E8Root, typename CacheView, typename Metadata>
void launch_fork_mode(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                      Metadata metadata, cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    auto* cache_k     = static_cast<std::int8_t*>(cache.k_pages.data);
    auto* cache_v     = static_cast<std::uint8_t*>(cache.v_pages.data);
    auto* scale_k     = static_cast<__half*>(cache.k_scale_pages.data);
    auto* scale_v     = static_cast<__half*>(cache.v_scale_pages.data);
    if (tokens >= 32) {
        constexpr int TokensPerTile = 8;
        const int max_tiles = static_cast<int>(div_up(tokens + TokensPerTile, TokensPerTile));
        const dim3 grid(static_cast<unsigned>(max_tiles), static_cast<unsigned>(Geometry::KVHeads),
                        static_cast<unsigned>(kKVCacheInt8Groups));
        kv_cache_append_fork_page_kernel<Geometry, PackedV, RotateK, RotateV, PackedK, E8Lattice,
                                         E8Root, Metadata><<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), metadata, cache_k, cache_v, scale_k,
            scale_v, tokens);
    } else {
        constexpr int FillWarps = kBlock / 32;
        const std::int64_t fill_units =
            static_cast<std::int64_t>(tokens) * Geometry::KVHeads * kKVCacheInt8Groups;
        const int grid = static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(FillWarps)));
        kv_cache_append_fork_kernel<Geometry, PackedV, RotateK, RotateV, PackedK, E8Lattice, E8Root,
                                    Metadata><<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), metadata, cache_k, cache_v, scale_k,
            scale_v, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void launch_fork_for(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                     Metadata metadata, cudaStream_t stream) {
    const KvForkModeFlags mode = kv_fork_mode_flags(cache.storage);
    if (mode.e8_root) {
        launch_fork_mode<Geometry, true, true, true, false, false, true>(k, v, positions, cache,
                                                                         metadata, stream);
    } else if (mode.e8_lattice) {
        launch_fork_mode<Geometry, true, true, true, true, true, false>(k, v, positions, cache,
                                                                        metadata, stream);
    } else if (mode.packed_k) {
        launch_fork_mode<Geometry, true, true, true, true, false, false>(k, v, positions, cache,
                                                                         metadata, stream);
    } else {
        launch_fork_mode<Geometry, true, true, true, false, false, false>(k, v, positions, cache,
                                                                          metadata, stream);
    }
}

template <typename CacheView, typename Metadata>
void dispatch_fork(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                   Metadata metadata, cudaStream_t stream) {
    if (!kv_storage_is_fork_mode(cache.storage)) {
        throw std::invalid_argument("kv_cache_append: storage is not a fork KV mode");
    }
    if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads) {
        launch_fork_for<KVCacheAppendD256Kv4>(k, v, positions, cache, metadata, stream);
        return;
    }
    launch_fork_for<KVCacheAppendD256Kv2>(k, v, positions, cache, metadata, stream);
}

} // namespace

void kv_cache_append_fork_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                 PagedKVLayerView cache, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    dispatch_fork(k, v, positions, cache, metadata, stream);
}

void kv_cache_append_fork_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                       const Tensor& valid_columns, const Tensor& table_rows,
                                       PagedKVBatchLayerView cache, cudaStream_t stream) {
    // The e8 attention family appends B>1 rows inside its small-T kernels; only its B=1 prompt
    // route and the MTP layer use this standalone path, and both write one row.
    if (k.ne[3] != 1) {
        throw std::invalid_argument("kv_cache_append: fork KV modes append one batch row");
    }
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        dispatch_fork(k, v, positions, cache, metadata, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
