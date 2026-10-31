#include "ops/kv_cache/kernels/fork_write.h"
#include "core/device.h"
#include "core/paged_kv_storage.h"
#include "ops/common/numeric/math.h"
#include "ops/kv_cache/kernels/write/fork.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::kv {
namespace {

constexpr int kThreads = 256;
// Single-row writes of at least this many tokens run in page-local tiles; batched rows and short
// writes run one warp per (token, kv_head, group).
constexpr int kPageTileMinimumTokens = 32;
constexpr int kTokensPerTile         = 8;

struct ForkWriteArguments {
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const std::int32_t* positions;
    std::uint8_t* keys;
    std::uint8_t* values;
    __half* key_scales;
    __half* value_scales;
    int width;
    int batch;
};

template <class Cache>
ForkWriteArguments arguments(const Tensor& k, const Tensor& v, const Tensor& positions,
                             const Cache& cache) {
    return {static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<std::uint8_t*>(cache.k_pages.data()),
            static_cast<std::uint8_t*>(cache.v_pages.data()),
            static_cast<__half*>(cache.k_scale_pages.data()),
            static_cast<__half*>(cache.v_scale_pages.data()),
            k.ne[2],
            k.ne[3]};
}

template <class G, KvCacheStorage Storage, class Metadata>
void launch_storage(const ForkWriteArguments& a, Metadata metadata, cudaStream_t stream) {
    if (a.batch == 1 && a.width >= kPageTileMinimumTokens) {
        const dim3 grid(common::div_up(a.width + kTokensPerTile, kTokensPerTile), G::KVHeads,
                        kInt8Groups);
        write_fork_page_kernel<G, Storage, Metadata><<<grid, kThreads, 0, stream>>>(
            a.k, a.v, a.positions, metadata, a.keys, a.values, a.key_scales, a.value_scales,
            a.width);
        return;
    }
    constexpr bool MultiBatch = requires { Metadata::masked; };
    const auto units          = static_cast<std::int64_t>(a.width) * G::KVHeads * kInt8Groups;
    const dim3 grid(common::div_up(units, static_cast<std::int64_t>(kThreads / 32)), 1, a.batch);
    write_fork_kernel<G, Storage, Metadata, MultiBatch><<<grid, kThreads, 0, stream>>>(
        a.k, a.v, a.positions, metadata, a.keys, a.values, a.key_scales, a.value_scales, a.width);
}

template <class G, class Metadata>
void storage_dispatch(KvCacheStorage storage, const ForkWriteArguments& a, Metadata metadata,
                      cudaStream_t stream) {
    switch (storage) {
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return launch_storage<G, KvCacheStorage::RotatedInt8KeyInt4ValueGroup64>(a, metadata,
                                                                                  stream);
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
        return launch_storage<G, KvCacheStorage::RotatedInt4KeyInt4ValueGroup64>(a, metadata,
                                                                                  stream);
    case KvCacheStorage::RK4V4E8:
        return launch_storage<G, KvCacheStorage::RK4V4E8>(a, metadata, stream);
    case KvCacheStorage::RK2V4E8:
        return launch_storage<G, KvCacheStorage::RK2V4E8>(a, metadata, stream);
    default:
        break;
    }
    throw std::logic_error("KV fork write: storage is not a fork KV mode");
}

template <class Metadata>
void geometry_dispatch(int heads, KvCacheStorage storage, const ForkWriteArguments& a,
                       Metadata metadata, cudaStream_t stream) {
    if (heads == 4)
        storage_dispatch<D256Geometry<4>>(storage, a, metadata, stream);
    else
        storage_dispatch<D256Geometry<2>>(storage, a, metadata, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_fork_write(const Tensor& k, const Tensor& v, const Tensor& positions,
                       PagedKVLayerView cache, cudaStream_t stream) {
    geometry_dispatch(
        k.ne[1], cache.storage, arguments(k, v, positions, cache),
        DirectMetadata{static_cast<const std::int64_t*>(cache.k_pages.records.page_offsets.data)},
        stream);
}

void launch_fork_write(const Tensor& k, const Tensor& v, const Tensor& positions,
                       const Tensor& valid_columns, const Tensor& table_rows,
                       PagedKVBatchLayerView cache, cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const BatchMetadata<Masked> metadata{
            static_cast<const std::int64_t*>(cache.k_pages.records.page_offsets.data),
            static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(table_rows.data),
            cache.k_pages.records.page_offsets.ne[0]};
        geometry_dispatch(k.ne[1], cache.storage, arguments(k, v, positions, cache), metadata,
                          stream);
    };
    if (valid_columns.data)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}

} // namespace ninfer::ops::detail::kv
