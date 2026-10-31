#pragma once

#include "core/dtype.h"
#include "ninfer/kv_cache_format.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer {

inline constexpr std::int32_t kD256KVCacheHeadDim = 256;

/** Physical data/scale planes for one K or V vector. */
struct PagedKVVectorLayout {
    DType data_dtype                  = DType::BF16;
    std::int32_t data_leading_extent  = 0;
    DType scale_dtype                 = DType::U8;
    std::int32_t scale_leading_extent = 0;

    [[nodiscard]] constexpr bool has_scale() const noexcept { return scale_leading_extent != 0; }

    [[nodiscard]] std::size_t physical_bytes() const {
        return static_cast<std::size_t>(data_leading_extent) * dtype_size(data_dtype) +
               static_cast<std::size_t>(scale_leading_extent) * dtype_size(scale_dtype);
    }

    friend bool operator==(const PagedKVVectorLayout&, const PagedKVVectorLayout&) = default;
};

/** Resolved physical plane schema for one paged K/V layer. */
struct PagedKVStorageLayout {
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    std::int32_t head_dim  = 0;
    PagedKVVectorLayout key;
    PagedKVVectorLayout value;

    [[nodiscard]] constexpr std::size_t planes_per_layer() const noexcept {
        return 2ULL + static_cast<std::size_t>(key.has_scale()) +
               static_cast<std::size_t>(value.has_scale());
    }

    [[nodiscard]] constexpr std::size_t logical_vector_bytes() const noexcept {
        return static_cast<std::size_t>(head_dim) * sizeof(std::uint16_t);
    }

    [[nodiscard]] constexpr std::size_t logical_bytes_per_token_head() const noexcept {
        return 2ULL * logical_vector_bytes();
    }

    [[nodiscard]] std::size_t physical_bytes_per_token_head() const {
        return key.physical_bytes() + value.physical_bytes();
    }

    friend bool operator==(const PagedKVStorageLayout&, const PagedKVStorageLayout&) = default;
};

[[nodiscard]] inline PagedKVVectorLayout paged_kv_vector_layout(KvCacheVectorFormat format,
                                                                std::int32_t dim) {
    const auto info = kv_cache_format_info(format);
    if (dim <= 0 || (info.group_size && dim % info.group_size))
        throw std::invalid_argument("unsupported KV-cache vector geometry");
    DType data;
    switch (format) {
    case KvCacheVectorFormat::BFloat16:
        data = DType::BF16;
        break;
    case KvCacheVectorFormat::Float16:
        data = DType::FP16;
        break;
    case KvCacheVectorFormat::Int8Group64F16:
        data = DType::I8;
        break;
    case KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0:
        data = DType::FP8_E4M3FN;
        break;
    case KvCacheVectorFormat::Nvfp4Group16Ue4m3:
    case KvCacheVectorFormat::Int4Group64F16:
    case KvCacheVectorFormat::E8LatticeGroup64F16:
    case KvCacheVectorFormat::E8RootGroup64F16:
        data = DType::U8;
        break;
    default:
        throw std::invalid_argument("unknown KV-cache vector format");
    }
    const std::int32_t data_extent = info.value_bits == 4   ? dim / 2
                                     : info.value_bits == 2 ? dim / 4
                                                            : dim;
    return {data, data_extent,
            info.scale == KvCacheScaleFormat::Float16 ? DType::FP16 : DType::U8,
            info.group_size ? dim / info.group_size : 0};
}

[[nodiscard]] inline PagedKVStorageLayout paged_kv_storage_layout(KvCacheStorage storage,
                                                                  std::int32_t head_dim) {
    const auto profile = kv_cache_profile_info(storage);
    if (head_dim <= 0 || (storage != KvCacheStorage::BFloat16 && head_dim != kD256KVCacheHeadDim))
        throw std::invalid_argument("unsupported paged KV-cache storage geometry");
    return {storage, head_dim, paged_kv_vector_layout(profile.key, head_dim),
            paged_kv_vector_layout(profile.value, head_dim)};
}

/** Fork-local (rtx4090-port): how a fork storage packs its planes. Kernels dispatch on these
 *  flags; the storage enum stays the single source of truth for them. */
struct KvForkModeFlags {
    bool packed_v   = false;
    bool rotate_k   = false;
    bool rotate_v   = false;
    bool packed_k   = false;
    bool e8_lattice = false;
    bool e8_root    = false;
};

/** Fork-local: the storages served by the fork append path and the causal_cache/e8/ family.
 *  Upstream Int8Group64 (d256-rotated keys) is not one of them. Ops route on the storage, never
 *  on a plane's vector format: rk8v4 keys share the Int8Group64F16 layout but are not
 *  d256-rotated. */
[[nodiscard]] constexpr bool kv_storage_is_fork_mode(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
    case KvCacheStorage::RK4V4E8:
    case KvCacheStorage::RK2V4E8:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] constexpr KvForkModeFlags kv_fork_mode_flags(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return {.packed_v = true, .rotate_k = true, .rotate_v = true};
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
        return {.packed_v = true, .rotate_k = true, .rotate_v = true, .packed_k = true};
    case KvCacheStorage::RK4V4E8:
        return {.packed_v   = true,
                .rotate_k   = true,
                .rotate_v   = true,
                .packed_k   = true,
                .e8_lattice = true};
    case KvCacheStorage::RK2V4E8:
        return {.packed_v = true, .rotate_k = true, .rotate_v = true, .e8_root = true};
    default:
        return {};
    }
}

} // namespace ninfer
