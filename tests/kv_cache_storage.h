#pragma once

#include "ninfer/types.h"
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::test {

inline KvCacheStorage parse_kv_cache_storage(std::string_view name) {
    if (name == "bf16") return KvCacheStorage::BFloat16;
    if (name == "int8") return KvCacheStorage::Int8Group64;
    if (name == "fp8") return KvCacheStorage::Fp8E4M3Row256;
    if (name == "nvfp4") return KvCacheStorage::Nvfp4Group16;
    if (name == "k8v4") return KvCacheStorage::Fp8KeyNvfp4Value;
    if (name == "rk8v4") return KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
    if (name == "rk4v4") return KvCacheStorage::RotatedInt4KeyInt4ValueGroup64;
    if (name == "rk4v4-e8") return KvCacheStorage::RK4V4E8;
    if (name == "rk2v4-e8") return KvCacheStorage::RK2V4E8;
    throw std::invalid_argument("unknown KV dtype: " + std::string(name));
}

} // namespace ninfer::test
