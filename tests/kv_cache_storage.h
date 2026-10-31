#pragma once

#include "ninfer/types.h"
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::test {

inline const char* kv_cache_storage_name(KvCacheStorage storage) {
    return kv_cache_profile_info(storage).name;
}

inline KvCacheStorage parse_kv_cache_storage(std::string_view name) {
    return kv_cache_storage_from_option(name);
}

// Fork-local (rtx4090-port): sm_89 startup refuses NVFP4 and K8V4 KV (E2M1 pack, TMA,
// setmaxnreg) and FP8 KV (block-scaled FP8 MMA), so their op cases are not registered there.
inline bool kv_cache_storage_runs_on_target(KvCacheStorage storage) {
#if defined(NINFER_SM86)
    return storage != KvCacheStorage::Nvfp4Group16 &&
           storage != KvCacheStorage::Fp8KeyNvfp4Value &&
           storage != KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
#else
    (void)storage;
    return true;
#endif
}

} // namespace ninfer::test
