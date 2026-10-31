#pragma once

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string_view>

namespace ninfer {

// A runtime cache vector's representation. Basis transforms and computation precision
// belong to the consuming Op. Page/plane geometry belongs to the cache layout.
enum class KvCacheVectorFormat : std::uint8_t {
    BFloat16,
    Float16,
    Int8Group64F16,
    Fp8E4m3fnGroup128Ue8m0,
    Nvfp4Group16Ue4m3,
    // Fork-local (rtx4090-port) formats, numbered from 0x80 so upstream formats keep their values.
    Int4Group64F16 = 0x80,
    E8LatticeGroup64F16,
    E8RootGroup64F16,
};

enum class KvCacheScaleFormat : std::uint8_t { None, Float16, Ue8m0, Ue4m3 };

struct KvCacheFormatInfo {
    int value_bits;
    int group_size; // Zero for direct values; groups run along contiguous features.
    KvCacheScaleFormat scale;

    [[nodiscard]] constexpr int scale_bytes() const noexcept {
        return scale == KvCacheScaleFormat::None ? 0 : scale == KvCacheScaleFormat::Float16 ? 2 : 1;
    }
};

// Quantized formats use multipliers: decoded = code_value * stored_scale.
// INT8: a=max(abs(x)); s=FP16_RNE(a/127); codes=clamp(RNE(x/s),-127,127),
// with zero codes when the represented s is zero.
// FP8: a=max(abs(x)); e=max(-127,ceil(log2(a/448))); s=2^e; scale=e+127;
// codes=E4M3FN_RNE_SATFINITE(x/s). A zero group stores scale=127 and zero codes.
// UE8M0 byte b in [0,254] represents 2^(b-127); 255 is not emitted.
// NVFP4: s=UE4M3_RNE_SATFINITE(clamp(a/6,2^-9,448)); codes=E2M1_RNE_SATFINITE(x/s).
// A zero group stores zero scale/codes. The lower feature occupies the low nibble;
// there is no matrix-level divisor. Writers consume finite FP32 codec inputs.
// Fork-local (rtx4090-port) formats, all G64 with FP16 scales:
// INT4: s=FP16_RNE(a/7); codes=clamp(RNE(x/s),-7,7), zero codes when s is zero. The lower
// feature occupies the low nibble (signed two's-complement nibbles).
// E8 lattice: s=FP16_RNE(a/7); each eight-feature block of x/s moves to its nearest E8 point,
// whose coordinates are rounded (RNE) and clamped to [-8,7] (no coset bit is stored) and packed
// as INT4.
// E8 root: s=FP16_RNE(a/7); each eight-feature block of x/s stores one E8 root index byte and one
// radius/axis byte (2 bits per feature).
[[nodiscard]] constexpr KvCacheFormatInfo kv_cache_format_info(KvCacheVectorFormat format) {
    switch (format) {
    case KvCacheVectorFormat::BFloat16:
        return {16, 0, KvCacheScaleFormat::None};
    case KvCacheVectorFormat::Float16:
        return {16, 0, KvCacheScaleFormat::None};
    case KvCacheVectorFormat::Int8Group64F16:
        return {8, 64, KvCacheScaleFormat::Float16};
    case KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0:
        return {8, 128, KvCacheScaleFormat::Ue8m0};
    case KvCacheVectorFormat::Nvfp4Group16Ue4m3:
        return {4, 16, KvCacheScaleFormat::Ue4m3};
    case KvCacheVectorFormat::Int4Group64F16:
    case KvCacheVectorFormat::E8LatticeGroup64F16:
        return {4, 64, KvCacheScaleFormat::Float16};
    case KvCacheVectorFormat::E8RootGroup64F16:
        return {2, 64, KvCacheScaleFormat::Float16};
    }
    throw std::invalid_argument("unknown KV-cache vector format");
}

// K/V profiles are distinct from vector format IDs, including on the internal wire.
enum class KvCacheStorage : std::uint8_t {
    BFloat16               = 0,
    Int8Group64            = 1,
    Fp8E4m3fnGroup128Ue8m0 = 2,
    Nvfp4Group16           = 3,
    Fp8KeyNvfp4Value       = 4,
    // Fork-local (rtx4090-port) profiles. Both vectors are rotated per 64-feature group (H64) by
    // the consuming Ops; upstream Int8Group64 keeps its D256-rotated keys. Numbered from 0x80 so a
    // future upstream profile cannot take the same wire value.
    RotatedInt8KeyInt4ValueGroup64 = 0x80,
    RotatedInt4KeyInt4ValueGroup64 = 0x81,
    RK4V4E8                        = 0x82,
    RK2V4E8                        = 0x83,
};

struct KvCacheProfileInfo {
    KvCacheVectorFormat key;
    KvCacheVectorFormat value;
    const char* name;
};

[[nodiscard]] constexpr KvCacheProfileInfo kv_cache_profile_info(KvCacheStorage storage) {
    using F = KvCacheVectorFormat;
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return {F::BFloat16, F::Float16, "bf16"};
    case KvCacheStorage::Int8Group64:
        return {F::Int8Group64F16, F::Int8Group64F16, "int8"};
    case KvCacheStorage::Fp8E4m3fnGroup128Ue8m0:
        return {F::Fp8E4m3fnGroup128Ue8m0, F::Fp8E4m3fnGroup128Ue8m0, "fp8"};
    case KvCacheStorage::Nvfp4Group16:
        return {F::Nvfp4Group16Ue4m3, F::Nvfp4Group16Ue4m3, "nvfp4"};
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return {F::Fp8E4m3fnGroup128Ue8m0, F::Nvfp4Group16Ue4m3, "k8v4"};
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return {F::Int8Group64F16, F::Int4Group64F16, "rk8v4"};
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
        return {F::Int4Group64F16, F::Int4Group64F16, "rk4v4"};
    case KvCacheStorage::RK4V4E8:
        return {F::E8LatticeGroup64F16, F::Int4Group64F16, "rk4v4-e8"};
    case KvCacheStorage::RK2V4E8:
        return {F::E8RootGroup64F16, F::Int4Group64F16, "rk2v4-e8"};
    }
    throw std::invalid_argument("unknown KV-cache storage profile");
}

[[nodiscard]] constexpr KvCacheStorage kv_cache_storage_from_option(std::string_view option) {
    for (auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                         KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, KvCacheStorage::Nvfp4Group16,
                         KvCacheStorage::Fp8KeyNvfp4Value,
                         KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,
                         KvCacheStorage::RotatedInt4KeyInt4ValueGroup64, KvCacheStorage::RK4V4E8,
                         KvCacheStorage::RK2V4E8})
        if (option == kv_cache_profile_info(storage).name) return storage;
    throw std::invalid_argument("unknown KV-cache storage option");
}

} // namespace ninfer
