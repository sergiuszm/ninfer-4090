#pragma once

#include "kv_cache_storage.h"
#include "ops/float_codec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::test {

// Fork-local (rtx4090-port): host restatement of the fork KV codecs of ninfer/kv_cache_format.h.
// Each 64-feature group is rotated by the normalized Sylvester H64 (entry (-1)^popcount(i & j) / 8)
// and scaled by FP16-RNE(absmax / levels). INT8 keys (rk8v4) take RNE codes clamped to
// [-127, 127]; INT4 vectors take RNE codes clamped to [-7, 7]; E8 lattice keys (rk4v4-e8) move each
// eight-feature block to its nearest E8 point, then round (RNE) and clamp to [-8, 7]. No coset bit
// is stored, so a half-integral E8 point is collapsed to integers. Byte j of an INT4 group holds
// feature 2j in its low nibble and 2j + 1 in its high nibble. The encoder repeats the device's
// FP32 operation order (butterfly rotation, pairwise distance sums), so cache bytes compare
// exactly. E8 root keys (rk2v4-e8) have no host restatement.
// From roofkid/ninfer-4080 5f377f22.

inline constexpr int kForkGroup   = 64;
inline constexpr int kForkGroups  = 4;
inline constexpr int kForkHeadDim = kForkGroup * kForkGroups;

enum class ForkVectorCodec { Int8, Int4, E8Lattice };

// The device's warp butterfly: lane l carries features l and l + 32.
inline void fork_rotate_h64(std::array<float, kForkGroup>& x) {
    constexpr int kLanes = kForkGroup / 2;
    for (int offset = 1; offset < kLanes; offset <<= 1) {
        const std::array<float, kForkGroup> before = x;
        for (int lane = 0; lane < kLanes; ++lane) {
            const bool high = (lane & offset) != 0;
            for (const int half : {0, kLanes}) {
                const float own   = before[static_cast<std::size_t>(half + lane)];
                const float other = before[static_cast<std::size_t>(half + (lane ^ offset))];
                x[static_cast<std::size_t>(half + lane)] = high ? other - own : own + other;
            }
        }
    }
    for (int lane = 0; lane < kLanes; ++lane) {
        const float a                              = x[static_cast<std::size_t>(lane)];
        const float b                              = x[static_cast<std::size_t>(lane + kLanes)];
        x[static_cast<std::size_t>(lane)]          = (a + b) * 0.125f;
        x[static_cast<std::size_t>(lane + kLanes)] = (a - b) * 0.125f;
    }
}

inline float fork_pairwise_sum8(const std::array<float, 8>& p) {
    return ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]));
}

// Nearest point of E8 = D8 U (D8 + 1/2): the nearest D8 point and the nearest D8 + 1/2 point
// (each fixing odd parity at the worst-rounded coordinate), keeping D8 on a distance tie.
inline std::array<float, 8> fork_nearest_e8(const std::array<float, 8>& x) {
    const auto coset_point = [&](float shift) {
        std::array<float, 8> point{};
        int sum          = 0;
        float worst_err  = -1.0f;
        int worst        = 0;
        float worst_diff = 0.0f;
        for (int i = 0; i < 8; ++i) {
            const float shifted = x[static_cast<std::size_t>(i)] - shift;
            const float rounded = std::rint(shifted);
            sum += static_cast<int>(rounded);
            const float err = std::abs(shifted - rounded);
            if (err > worst_err) {
                worst_err  = err;
                worst      = i;
                worst_diff = shifted - rounded;
            }
            point[static_cast<std::size_t>(i)] = rounded + shift;
        }
        if ((sum & 1) != 0)
            point[static_cast<std::size_t>(worst)] += worst_diff >= 0.0f ? 1.0f : -1.0f;
        return point;
    };
    const std::array<float, 8> d8   = coset_point(0.0f);
    const std::array<float, 8> half = coset_point(0.5f);
    std::array<float, 8> d8_error{}, half_error{};
    for (int i = 0; i < 8; ++i) {
        const float e0 = x[static_cast<std::size_t>(i)] - d8[static_cast<std::size_t>(i)];
        const float e1 = x[static_cast<std::size_t>(i)] - half[static_cast<std::size_t>(i)];
        d8_error[static_cast<std::size_t>(i)]   = e0 * e0;
        half_error[static_cast<std::size_t>(i)] = e1 * e1;
    }
    return fork_pairwise_sum8(d8_error) <= fork_pairwise_sum8(half_error) ? d8 : half;
}

// Encodes one 64-feature group from public BF16-represented FP32 values. INT8 writes 64 code bytes,
// INT4 and E8 lattice write 32 packed bytes. Returns the FP16 scale bits.
inline std::uint16_t fork_encode_group(const float* input, ForkVectorCodec codec,
                                       std::uint8_t* codes) {
    using namespace float_codec;
    std::array<float, kForkGroup> x{};
    std::copy_n(input, kForkGroup, x.begin());
    fork_rotate_h64(x);
    float absmax = 0.0f;
    for (const float value : x) absmax = std::max(absmax, std::abs(value));
    const float levels             = codec == ForkVectorCodec::Int8 ? 127.0f : 7.0f;
    const std::uint16_t scale_bits = f32_to_f16_bits(absmax > 0.0f ? absmax / levels : 0.0f);
    const float stored_scale       = f16_bits_to_f32(scale_bits);
    const float inverse_scale      = stored_scale > 0.0f ? 1.0f / stored_scale : 0.0f;

    std::array<std::int32_t, kForkGroup> code{};
    if (codec == ForkVectorCodec::E8Lattice) {
        for (int block = 0; block < kForkGroup / 8; ++block) {
            std::array<float, 8> scaled{};
            for (int i = 0; i < 8; ++i)
                scaled[static_cast<std::size_t>(i)] =
                    x[static_cast<std::size_t>(block * 8 + i)] * inverse_scale;
            const std::array<float, 8> point = fork_nearest_e8(scaled);
            for (int i = 0; i < 8; ++i)
                code[static_cast<std::size_t>(block * 8 + i)] = std::clamp(
                    static_cast<std::int32_t>(std::rint(point[static_cast<std::size_t>(i)])), -8,
                    7);
        }
    } else if (inverse_scale != 0.0f) {
        const int limit = codec == ForkVectorCodec::Int8 ? 127 : 7;
        for (int i = 0; i < kForkGroup; ++i)
            code[static_cast<std::size_t>(i)] = std::clamp(
                round_even_to_i32(x[static_cast<std::size_t>(i)] * inverse_scale), -limit, limit);
    }
    if (codec == ForkVectorCodec::Int8) {
        for (int i = 0; i < kForkGroup; ++i)
            codes[i] = static_cast<std::uint8_t>(code[static_cast<std::size_t>(i)]);
    } else {
        for (int pair = 0; pair < kForkGroup / 2; ++pair) {
            const auto low  = static_cast<unsigned>(code[static_cast<std::size_t>(2 * pair)]);
            const auto high = static_cast<unsigned>(code[static_cast<std::size_t>(2 * pair + 1)]);
            codes[pair]     = static_cast<std::uint8_t>((low & 0x0fu) | ((high & 0x0fu) << 4));
        }
    }
    return scale_bits;
}

// The vector codec a fork storage applies to K or V. rk2v4-e8 keys (E8 root) are not restated.
inline ForkVectorCodec fork_vector_codec(KvCacheStorage storage, bool key) {
    if (!key) return ForkVectorCodec::Int4;
    switch (storage) {
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return ForkVectorCodec::Int8;
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
        return ForkVectorCodec::Int4;
    case KvCacheStorage::RK4V4E8:
        return ForkVectorCodec::E8Lattice;
    default:
        throw std::invalid_argument("no host restatement of this fork KV key codec");
    }
}

// Code bytes of one 64-feature group.
inline int fork_group_code_bytes(ForkVectorCodec codec) {
    return codec == ForkVectorCodec::Int8 ? kForkGroup : kForkGroup / 2;
}

// Encodes one D256 row: codes at codes[0, 4 * group bytes), FP16 scale bits at scales[0, 4).
inline void fork_encode_row(const float* input, ForkVectorCodec codec, std::uint8_t* codes,
                            std::uint16_t* scales) {
    const int bytes = fork_group_code_bytes(codec);
    for (int group = 0; group < kForkGroups; ++group)
        scales[group] = fork_encode_group(input + group * kForkGroup, codec, codes + group * bytes);
}

} // namespace ninfer::test
