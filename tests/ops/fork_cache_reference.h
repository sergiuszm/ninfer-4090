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
// exactly. E8 root keys (rk2v4-e8) are described at fork_e8_root_encode.
// From roofkid/ninfer-4080 5f377f22.

inline constexpr int kForkGroup   = 64;
inline constexpr int kForkGroups  = 4;
inline constexpr int kForkHeadDim = kForkGroup * kForkGroups;

enum class ForkVectorCodec { Int8, Int4, E8Lattice, E8Root };

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

// One H64-rotated group and its FP16 scale FP16-RNE(absmax / levels).
struct ForkRotatedGroup {
    std::array<float, kForkGroup> x{};
    std::uint16_t scale_bits = 0;
    float scale              = 0.0f;
    float inverse_scale      = 0.0f;
};

inline ForkRotatedGroup fork_rotate_group(const float* input, ForkVectorCodec codec) {
    using namespace float_codec;
    ForkRotatedGroup group;
    std::copy_n(input, kForkGroup, group.x.begin());
    fork_rotate_h64(group.x);
    float absmax = 0.0f;
    for (const float value : group.x) absmax = std::max(absmax, std::abs(value));
    const float levels  = codec == ForkVectorCodec::Int8 ? 127.0f : 7.0f;
    group.scale_bits    = f32_to_f16_bits(absmax > 0.0f ? absmax / levels : 0.0f);
    group.scale         = f16_bits_to_f32(group.scale_bits);
    group.inverse_scale = group.scale > 0.0f ? 1.0f / group.scale : 0.0f;
    return group;
}

// E8 root keys (rk2v4-e8), restated from ops/kv_cache/kernels/e8_root_codec.cuh. Block b of a
// rotated group (features 8b..8b+7) takes bytes 2b and 2b + 1 of the group's 16 code bytes; the
// group scale is the INT4 one (absmax / 7). The first byte selects one of the 240 E8 roots r
// (norm sqrt(2)): codes 0..111 are (+-1, +-1) on pair code / 4 of kForkE8RootPairs (bit 1 sets
// the first sign positive, bit 0 the second); codes 112..239 are (+-1/2)^8, bit i < 7 sets entry
// i positive and entry 7 completes an even count of negative entries. The second byte holds the
// radius index (high nibble, 0 = zero block) and the signed axis e (low nibble: 2 * dimension +
// negative). The device decodes entry i to the INT8 code rne((4 r_i + e_i) * radius scale) and
// multiplies it by the group scale.
inline constexpr std::array<std::array<int, 2>, 28> kForkE8RootPairs = [] {
    std::array<std::array<int, 2>, 28> pairs{};
    std::size_t next = 0;
    for (int i = 0; i < 8; ++i)
        for (int j = i + 1; j < 8; ++j) pairs[next++] = {i, j};
    return pairs;
}();

inline constexpr std::array<float, 16> kForkE8RootRadiusScale{
    0.0000f, 0.0992f, 0.1250f, 0.1575f, 0.1984f, 0.2500f, 0.3150f, 0.3969f,
    0.5000f, 0.6300f, 0.7937f, 1.0000f, 1.2599f, 1.5874f, 2.0000f, 2.5198f};

// A decision of the device codec that lies this close to an alternative may differ from this
// restatement, which repeats the FP32 formulas but not their exact operation order.
inline constexpr float kForkE8RootTieMargin = 1.0e-4f;

struct ForkE8RootCode {
    std::uint8_t root     = 0;
    std::uint8_t rad_axis = 0;
};

// 4 r for root code `code`. Codes 240..255 are not roots and decode to zero, as on the device.
inline std::array<int, 8> fork_e8_root4(int code) {
    std::array<int, 8> root{};
    if (code < 112) {
        const auto& pair = kForkE8RootPairs[static_cast<std::size_t>(code >> 2)];
        root[static_cast<std::size_t>(pair[0])] = (code & 2) ? 4 : -4;
        root[static_cast<std::size_t>(pair[1])] = (code & 1) ? 4 : -4;
    } else if (code < 240) {
        int negative = 0;
        for (std::size_t i = 0; i < 7; ++i) {
            const bool positive = (((code - 112) >> i) & 1) != 0;
            root[i]             = positive ? 2 : -2;
            negative += positive ? 0 : 1;
        }
        root[7] = (negative & 1) ? -2 : 2;
    }
    return root;
}

inline std::array<std::int8_t, 8> fork_e8_root_decode(ForkE8RootCode code) {
    std::array<std::int8_t, 8> out{};
    const auto radius = static_cast<std::size_t>(code.rad_axis >> 4);
    if (radius == 0) return out;
    std::array<int, 8> direction = fork_e8_root4(code.root);
    direction[static_cast<std::size_t>((code.rad_axis & 15) >> 1)] += (code.rad_axis & 1) ? -1 : 1;
    for (std::size_t i = 0; i < 8; ++i)
        out[i] = static_cast<std::int8_t>(float_codec::round_even_to_i32(
            static_cast<float>(direction[i]) * kForkE8RootRadiusScale[radius]));
    return out;
}

// Block norm in the device order (XOR 4, 2, 1 butterfly over the block's eight lanes).
inline float fork_e8_block_norm(const float* x) {
    const auto square = [&](int i) { return x[i] * x[i]; };
    return std::sqrt(((square(0) + square(4)) + (square(2) + square(6))) +
                     ((square(1) + square(5)) + (square(3) + square(7))));
}

// Norm relative to the group's reference norm scale * sqrt(8).
inline float fork_e8_relative_radius(float norm, float scale) {
    return norm / (scale * 2.82842712474619f + 1e-8f);
}

inline float fork_e8_radius_level(float relative) {
    return 3.0f * (std::log(relative) * 1.4426950408889634f) + 8.0f;
}

inline int fork_e8_radius_index(float relative) {
    if (relative < 0.08f) return 0;
    return std::clamp(static_cast<int>(std::rint(fork_e8_radius_level(relative))), 1, 15);
}

inline std::array<float, 8> fork_e8_unit(const float* x, float norm) {
    const float inverse = 1.0f / (norm + 1e-8f);
    std::array<float, 8> u{};
    for (std::size_t i = 0; i < 8; ++i) u[i] = x[i] * inverse;
    return u;
}

// u minus its projection on root r (code `root`).
inline std::array<float, 8> fork_e8_residual(const std::array<float, 8>& u, int root) {
    constexpr float kInvSqrt2 = 0.7071067811865475f;
    const std::array<int, 8> root4 = fork_e8_root4(root);
    std::array<float, 8> v{};
    float dot = 0.0f;
    for (std::size_t i = 0; i < 8; ++i) {
        v[i] = 0.25f * static_cast<float>(root4[i]);
        dot += u[i] * v[i] * kInvSqrt2;
    }
    std::array<float, 8> residual{};
    for (std::size_t i = 0; i < 8; ++i) residual[i] = u[i] - dot * (v[i] * kInvSqrt2);
    return residual;
}

// <u, r> of every root code, best first: {best score, best code, second score}.
struct ForkE8RootScores {
    float best   = -1.0f;
    int code     = 0;
    float second = -1.0f;
};

inline float fork_e8_root_score(const std::array<float, 8>& u, int code) {
    const std::array<int, 8> root4 = fork_e8_root4(code);
    float score                    = 0.0f;
    for (std::size_t i = 0; i < 8; ++i) score += 0.25f * static_cast<float>(root4[i]) * u[i];
    return score;
}

inline ForkE8RootScores fork_e8_root_scores(const std::array<float, 8>& u) {
    ForkE8RootScores scores;
    for (int code = 0; code < 240; ++code) {
        const float score = fork_e8_root_score(u, code);
        if (score > scores.best) {
            scores.second = scores.best;
            scores.best   = score;
            scores.code   = code;
        } else if (score > scores.second) {
            scores.second = score;
        }
    }
    return scores;
}

// Scalar encoder of one rotated block x (e8_encode_cylinder_8d): a radius index, the root nearest
// to x / |x| (largest <u, r>) and the largest-magnitude entry of the residual as signed axis.
inline ForkE8RootCode fork_e8_root_encode(const float* x, float scale) {
    const float norm = fork_e8_block_norm(x);
    const int radius = fork_e8_radius_index(fork_e8_relative_radius(norm, scale));
    if (radius == 0) return {};
    const std::array<float, 8> u = fork_e8_unit(x, norm);

    float pair_score = -1.0f;
    std::size_t pair = 0;
    for (std::size_t p = 0; p < kForkE8RootPairs.size(); ++p) {
        const auto& [i, j] = kForkE8RootPairs[p];
        const float score  = std::abs(u[static_cast<std::size_t>(i)]) +
                            std::abs(u[static_cast<std::size_t>(j)]);
        if (score > pair_score) {
            pair_score = score;
            pair       = p;
        }
    }
    float sum_abs        = 0.0f;
    int negative         = 0;
    std::size_t smallest = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        sum_abs += std::abs(u[i]);
        negative += u[i] < 0.0f ? 1 : 0;
        if (std::abs(u[i]) < std::abs(u[smallest])) smallest = i;
    }
    const bool odd          = (negative & 1) != 0;
    const float half_score  = 0.5f * sum_abs - (odd ? std::abs(u[smallest]) : 0.0f);
    int root                = 0;
    if (pair_score >= half_score) {
        const auto& [i, j] = kForkE8RootPairs[pair];
        root = static_cast<int>(pair) * 4 + (u[static_cast<std::size_t>(i)] >= 0.0f ? 2 : 0) +
               (u[static_cast<std::size_t>(j)] >= 0.0f ? 1 : 0);
    } else {
        int bits = 0;
        for (std::size_t i = 0; i < 7; ++i) {
            const bool positive = (u[i] >= 0.0f) != (odd && i == smallest);
            bits |= positive ? (1 << i) : 0;
        }
        root = 112 + bits;
    }

    const std::array<float, 8> residual = fork_e8_residual(u, root);
    std::size_t axis                    = 0;
    for (std::size_t i = 1; i < 8; ++i)
        if (std::abs(residual[i]) > std::abs(residual[axis])) axis = i;
    const int axis_code = static_cast<int>(axis << 1) | (residual[axis] >= 0.0f ? 0 : 1);
    return {static_cast<std::uint8_t>(root), static_cast<std::uint8_t>((radius << 4) | axis_code)};
}

// True when the device could take another decision than fork_e8_root_encode for block x because
// the radius rounding, the zero threshold, the root or the residual axis lies within
// kForkE8RootTieMargin of an alternative. Throws if the encoder misses the best root of the
// exhaustive search, which checks this restatement.
inline bool fork_e8_root_ambiguous(const float* x, float scale) {
    constexpr float margin = kForkE8RootTieMargin;
    const float norm       = fork_e8_block_norm(x);
    const float relative   = fork_e8_relative_radius(norm, scale);
    if (std::abs(relative - 0.08f) < 0.08f * margin) return true;
    if (relative < 0.08f) return false;
    const float level = fork_e8_radius_level(relative);
    if (level > 1.0f && level < 15.0f && std::abs(level - std::floor(level) - 0.5f) < margin)
        return true;
    const std::array<float, 8> u  = fork_e8_unit(x, norm);
    const ForkE8RootScores scores = fork_e8_root_scores(u);
    if (scores.best - scores.second < margin) return true;
    const std::array<float, 8> residual = fork_e8_residual(u, scores.code);
    float top = -1.0f, next = -1.0f;
    for (const float value : residual) {
        const float magnitude = std::abs(value);
        if (magnitude > top) {
            next = top;
            top  = magnitude;
        } else if (magnitude > next) {
            next = magnitude;
        }
    }
    if (top - next < margin || top < margin) return true;
    if (fork_e8_root_encode(x, scale).root != scores.code)
        throw std::logic_error("host E8 root encoder misses the best root");
    return false;
}

// True when `code` is a code the device codec may write for block x: each decision is the best
// one or lies within kForkE8RootTieMargin of it.
inline bool fork_e8_root_admissible(const float* x, float scale, ForkE8RootCode code) {
    constexpr float margin = kForkE8RootTieMargin;
    const float norm       = fork_e8_block_norm(x);
    const float relative   = fork_e8_relative_radius(norm, scale);
    const int radius       = code.rad_axis >> 4;
    if (radius == 0)
        return code.root == 0 && code.rad_axis == 0 && relative < 0.08f * (1.0f + margin);
    if (relative < 0.08f * (1.0f - margin) || code.root >= 240) return false;
    const float level = fork_e8_radius_level(relative);
    const int lowest  = std::clamp(static_cast<int>(std::rint(level - margin)), 1, 15);
    const int highest = std::clamp(static_cast<int>(std::rint(level + margin)), 1, 15);
    if (radius < lowest || radius > highest) return false;
    const std::array<float, 8> u = fork_e8_unit(x, norm);
    if (fork_e8_root_score(u, code.root) < fork_e8_root_scores(u).best - margin) return false;
    const std::array<float, 8> residual = fork_e8_residual(u, code.root);
    float top                           = 0.0f;
    for (const float value : residual) top = std::max(top, std::abs(value));
    const float axis_value = residual[static_cast<std::size_t>((code.rad_axis & 15) >> 1)];
    if (std::abs(axis_value) < top - margin) return false;
    return std::abs(axis_value) < margin || (axis_value < 0.0f) == ((code.rad_axis & 1) != 0);
}

// Encodes one 64-feature group from public BF16-represented FP32 values. INT8 writes 64 code bytes,
// INT4 and E8 lattice write 32 packed bytes, E8 root 16 bytes. Returns the FP16 scale bits.
inline std::uint16_t fork_encode_group(const float* input, ForkVectorCodec codec,
                                       std::uint8_t* codes) {
    using namespace float_codec;
    const ForkRotatedGroup group = fork_rotate_group(input, codec);
    const auto& x                = group.x;
    const float inverse_scale    = group.inverse_scale;
    if (codec == ForkVectorCodec::E8Root) {
        for (int block = 0; block < kForkGroup / 8; ++block) {
            const ForkE8RootCode code = fork_e8_root_encode(x.data() + 8 * block, group.scale);
            codes[2 * block]          = code.root;
            codes[2 * block + 1]      = code.rad_axis;
        }
        return group.scale_bits;
    }

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
    return group.scale_bits;
}

// The vector codec a fork storage applies to K or V.
inline ForkVectorCodec fork_vector_codec(KvCacheStorage storage, bool key) {
    if (!key) return ForkVectorCodec::Int4;
    switch (storage) {
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return ForkVectorCodec::Int8;
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
        return ForkVectorCodec::Int4;
    case KvCacheStorage::RK4V4E8:
        return ForkVectorCodec::E8Lattice;
    case KvCacheStorage::RK2V4E8:
        return ForkVectorCodec::E8Root;
    default:
        throw std::invalid_argument("not a fork KV storage");
    }
}

// Code bytes of one 64-feature group.
inline int fork_group_code_bytes(ForkVectorCodec codec) {
    switch (codec) {
    case ForkVectorCodec::Int8:
        return kForkGroup;
    case ForkVectorCodec::E8Root:
        return kForkGroup / 4;
    default:
        return kForkGroup / 2;
    }
}

// Encodes one D256 row: codes at codes[0, 4 * group bytes), FP16 scale bits at scales[0, 4).
inline void fork_encode_row(const float* input, ForkVectorCodec codec, std::uint8_t* codes,
                            std::uint16_t* scales) {
    const int bytes = fork_group_code_bytes(codec);
    for (int group = 0; group < kForkGroups; ++group)
        scales[group] = fork_encode_group(input + group * kForkGroup, codec, codes + group * bytes);
}

// Bit 8 * group + block is set for each E8 root block of one D256 key row whose device code may
// differ from fork_encode_row (fork_e8_root_ambiguous).
inline std::uint32_t fork_e8_root_ambiguous_blocks(const float* input) {
    std::uint32_t blocks = 0;
    for (int group = 0; group < kForkGroups; ++group) {
        const ForkRotatedGroup rotated =
            fork_rotate_group(input + group * kForkGroup, ForkVectorCodec::E8Root);
        for (int block = 0; block < kForkGroup / 8; ++block)
            if (fork_e8_root_ambiguous(rotated.x.data() + 8 * block, rotated.scale))
                blocks |= 1u << (8 * group + block);
    }
    return blocks;
}

// Whether the device wrote an admissible code (fork_e8_root_admissible) for every block of
// `blocks` in the 64 code bytes of one D256 key row.
inline bool fork_e8_root_blocks_admissible(const float* input, const std::uint8_t* codes,
                                           std::uint32_t blocks) {
    for (int group = 0; group < kForkGroups; ++group) {
        const ForkRotatedGroup rotated =
            fork_rotate_group(input + group * kForkGroup, ForkVectorCodec::E8Root);
        for (int block = 0; block < kForkGroup / 8; ++block) {
            const int index = 8 * group + block;
            if (!(blocks & (1u << index))) continue;
            const ForkE8RootCode code{codes[2 * index], codes[2 * index + 1]};
            if (!fork_e8_root_admissible(rotated.x.data() + 8 * block, rotated.scale, code))
                return false;
        }
    }
    return true;
}

} // namespace ninfer::test
