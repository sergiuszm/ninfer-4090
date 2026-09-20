// portable_u128.h — 128-bit unsigned arithmetic that builds on both GCC/Clang and MSVC.
//
// The codebase uses `unsigned __int128` for 64-bit *saturating* math (overflow-clamped
// products and the prefill attention-pairs cost model). MSVC on x64 has no `__int128`
// (hard error C4235), so this shim provides an API that is:
//   - bit-identical to native `unsigned __int128` on GCC/Clang (the tested Linux/5090 path),
//   - a two-limb emulation on MSVC, using 32-bit limb math so it itself needs no 64x64->128
//     intrinsics.
// All operations are `constexpr` so they can be used in constant expressions (the cost model
// does `constexpr U128 maximum = ~0`). Only the operations the codebase actually uses exist.

#pragma once

#include <cstdint>

#if defined(__SIZEOF_INT128__)
// GCC / Clang: native 128-bit type. Every op is a single native op -> identical results.
namespace ninfer {
inline namespace detail {
using u128 = unsigned __int128;
inline constexpr u128 u128_from64(std::uint64_t v) noexcept { return static_cast<u128>(v); }
inline constexpr u128 u128_mul64(std::uint64_t a, std::uint64_t b) noexcept { return static_cast<u128>(a) * b; }
inline constexpr u128 u128_add(u128 a, u128 b) noexcept { return a + b; }
inline constexpr u128 u128_sub(u128 a, u128 b) noexcept { return a - b; }
inline constexpr u128 u128_shl(u128 a, int n) noexcept { return a << n; }
inline constexpr u128 u128_shr(u128 a, int n) noexcept { return a >> n; }
inline constexpr u128 u128_not(u128 a) noexcept { return ~a; }
inline constexpr bool u128_gt(u128 a, u128 b) noexcept { return a > b; }
inline constexpr bool u128_ge(u128 a, u128 b) noexcept { return a >= b; }
inline constexpr std::uint64_t u128_to64(u128 a) noexcept { return static_cast<std::uint64_t>(a); }
} // namespace detail
} // namespace ninfer

#else
// MSVC: emulate with {hi, lo} 64-bit limbs via 32-bit limb arithmetic (no intrinsics).
namespace ninfer {
inline namespace detail {
struct u128 {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;
};
inline constexpr u128 u128_from64(std::uint64_t v) noexcept { return {v, 0}; }
// 64x64 -> 128, using four 32x32->64 partial products (each < 2^64, so no overflow).
inline constexpr u128 u128_mul64(std::uint64_t a, std::uint64_t b) noexcept {
    const std::uint64_t a0 = a & 0xffffffffull, a1 = a >> 32;
    const std::uint64_t b0 = b & 0xffffffffull, b1 = b >> 32;
    const std::uint64_t p0 = a0 * b0;
    const std::uint64_t p1 = a0 * b1;
    const std::uint64_t p2 = a1 * b0;
    const std::uint64_t p3 = a1 * b1;
    const std::uint64_t mid = (p0 >> 32) + (p1 & 0xffffffffull) + (p2 & 0xffffffffull);
    const std::uint64_t lo  = (mid << 32) | (p0 & 0xffffffffull);
    const std::uint64_t hi  = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    return {lo, hi};
}
inline constexpr u128 u128_add(u128 a, u128 b) noexcept { return {a.lo + b.lo, a.hi + b.hi + (a.lo + b.lo < a.lo ? 1 : 0)}; }
inline constexpr u128 u128_sub(u128 a, u128 b) noexcept { return {a.lo - b.lo, a.hi - b.hi - (a.lo < b.lo ? 1 : 0)}; }
inline constexpr u128 u128_shl(u128 a, int n) noexcept {
    if (n == 0) { return a; }
    if (n < 32) { return {a.lo << n, (a.hi << n) | (a.lo >> (64 - n))}; }
    return {0, a.lo << (n - 32)};
}
inline constexpr u128 u128_shr(u128 a, int n) noexcept {
    if (n == 0) { return a; }
    if (n < 32) { return {(a.lo >> n) | (a.hi << (64 - n)), a.hi >> n}; }
    return {a.hi >> (n - 32), 0};
}
inline constexpr u128 u128_not(u128 a) noexcept { return {!a.lo, !a.hi}; }
inline constexpr bool u128_gt(u128 a, u128 b) noexcept { return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo); }
inline constexpr bool u128_ge(u128 a, u128 b) noexcept { return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo); }
inline constexpr std::uint64_t u128_to64(u128 a) noexcept { return a.lo; }
} // namespace detail
} // namespace ninfer
#endif
