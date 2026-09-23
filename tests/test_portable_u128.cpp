#include "runtime/portable_u128.h"

#include <cstdint>
#include <limits>

namespace {

using Emulated = ninfer::detail::u128;
using Native   = unsigned __int128;

bool matches(Emulated emulated, Native native) {
    return emulated.hi == static_cast<std::uint64_t>(native >> 64) &&
           emulated.lo == static_cast<std::uint64_t>(native);
}

} // namespace

int main() {
    using namespace ninfer::detail;

    const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t q32_one = 1ull << 32;
    const std::uint64_t coefficient = 5ull << 32;
    const std::uint64_t units = 1000;

    if (!matches(u128_not(u128_from64(0)), ~Native(0))) { return 1; }
    if (!matches(u128_shl(u128_from64(maximum), 32), Native(maximum) << 32)) { return 1; }
    if (!matches(u128_shl(u128_from64(1), 40), Native(1) << 40)) { return 1; }

    const Native product = Native(123456789ull) * 987654321ull;
    if (!matches(u128_mul64(123456789ull, 987654321ull), product)) { return 1; }
    if (!matches(u128_shr(u128_mul64(123456789ull, 987654321ull), 32), product >> 32)) {
        return 1;
    }

    const Emulated emulated_product = u128_mul64(coefficient, units);
    const Emulated emulated_rounded = u128_add(emulated_product, u128_from64(q32_one - 1));
    const std::uint64_t emulated_result = u128_to64(u128_shr(emulated_rounded, 32));
    const Native native_product = Native(coefficient) * units;
    const std::uint64_t native_result = static_cast<std::uint64_t>(
        (native_product + q32_one - 1) >> 32);
    return emulated_result == native_result ? 0 : 1;
}