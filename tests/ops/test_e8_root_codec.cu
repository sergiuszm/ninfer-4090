// Coverage for the production E8 root codec in src/ops/kernel/e8_root_codec.cuh.
//
// tools/test_kv/ carries an independent reimplementation of the E8 mathematics; it
// never includes this header, so it cannot catch a regression in the code the rk2v4-e8
// fill path actually compiles. This suite drives the shipped functions directly.
//
// The warp encoder derives its radius index from a per-8-lane-subgroup norm, so the
// four subgroups of one warp can disagree about whether their radius is zero. Every
// __shfl*_sync inside it names the full 32-lane mask, which makes "one subgroup takes
// a different path" undefined behaviour. The contamination case below pins the contract
// down: zeroing one subgroup must not move the codes of its neighbours.
//
// That case is a contract guard, not a reproducer. Measured on sm_89 with CUDA 13.1.2,
// the pre-hardening codec passes it and compute-sanitizer --tool synccheck reports no
// error, because the shuffles use XOR masks {1,2,4} that never cross an 8-lane boundary,
// so a subgroup that returned early holds nothing its neighbours read. The divergence is
// a specification violation whose damage is latent - a different arch, scheduler, or
// compiler is free to make it real. The guard exists so that day is a red test.

#include "ops/kernel/e8_root_codec.cuh"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kDims             = 8;  // one E8 subspace
constexpr int kLanesPerWarp     = 32;
constexpr int kSubgroupsPerWarp = kLanesPerWarp / kDims;

// One warp per row. Lane L carries dimension (L & 7) of subgroup (L >> 3), the same
// mapping the rk2v4-e8 fill kernels use.
__global__ void encode_warp_kernel(const float* values, const float* scales,
                                   std::uint8_t* roots, std::uint8_t* rad_axes) {
    const int warp        = blockIdx.x;
    const int lane        = threadIdx.x;
    const float value     = values[warp * kLanesPerWarp + lane];
    std::uint8_t root     = 0;
    std::uint8_t rad_axis = 0;
    ops::e8_encode_cylinder_8d_warp(value, scales[warp], root, rad_axis, lane);
    if ((lane & 7) == 0) {
        const int subgroup = warp * kSubgroupsPerWarp + (lane >> 3);
        roots[subgroup]    = root;
        rad_axes[subgroup] = rad_axis;
    }
}

// Scalar reference over the same subspaces.
__global__ void encode_scalar_kernel(const float* values, const float* scales,
                                     std::uint8_t* roots, std::uint8_t* rad_axes,
                                     int subgroups) {
    const int subgroup = blockIdx.x * blockDim.x + threadIdx.x;
    if (subgroup >= subgroups) return;
    float rot[kDims];
#pragma unroll
    for (int i = 0; i < kDims; ++i) rot[i] = values[subgroup * kDims + i];
    std::uint8_t root     = 0;
    std::uint8_t rad_axis = 0;
    ops::e8_encode_cylinder_8d(rot, scales[subgroup / kSubgroupsPerWarp], root, rad_axis);
    roots[subgroup]    = root;
    rad_axes[subgroup] = rad_axis;
}

struct Codes {
    std::vector<std::uint8_t> roots;
    std::vector<std::uint8_t> rad_axes;
};

Codes encode_warp(const std::vector<float>& values, const std::vector<float>& scales) {
    const int warps     = static_cast<int>(scales.size());
    const int subgroups = warps * kSubgroupsPerWarp;
    GuardedDeviceBuffer device_values(values.size() * sizeof(float));
    GuardedDeviceBuffer device_scales(scales.size() * sizeof(float));
    GuardedDeviceBuffer device_roots(static_cast<std::size_t>(subgroups));
    GuardedDeviceBuffer device_rad_axes(static_cast<std::size_t>(subgroups));
    device_values.copy_from_host(values.data(), device_values.bytes());
    device_scales.copy_from_host(scales.data(), device_scales.bytes());

    encode_warp_kernel<<<warps, kLanesPerWarp>>>(
        static_cast<const float*>(device_values.data()),
        static_cast<const float*>(device_scales.data()),
        static_cast<std::uint8_t*>(device_roots.data()),
        static_cast<std::uint8_t*>(device_rad_axes.data()));
    cuda_check_last_launch("encode_warp_kernel");
    cuda_synchronize();

    Codes codes{from_device<std::uint8_t>(device_roots.data(), subgroups),
                from_device<std::uint8_t>(device_rad_axes.data(), subgroups)};
    if (device_values.verify_guards("e8 warp values") ||
        device_roots.verify_guards("e8 warp roots") ||
        device_rad_axes.verify_guards("e8 warp rad_axes")) {
        codes.roots.clear();
    }
    return codes;
}

Codes encode_scalar(const std::vector<float>& values, const std::vector<float>& scales) {
    const int subgroups = static_cast<int>(scales.size()) * kSubgroupsPerWarp;
    GuardedDeviceBuffer device_values(values.size() * sizeof(float));
    GuardedDeviceBuffer device_scales(scales.size() * sizeof(float));
    GuardedDeviceBuffer device_roots(static_cast<std::size_t>(subgroups));
    GuardedDeviceBuffer device_rad_axes(static_cast<std::size_t>(subgroups));
    device_values.copy_from_host(values.data(), device_values.bytes());
    device_scales.copy_from_host(scales.data(), device_scales.bytes());

    constexpr int kBlock = 128;
    encode_scalar_kernel<<<(subgroups + kBlock - 1) / kBlock, kBlock>>>(
        static_cast<const float*>(device_values.data()),
        static_cast<const float*>(device_scales.data()),
        static_cast<std::uint8_t*>(device_roots.data()),
        static_cast<std::uint8_t*>(device_rad_axes.data()), subgroups);
    cuda_check_last_launch("encode_scalar_kernel");
    cuda_synchronize();

    return {from_device<std::uint8_t>(device_roots.data(), subgroups),
            from_device<std::uint8_t>(device_rad_axes.data(), subgroups)};
}

std::vector<float> make_values(int warps, std::uint32_t seed) {
    std::vector<float> values(static_cast<std::size_t>(warps) * kLanesPerWarp);
    fill_uniform(values, seed, -1.0f, 1.0f);
    return values;
}

void zero_subgroup(std::vector<float>& values, int warp, int subgroup) {
    const std::size_t base = static_cast<std::size_t>(warp) * kLanesPerWarp + subgroup * kDims;
    for (int i = 0; i < kDims; ++i) values[base + i] = 0.0f;
}

// A zero-radius subgroup must emit the zero code and, crucially, must not disturb the
// subgroups that share its warp.
int run_contamination_case(const char* label, int warps, std::uint32_t seed,
                           const std::vector<int>& zeroed_subgroups) {
    const std::vector<float> scales(static_cast<std::size_t>(warps), 0.25f);
    const std::vector<float> dense = make_values(warps, seed);
    const Codes reference          = encode_warp(dense, scales);
    if (reference.roots.empty()) return 1;

    std::vector<float> sparse = dense;
    for (int warp = 0; warp < warps; ++warp) {
        for (int subgroup : zeroed_subgroups) zero_subgroup(sparse, warp, subgroup);
    }
    const Codes got = encode_warp(sparse, scales);
    if (got.roots.empty()) return 1;

    std::vector<std::uint8_t> expected_roots    = reference.roots;
    std::vector<std::uint8_t> expected_rad_axes = reference.rad_axes;
    for (int warp = 0; warp < warps; ++warp) {
        for (int subgroup : zeroed_subgroups) {
            const int index          = warp * kSubgroupsPerWarp + subgroup;
            expected_roots[index]    = 0;
            expected_rad_axes[index] = 0;
        }
    }

    int failures = verify_exact(label, got.roots, expected_roots);
    failures += verify_exact(label, got.rad_axes, expected_rad_axes);
    return failures;
}

std::vector<std::uint8_t> radius_indices(const std::vector<std::uint8_t>& rad_axes) {
    std::vector<std::uint8_t> radii(rad_axes.size());
    for (std::size_t i = 0; i < rad_axes.size(); ++i) radii[i] = rad_axes[i] >> 4;
    return radii;
}

// The warp encoder and the scalar encoder implement the same cylinder quantizer, so the
// root and the radius index must agree exactly.
//
// The residual axis nibble is deliberately excluded. When the primary root is Type A it
// carries +-1 in exactly two coordinates, which leaves the residual exactly symmetric
// across that pair - |res| ties to the last bit, not by a rounding margin. The two
// encoders then break the tie differently: the scalar scan keeps the first maximum while
// the warp argmax keeps the other one. Either axis is an equally valid nearest axis, and
// only the warp encoder ever writes the cache (the scalar path reaches production through
// no call site), so this is an unspecified tie-break rather than a defect. Changing it
// would rewrite stored rk2v4-e8 codes, so it is documented here rather than normalised.
int run_parity_case(const char* label, int warps, std::uint32_t seed) {
    const std::vector<float> scales(static_cast<std::size_t>(warps), 0.25f);
    const std::vector<float> values = make_values(warps, seed);
    const Codes warp_codes          = encode_warp(values, scales);
    if (warp_codes.roots.empty()) return 1;
    const Codes scalar_codes = encode_scalar(values, scales);

    int failures = verify_exact(label, warp_codes.roots, scalar_codes.roots);
    failures += verify_exact(label, radius_indices(warp_codes.rad_axes),
                             radius_indices(scalar_codes.rad_axes));
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_parity_case("e8 cylinder warp/scalar parity", 512, 1101u);
    failures += run_contamination_case("e8 cylinder zero subgroup 0", 512, 2201u, {0});
    failures += run_contamination_case("e8 cylinder zero subgroup 3", 512, 2202u, {3});
    failures += run_contamination_case("e8 cylinder zero subgroups 1,2", 512, 2203u, {1, 2});
    failures += run_contamination_case("e8 cylinder all subgroups zero", 64, 2204u, {0, 1, 2, 3});
    std::cout << (failures ? "FAIL" : "OK") << " e8_root_codec\n";
    return failures ? 1 : 0;
}
