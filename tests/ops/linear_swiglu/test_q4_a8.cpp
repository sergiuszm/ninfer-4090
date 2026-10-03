#include "ops/linear_swiglu/linear_swiglu_test_common.h"
#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;
    try {
        constexpr std::array<std::int32_t, 9> tokens{1, 127, 128, 129, 255, 256, 257, 1024, 4097};
        return run_profile("LinearSwiGLU Q4_A8",
                           {QType::Q4G64_F16S, 34816, 5120, 17408, 1401U, ActivationCompute::A8},
                           tokens);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
