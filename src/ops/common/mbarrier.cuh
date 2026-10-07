#pragma once

#include "ops/common/memory.cuh"

#include <cstdint>

namespace ninfer::ops {

// Fork-local (rtx4090-port): mbarrier transactions and their fences need sm_90. An sm_89 build
// still compiles every caller, but these bodies trap there. Host code keeps sm_89 off the TMA
// paths (catch-up #4 stage 1, docs/maintainer/port-ledger.md).

__device__ __forceinline__ void cta_mbarrier_init(std::uint64_t* barrier, std::uint32_t arrivals) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap();
#else
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(arrivals)
                 : "memory");
#endif
}

__device__ __forceinline__ void cta_mbarrier_wait(std::uint64_t* barrier, std::uint32_t phase) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap();
#else
    constexpr std::uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n"
                 ".reg .pred done;\n"
                 "wait_loop_%=: \n"
                 "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
                 "@done bra wait_done_%=;\n"
                 "bra wait_loop_%=;\n"
                 "wait_done_%=: \n"
                 "}\n"
                 :
                 : "r"(smem_addr(barrier)), "r"(phase), "r"(kSuspendTicks)
                 : "memory");
#endif
}

__device__ __forceinline__ void cta_mbarrier_arrive(std::uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap();
#else
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(smem_addr(barrier)) : "memory");
#endif
}

__device__ __forceinline__ void cta_mbarrier_arrive_expect_tx(std::uint64_t* barrier,
                                                              std::uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap();
#else
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(bytes)
                 : "memory");
#endif
}

__device__ __forceinline__ void cta_mbarrier_fence_init() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap();
#else
    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
#endif
}

} // namespace ninfer::ops
