#pragma once

// Fork-local (rtx4090-port): small-T launch templates of the causal_cache/e8/ family. Each head
// geometry is instantiated in its own translation unit (small_t_h24.cu, small_t_h16.cu) so the
// kernel variants compile in parallel.

#include "core/device.h" // CUDA_CHECK
#include "ops/common/numeric/math.h"
#include "ops/softmax_attention/dense/causal_cache/e8/plan.h"
#include "ops/softmax_attention/kernels/causal/e8/small_t_i8.cuh"
#include "ops/softmax_attention/kernels/causal/operands.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::e8_launch {

// Every fork storage stores byte code planes and FP16 G64 scales.
template <bool Writable>
using E8KvCacheView = QuantizedCausalCacheView<std::uint8_t, __half, __half, Writable>;

struct SmallTInvocation {
    std::int32_t full_width   = 0;
    std::int32_t column_begin = 0;
    std::int32_t width        = 0;
    std::int32_t batch_size   = 1;
};

template <typename Geometry, int TokenTile, KvCacheStorage Storage, bool MultiBatch, bool Masked,
          typename CacheInput, typename CacheView>
void launch_partial(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                    CacheView cache, const SmallTInvocation& invocation,
                    std::int32_t logical_capacity, std::int32_t implementation_window,
                    std::int32_t splits, CausalPartialView partial, cudaStream_t stream) {
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    const auto launch =
        [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
            constexpr std::size_t kDynamicBytes =
                DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * e8::kCausalHeadDim) : 0ULL;
            const auto kernel = e8::causal_attention_small_t_i8_tiled_kernel<
                Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm, KeyBlock, DynamicArena, Storage,
                MultiBatch, Masked, CacheInput, CacheView>;
            if constexpr (DynamicArena) {
                static const cudaError_t attr =
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(kDynamicBytes));
                CUDA_CHECK(attr);
            }
            kernel<<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), cache, invocation.full_width,
                invocation.column_begin, logical_capacity, scale, partial.acc, partial.maximum,
                partial.sum);
        };
    if constexpr (TokenTile >= 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key
        // loop iterations; dynamic smem avoids penalizing the long-context path.
        if (implementation_window > 128 && implementation_window <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (implementation_window <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (implementation_window <= 8198) {
            launch.template operator()<12, 1, 64, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            // Two Q row tiles for the 27B group of six.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            // Three Q row tiles for the 35B group of eight. The 24/12-warp
            // routes retain eight/four consumer warps per tile; the 6-warp
            // route is reserved for long windows where CTA residency wins.
            if (implementation_window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

// Routes on the storage profile; a storage without a case here is refused, never served by
// another storage's kernels.
template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput,
          typename CacheView>
void launch_partial_for_storage(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                                KvCacheStorage storage, CacheView cache,
                                const SmallTInvocation& invocation,
                                std::int32_t logical_capacity, std::int32_t implementation_window,
                                std::int32_t splits, CausalPartialView partial,
                                cudaStream_t stream) {
    const auto launch = [&]<KvCacheStorage Storage>() {
        launch_partial<Geometry, TokenTile, Storage, MultiBatch, Masked>(
            q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
            splits, partial, stream);
    };
    switch (storage) {
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return launch.template operator()<KvCacheStorage::RotatedInt8KeyInt4ValueGroup64>();
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
        return launch.template operator()<KvCacheStorage::RotatedInt4KeyInt4ValueGroup64>();
    case KvCacheStorage::RK4V4E8:
        return launch.template operator()<KvCacheStorage::RK4V4E8>();
    case KvCacheStorage::RK2V4E8:
        return launch.template operator()<KvCacheStorage::RK2V4E8>();
    default:
        break;
    }
    throw std::invalid_argument("e8 causal attention: storage is not a fork KV mode");
}

template <typename Geometry, int TokenTile, typename CacheInput, typename CacheView>
void launch_partial_for_profile(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                                KvCacheStorage storage, CacheView cache,
                                const SmallTInvocation& invocation,
                                std::int32_t logical_capacity, std::int32_t implementation_window,
                                std::int32_t splits, CausalPartialView partial,
                                cudaStream_t stream) {
    const auto launch = [&]<bool MultiBatch, bool Masked>() {
        launch_partial_for_storage<Geometry, TokenTile, MultiBatch, Masked>(
            q, input, pos, scale, storage, cache, invocation, logical_capacity,
            implementation_window, splits, partial, stream);
    };
    const bool masked = cache.valid_columns != nullptr;
    if (invocation.batch_size == 1) {
        if (masked) {
            launch.template operator()<false, true>();
        } else {
            launch.template operator()<false, false>();
        }
    } else if (masked) {
        launch.template operator()<true, true>();
    } else {
        launch.template operator()<true, false>();
    }
}

template <typename Geometry>
void launch_reduce(const Tensor& pos, const std::int32_t* valid_columns,
                   const SmallTInvocation& invocation, std::int32_t splits,
                   CausalPartialView partial, Tensor& out, cudaStream_t stream) {
    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = Geometry::QHeads == 24 ? 256 : 64;
    const dim3 grid(Geometry::QHeads, common::div_up(e8::kCausalHeadDim, kDChunk),
                    invocation.width * invocation.batch_size);
    const auto launch = [&]<bool MultiBatch, bool Masked, bool Offset>() {
        const auto kernel =
            e8::causal_attention_small_t_reduce_output_kernel<Geometry, kDChunk, MultiBatch,
                                                              Masked, Offset>;
        kernel<<<grid, kReduceBlock, 0, stream>>>(
            partial.acc, partial.maximum, partial.sum, static_cast<const std::int32_t*>(pos.data),
            valid_columns, invocation.width, invocation.full_width, invocation.column_begin,
            invocation.batch_size, splits, static_cast<__nv_bfloat16*>(out.data));
    };
    const auto launch_offset = [&]<bool MultiBatch, bool Masked>() {
        if (invocation.column_begin == 0) {
            launch.template operator()<MultiBatch, Masked, false>();
        } else {
            launch.template operator()<MultiBatch, Masked, true>();
        }
    };
    if (invocation.batch_size == 1) {
        if (valid_columns) {
            launch_offset.template operator()<false, true>();
        } else {
            launch_offset.template operator()<false, false>();
        }
    } else if (valid_columns) {
        launch_offset.template operator()<true, true>();
    } else {
        launch_offset.template operator()<true, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

// The cache stores V rotated per 64-dimension group; rotate the BF16 output back.
template <typename Geometry>
void launch_inverse_rotate(const std::int32_t* valid_columns, const SmallTInvocation& invocation,
                           Tensor& out, cudaStream_t stream) {
    const int units =
        invocation.batch_size * invocation.width * Geometry::QHeads * kv::kInt8Groups;
    kv_cache_inverse_rotate_output_kernel<Geometry::QHeads><<<units, 32, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out.data), invocation.width, invocation.full_width,
        invocation.column_begin, valid_columns);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheInput>
void small_t_launch_for(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                        KvCacheStorage storage, E8KvCacheView<CacheInput::writes_cache> cache,
                        const SmallTInvocation& invocation,
                        CausalAttentionExecutionEnvelope envelope, CausalPartialView partial,
                        Tensor& out, cudaStream_t stream) {
    const auto logical_capacity      = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto implementation_window = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits =
        e8_kv_split_capacity(Geometry::QHeads, invocation.width, envelope, invocation.batch_size);
    const auto launch = [&]<int TokenTile>() {
        launch_partial_for_profile<Geometry, TokenTile>(q, input, pos, scale, storage, cache,
                                                        invocation, logical_capacity,
                                                        implementation_window, splits, partial,
                                                        stream);
    };
    switch (invocation.width) {
    case 1:
        launch.template operator()<1>();
        break;
    case 2:
        launch.template operator()<2>();
        break;
    case 3:
        launch.template operator()<3>();
        break;
    case 4:
        launch.template operator()<4>();
        break;
    case 5:
        launch.template operator()<5>();
        break;
    case 6:
        launch.template operator()<6>();
        break;
    case 7:
        if constexpr (Geometry::QHeads == 24) {
            launch.template operator()<7>();
            break;
        }
        throw std::invalid_argument("e8 causal attention: unsupported query-row tile");
    case 8:
        if constexpr (Geometry::QHeads == 24) {
            launch.template operator()<8>();
            break;
        }
        throw std::invalid_argument("e8 causal attention: unsupported query-row tile");
    default:
        throw std::invalid_argument("e8 causal attention small-T: unsupported T");
    }
    launch_reduce<Geometry>(pos, cache.valid_columns, invocation, splits, partial, out, stream);
    launch_inverse_rotate<Geometry>(cache.valid_columns, invocation, out, stream);
}

#define NINFER_E8_SMALL_T_INSTANCE(PREFIX, GEOMETRY, INPUT)                                        \
    PREFIX template void small_t_launch_for<GEOMETRY, INPUT>(                                      \
        const Tensor&, INPUT, const Tensor&, float, KvCacheStorage,                                \
        E8KvCacheView<INPUT::writes_cache>, const SmallTInvocation&,                               \
        CausalAttentionExecutionEnvelope, CausalPartialView, Tensor&, cudaStream_t)

NINFER_E8_SMALL_T_INSTANCE(extern, e8::CausalD256H24Kv4, CausalAppendInput);
NINFER_E8_SMALL_T_INSTANCE(extern, e8::CausalD256H24Kv4, CausalCachedInput);
NINFER_E8_SMALL_T_INSTANCE(extern, e8::CausalD256H16Kv2, CausalAppendInput);
NINFER_E8_SMALL_T_INSTANCE(extern, e8::CausalD256H16Kv2, CausalCachedInput);

} // namespace ninfer::ops::detail::e8_launch
