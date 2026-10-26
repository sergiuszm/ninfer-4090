#pragma once

// Fork-local (rtx4090-port): small-T launch templates of the causal_cache/e8/ family. Each head
// geometry is instantiated in its own translation unit (small_t_h24.cu, small_t_h16.cu) so the
// kernel variants compile in parallel.

#include "core/device.h" // CUDA_CHECK
#include "core/paged_kv_storage.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/e8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/e8/small_t_i8.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::e8_launch {

struct SmallTInvocation {
    const Tensor* valid_columns = nullptr;
    const Tensor* table_rows    = nullptr;
    std::int32_t full_width     = 0;
    std::int32_t column_begin   = 0;
    std::int32_t width          = 0;
    std::int32_t batch_size     = 1;
};

template <typename Geometry, int TokenTile, bool PackedV, bool RotateK, bool RotateV, bool PackedK,
          bool E8Lattice, bool E8Root, bool MultiBatch, bool Masked, typename CacheInput>
void launch_partial(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                    PagedKVBatchLayerView cache, const SmallTInvocation& invocation,
                    std::int32_t logical_capacity, std::int32_t implementation_window,
                    std::int32_t splits, E8SmallTPartials& partial, cudaStream_t stream) {
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    const auto launch =
        [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
            constexpr std::size_t kDynamicBytes =
                DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * e8::kCausalHeadDim) : 0ULL;
            const auto kernel = e8::causal_attention_small_t_i8_tiled_kernel<
                Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm, KeyBlock, DynamicArena, PackedV,
                RotateK, RotateV, PackedK, E8Lattice, E8Root, MultiBatch, Masked, CacheInput>;
            if constexpr (DynamicArena) {
                static const cudaError_t attr =
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(kDynamicBytes));
                CUDA_CHECK(attr);
            }
            kernel<<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::uint8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, scale, static_cast<float*>(partial.acc.data),
                static_cast<float*>(partial.m.data), static_cast<float*>(partial.l.data));
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

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_partial_for_mode(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                             PagedKVBatchLayerView cache, const SmallTInvocation& invocation,
                             std::int32_t logical_capacity, std::int32_t implementation_window,
                             std::int32_t splits, E8SmallTPartials& partial, cudaStream_t stream) {
    const KvForkModeFlags mode = kv_fork_mode_flags(cache.storage);
    const auto launch          = [&]<bool PackedK, bool E8Lattice, bool E8Root>() {
        launch_partial<Geometry, TokenTile, true, true, true, PackedK, E8Lattice, E8Root,
                                MultiBatch, Masked>(q, input, pos, scale, cache, invocation,
                                                    logical_capacity, implementation_window, splits, partial,
                                                    stream);
    };
    if (mode.e8_root) {
        launch.template operator()<false, false, true>();
    } else if (mode.e8_lattice) {
        launch.template operator()<true, true, false>();
    } else if (mode.packed_k) {
        launch.template operator()<true, false, false>();
    } else {
        launch.template operator()<false, false, false>();
    }
}

template <typename Geometry, int TokenTile, typename CacheInput>
void launch_partial_for_profile(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                                PagedKVBatchLayerView cache, const SmallTInvocation& invocation,
                                std::int32_t logical_capacity, std::int32_t implementation_window,
                                std::int32_t splits, E8SmallTPartials& partial,
                                cudaStream_t stream) {
    const auto launch = [&]<bool MultiBatch, bool Masked>() {
        launch_partial_for_mode<Geometry, TokenTile, MultiBatch, Masked>(
            q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
            splits, partial, stream);
    };
    const bool masked = invocation.valid_columns != nullptr;
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

// Every fork storage stores V rotated per 64-dimension group (KvForkModeFlags::rotate_v), so the
// reduce always rotates the merged FP32 output back before its only BF16 rounding.
template <typename Geometry>
void launch_reduce(const Tensor& pos, const SmallTInvocation& invocation, std::int32_t splits,
                   const E8SmallTPartials& partial, Tensor& out, cudaStream_t stream) {
    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = Geometry::QHeads == 24 ? 256 : 64;
    const dim3 grid(Geometry::QHeads, div_up(e8::kCausalHeadDim, kDChunk),
                    invocation.width * invocation.batch_size);
    const auto launch = [&]<bool MultiBatch, bool Masked, bool Offset>() {
        const auto kernel =
            e8::causal_attention_small_t_reduce_output_kernel<Geometry, kDChunk, true, MultiBatch,
                                                              Masked, Offset, true>;
        kernel<<<grid, kReduceBlock, 0, stream>>>(
            static_cast<const float*>(partial.acc.data), static_cast<const float*>(partial.m.data),
            static_cast<const float*>(partial.l.data), static_cast<const std::int32_t*>(pos.data),
            invocation.valid_columns
                ? static_cast<const std::int32_t*>(invocation.valid_columns->data)
                : nullptr,
            invocation.width, invocation.full_width, invocation.column_begin, invocation.batch_size,
            splits, static_cast<__nv_bfloat16*>(out.data));
    };
    const auto launch_offset = [&]<bool MultiBatch, bool Masked>() {
        if (invocation.column_begin == 0) {
            launch.template operator()<MultiBatch, Masked, false>();
        } else {
            launch.template operator()<MultiBatch, Masked, true>();
        }
    };
    if (invocation.batch_size == 1) {
        if (invocation.valid_columns) {
            launch_offset.template operator()<false, true>();
        } else {
            launch_offset.template operator()<false, false>();
        }
    } else if (invocation.valid_columns) {
        launch_offset.template operator()<true, true>();
    } else {
        launch_offset.template operator()<true, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheInput>
void small_t_launch_for(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                        PagedKVBatchLayerView cache, const SmallTInvocation& invocation,
                        CausalAttentionExecutionEnvelope envelope, E8SmallTPartials& partial,
                        Tensor& out, cudaStream_t stream) {
    const auto logical_capacity      = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto implementation_window = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits =
        e8_kv_split_capacity(Geometry::QHeads, invocation.width, envelope, invocation.batch_size);
    const auto launch = [&]<int TokenTile>() {
        launch_partial_for_profile<Geometry, TokenTile>(q, input, pos, scale, cache, invocation,
                                                        logical_capacity, implementation_window,
                                                        splits, partial, stream);
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
    launch_reduce<Geometry>(pos, invocation, splits, partial, out, stream);
}

#define NINFER_E8_SMALL_T_INSTANCE(PREFIX, GEOMETRY, INPUT)                                        \
    PREFIX template void small_t_launch_for<GEOMETRY, INPUT>(                                      \
        const Tensor&, INPUT, const Tensor&, float, PagedKVBatchLayerView,                         \
        const SmallTInvocation&, CausalAttentionExecutionEnvelope, E8SmallTPartials&, Tensor&,     \
        cudaStream_t)

NINFER_E8_SMALL_T_INSTANCE(extern, e8::CausalD256H24Kv4, e8::CausalAppendInput);
NINFER_E8_SMALL_T_INSTANCE(extern, e8::CausalD256H24Kv4, e8::CausalCachedInput);
NINFER_E8_SMALL_T_INSTANCE(extern, e8::CausalD256H16Kv2, e8::CausalAppendInput);
NINFER_E8_SMALL_T_INSTANCE(extern, e8::CausalD256H16Kv2, e8::CausalCachedInput);

} // namespace ninfer::ops::detail::e8_launch
