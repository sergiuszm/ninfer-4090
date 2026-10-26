#pragma once

// Fork-local (rtx4090-port): route and workspace policy of the causal_cache/e8/ family. The
// family serves every fork KV storage (kv_storage_is_fork_mode): rk8v4, rk4v4, rk4v4-e8 and
// rk2v4-e8. All of them run the same int8 small-T and prompt kernels, so the policy does not
// depend on the storage.

#include "core/tensor.h"
#include "ninfer/ops/softmax_attention.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class E8KvRoute { SmallT, ChunkedSmallT, Prompt };

[[nodiscard]] E8KvRoute e8_kv_resolve_route(std::int32_t q_heads, std::int32_t width,
                                            std::int32_t batch_size,
                                            CausalAttentionExecutionEnvelope envelope);

// Query columns per small-T launch of the chunked route.
[[nodiscard]] std::int32_t e8_kv_chunk_tokens(std::int32_t q_heads, std::int32_t width,
                                              std::int32_t batch_size,
                                              CausalAttentionExecutionEnvelope envelope);

// Upper bound of the device-side active-split policy for one small-T launch.
[[nodiscard]] std::int32_t e8_kv_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                                CausalAttentionExecutionEnvelope envelope,
                                                std::int32_t batch_size = 1);

[[nodiscard]] std::size_t e8_kv_workspace_bytes(std::int32_t q_heads, std::int32_t batch_size,
                                                std::int32_t min_width, std::int32_t max_width,
                                                CausalAttentionExecutionEnvelope envelope,
                                                std::int32_t multiprocessor_count);

struct E8SmallTPartials {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <class Allocator>
E8SmallTPartials allocate_e8_small_t_partials(Allocator& workspace, std::int32_t q_heads,
                                              std::int32_t tokens, std::int32_t splits,
                                              std::int32_t batch_size) {
    constexpr std::int32_t kHeadDim = 256;
    return {
        workspace.alloc(DType::FP32, {kHeadDim, q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
    };
}

} // namespace ninfer::ops::detail
