// Fork-local (rtx4090-port): route, chunk and split policy of the causal_cache/e8/ family. The
// numbers are the rtx4090-port int8-family values (causal_softmax_attention.cpp, small_t.cu).
#include "ops/softmax_attention/dense/causal_cache/e8/plan.h"

#include "core/layout.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/e8/geometry.cuh"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kMaximumVerifyTokens = 16;

// Supplies an upper bound for the device-side active-split policy over one explicit execution
// envelope. Eager calls normally pass an exact window; graph calls pass their target-private
// replay interval.
template <typename Geometry>
std::int32_t split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::SmallTMaximumSplits; }

    constexpr std::int32_t kMinSplits = 4 * Geometry::SmallTSplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / Geometry::SmallTSplitScale);
    if (window > 4096) { include_tier(8198, 128 / Geometry::SmallTSplitScale); }
    if (window > 8198) { include_tier(16390, 256 / Geometry::SmallTSplitScale); }
    if (window > 16390) { include_tier(window, 480 / Geometry::SmallTSplitScale); }

    return (splits < Geometry::SmallTMaximumSplits) ? splits : Geometry::SmallTMaximumSplits;
}

template <typename Geometry>
std::int32_t split_count(std::int32_t window, std::int32_t tokens) {
    // A 64-key default split just above a 32-key boundary makes the partial kernel execute a
    // nearly empty second tile. T=5 uses one 32-key tile per split; the short T>=6 profile keeps
    // all newly appended rows in one tail split while retaining a useful B=8 grid.
    if (tokens == 5 && window > 128 && window <= 512) {
        return div_up(window, 32 / Geometry::SmallTSplitScale);
    }
    if (tokens >= 6 && window > 128 && window <= 160) {
        constexpr std::int32_t kKeysPerSplit = Geometry::SmallTSplitScale == 2 ? 17 : 24;
        return div_up(window, kKeysPerSplit);
    }
    // Bc=64 is one CTA/SM on these model shapes. Keep the 8K grid at or below
    // one 170-SM wave after accounting for the geometry's KV-head count.
    if (tokens >= 6 && window > 5000 && window <= 8198) {
        const std::int32_t splits   = div_up(window, 192 / Geometry::SmallTSplitScale);
        constexpr std::int32_t kMin = 4 * Geometry::SmallTSplitScale;
        constexpr std::int32_t kMax = 42 * Geometry::SmallTSplitScale;
        const std::int32_t clamped  = (splits > kMin) ? splits : kMin;
        return (clamped < kMax) ? clamped : kMax;
    }
    return split_upper_bound<Geometry>(window);
}

template <typename Geometry>
std::int32_t launch_capacity(CausalAttentionExecutionEnvelope envelope, std::int32_t tokens) {
    std::int32_t capacity = 0;
    const auto include    = [&](std::uint32_t window) {
        if (window < envelope.min_visible_keys || window > envelope.max_visible_keys) { return; }
        const auto splits = split_count<Geometry>(static_cast<std::int32_t>(window), tokens);
        capacity          = capacity > splits ? capacity : splits;
    };
    include(envelope.min_visible_keys);
    include(envelope.max_visible_keys);
    // The policy is monotonic inside these finite segments and may drop when crossing a boundary.
    // Evaluating every segment end plus both interval ends gives the exact interval maximum.
    constexpr std::uint32_t ends[] = {128, 160, 512, 4096, 5000, 8198, 16390};
    for (const std::uint32_t end : ends) { include(end); }
    return capacity;
}

} // namespace

E8KvRoute e8_kv_resolve_route(std::int32_t q_heads, std::int32_t width, std::int32_t batch_size,
                              CausalAttentionExecutionEnvelope envelope) {
    // The route depends on the call shape only, never on the visible-key envelope. A CUDA graph
    // captured at one envelope is updated in place for another, so both must have the same node
    // topology. rtx4090-port also took the prompt route for 9-16 columns over short windows (up
    // to 256 keys for 24 query heads, 512/1024 for 16); that changed the node count at the
    // window boundary.
    (void)batch_size;
    (void)envelope;
    if (width > kMaximumVerifyTokens) { return E8KvRoute::Prompt; }
    const std::int32_t small_t_width = q_heads == 24 ? 8 : 6;
    return width <= small_t_width ? E8KvRoute::SmallT : E8KvRoute::ChunkedSmallT;
}

std::int32_t e8_kv_chunk_tokens(std::int32_t q_heads, std::int32_t width, std::int32_t batch_size,
                                CausalAttentionExecutionEnvelope envelope) {
    if (q_heads == 16) { return 6; }
    // INT8 benefits from 5+4/5 at long contexts.
    if (batch_size == 1 && width >= 9 && width <= 10 && envelope.max_visible_keys > 4096) {
        return (width + 1) / 2;
    }
    return 8;
}

std::int32_t e8_kv_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                  CausalAttentionExecutionEnvelope envelope,
                                  std::int32_t batch_size) {
    if (tokens < 1 || tokens > (q_heads == 24 ? 8 : 6) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys) {
        throw std::invalid_argument("e8 causal attention split capacity: invalid profile");
    }
    if (q_heads == e8::CausalD256H24Kv4::QHeads) {
        const std::int32_t capacity = launch_capacity<e8::CausalD256H24Kv4>(envelope, tokens);
        if (batch_size > 1) {
            // Keep complete grids within one or two 170-SM waves. Rounding from 160 CTAs
            // leaves room for the indivisible 4*B group, including B=3/5/6/7.
            const bool narrow              = tokens <= 5;
            const std::int32_t target_ctas = narrow || envelope.max_visible_keys > 4096 ? 320 : 160;
            const std::int32_t grid_limit  = div_up(target_ctas, 4 * batch_size);
            // A split stages at most 64 physical-page IDs. Leave two 64-key pages for
            // key-tile rounding and page alignment at the 262144-key resource limit.
            const std::int32_t page_limit =
                div_up(static_cast<std::int32_t>(envelope.max_visible_keys), 3968);
            return std::min(capacity, std::max({4, grid_limit, page_limit}));
        }
        return capacity;
    }
    if (q_heads == e8::CausalD256H16Kv2::QHeads) {
        return launch_capacity<e8::CausalD256H16Kv2>(envelope, tokens);
    }
    throw std::invalid_argument("e8 causal attention split capacity: unsupported head geometry");
}

std::size_t e8_kv_workspace_bytes(std::int32_t q_heads, std::int32_t batch_size,
                                  std::int32_t min_width, std::int32_t max_width,
                                  CausalAttentionExecutionEnvelope envelope,
                                  std::int32_t multiprocessor_count) {
    // The fork policy is tuned for the RTX 4090 and does not scale with the SM count yet.
    (void)multiprocessor_count;
    const auto chunk_capacity = [&](std::int32_t width) {
        const std::int32_t splits = e8_kv_split_capacity(q_heads, width, envelope, batch_size);
        WorkspaceLayoutBuilder layout;
        (void)allocate_e8_small_t_partials(layout, q_heads, width, splits, batch_size);
        return layout.peak_bytes(1);
    };
    const auto exact_capacity = [&](std::int32_t width) {
        const E8KvRoute route = e8_kv_resolve_route(q_heads, width, batch_size, envelope);
        if (route == E8KvRoute::Prompt) { return std::size_t{0}; }
        if (route == E8KvRoute::SmallT) { return chunk_capacity(width); }
        std::size_t maximum         = 0;
        const std::int32_t per_call = e8_kv_chunk_tokens(q_heads, width, batch_size, envelope);
        for (std::int32_t begin = 0; begin < width; begin += per_call) {
            maximum = std::max(maximum, chunk_capacity(std::min(per_call, width - begin)));
        }
        return maximum;
    };

    std::size_t maximum = 0;
    if (min_width <= kMaximumVerifyTokens) {
        const std::int32_t last = std::min(max_width, kMaximumVerifyTokens);
        for (std::int32_t width = min_width; width <= last; ++width) {
            maximum = std::max(maximum, exact_capacity(width));
        }
    }
    return maximum;
}

} // namespace ninfer::ops::detail
