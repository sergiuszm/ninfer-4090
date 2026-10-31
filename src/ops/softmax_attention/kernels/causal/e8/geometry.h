#pragma once

// Fork-local (rtx4090-port): head geometries of the causal_cache/e8/ family. They add the small-T
// split scale to upstream's head mapping, so the family keeps its own names in ops::e8.

#include "ops/softmax_attention/kernels/causal/geometry.h"

namespace ninfer::ops::e8 {

template <int QHeadsValue, int KVHeadsValue, int SmallTSplitScaleValue>
struct CausalAttentionGeometry : detail::AttentionHeadMapping<QHeadsValue, KVHeadsValue> {
    static_assert(SmallTSplitScaleValue > 0);

    static constexpr int SmallTSplitScale    = SmallTSplitScaleValue;
    static constexpr int SmallTMaximumSplits = 85 * SmallTSplitScale;
};

using CausalD256H24Kv4 = CausalAttentionGeometry<24, 4, 1>;
using CausalD256H16Kv2 = CausalAttentionGeometry<16, 2, 2>;

} // namespace ninfer::ops::e8
