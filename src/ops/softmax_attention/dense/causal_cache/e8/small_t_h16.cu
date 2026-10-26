// Fork-local (rtx4090-port): small-T instances of the causal_cache/e8/ family for the
// 16/2 query/KV head geometry.
#include "ops/softmax_attention/dense/causal_cache/e8/small_t_launch.cuh"

namespace ninfer::ops::detail::e8_launch {

NINFER_E8_SMALL_T_INSTANCE(, e8::CausalD256H16Kv2, e8::CausalAppendInput);
NINFER_E8_SMALL_T_INSTANCE(, e8::CausalD256H16Kv2, e8::CausalCachedInput);

} // namespace ninfer::ops::detail::e8_launch
