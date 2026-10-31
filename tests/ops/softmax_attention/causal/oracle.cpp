#include "fixture.h"

namespace ninfer::test::causal_attention {
namespace {

// Applies the normalized Sylvester H64 (symmetric and self-inverse) to every 64-dimension group
// of one head row, in FP64.
void normalized_hadamard_g64(std::vector<double>& values, std::size_t row_base) {
    for (std::int32_t group = 0; group < kQuantGroups; ++group) {
        const std::size_t base = row_base + static_cast<std::size_t>(group * kQuantGroup);
        std::array<double, kQuantGroup> input{};
        for (int i = 0; i < kQuantGroup; ++i) input[i] = values[base + i];
        for (unsigned row = 0; row < kQuantGroup; ++row) {
            double sum = 0.0;
            for (unsigned column = 0; column < kQuantGroup; ++column)
                sum += ((std::popcount(row & column) & 1) ? -0.125 : 0.125) * input[column];
            values[base + row] = sum;
        }
    }
}

} // namespace

std::vector<double> ideal_attention(const std::vector<float>& q, const HostCache& cache,
                                    const std::vector<std::int32_t>& positions) {
    const Geometry& geometry = cache.geometry;
    const int tokens = positions.size(), visible = positions.back() + 1;
    // Fork-local rk4v4-e8 rotates each 64-dimension group (H64) instead of the whole d256 row.
    const bool group_rotation = cache.storage == KvCacheStorage::RK4V4E8;
    const bool rotate_q       = cache.storage != KvCacheStorage::BFloat16 && !group_rotation;
    const bool rotate_v       = cache.storage == KvCacheStorage::Nvfp4Group16 ||
                                cache.storage == KvCacheStorage::Fp8KeyNvfp4Value;
    std::vector<double> query(q.begin(), q.end()), output(q.size());
    if (group_rotation)
        for (int token = 0; token < tokens; ++token)
            for (int head = 0; head < geometry.q_heads; ++head)
                normalized_hadamard_g64(query, q_index(geometry, head, 0, token));
    if (rotate_q)
        for (int token = 0; token < tokens; ++token)
            for (int head = 0; head < geometry.q_heads; ++head) {
                std::array<double, kHeadDim> row{};
                for (int d = 0; d < kHeadDim; ++d) row[d] = q[q_index(geometry, head, d, token)];
                normalized_hadamard_d256(row);
                for (int d = 0; d < kHeadDim; ++d)
                    query[q_index(geometry, head, d, token)] = row[d];
            }
    // Decode the persistent public representation once. No private Q quantization, staging
    // casts, partial rounding or inverse-transform materialization enters this oracle.
    const auto index = [&](int d, int head, int pos) {
        return std::size_t(d) + kHeadDim * (std::size_t(pos) + std::size_t(visible) * head);
    };
    std::vector<double> keys(std::size_t(visible) * geometry.kv_heads * kHeadDim),
        values(keys.size());
    for (int head = 0; head < geometry.kv_heads; ++head)
        for (int pos = 0; pos < visible; ++pos)
            for (int d = 0; d < kHeadDim; ++d) {
                keys[index(d, head, pos)]   = cache_value(cache, true, head, pos, d);
                values[index(d, head, pos)] = cache_value(cache, false, head, pos, d);
            }
    const auto products = std::int64_t(tokens) * geometry.q_heads * visible * kHeadDim;
    naive_dense_softmax_attention(
        op_geometry(geometry), tokens, visible, double(kAttentionScale),
        [&](int d, int head, int token) { return query[q_index(geometry, head, d, token)]; },
        [&](int d, int head, int pos) { return keys[index(d, head, pos)]; },
        [&](int d, int head, int pos) { return values[index(d, head, pos)]; },
        [&](int token, int pos) { return pos <= positions[token]; },
        [&](int d, int head, int token, double value) {
            output[q_index(geometry, head, d, token)] = value;
        },
        products >= (1 << 21));
    if (group_rotation)
        for (int token = 0; token < tokens; ++token)
            for (int head = 0; head < geometry.q_heads; ++head)
                normalized_hadamard_g64(output, q_index(geometry, head, 0, token));
    if (rotate_v)
        for (int token = 0; token < tokens; ++token)
            for (int head = 0; head < geometry.q_heads; ++head) {
                std::array<double, kHeadDim> row{};
                for (int d = 0; d < kHeadDim; ++d)
                    row[d] = output[q_index(geometry, head, d, token)];
                normalized_hadamard_d256(row);
                for (int d = 0; d < kHeadDim; ++d)
                    output[q_index(geometry, head, d, token)] = row[d];
            }
    return output;
}

ReductionCriterion attention_criterion(KvCacheStorage storage) {
    if (storage == KvCacheStorage::BFloat16) return kAttentionBf16Criterion;
    if (storage == KvCacheStorage::Int8Group64) return kAttentionInt8Criterion;
    if (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) return kAttentionFp8Criterion;
    if (storage == KvCacheStorage::Nvfp4Group16) return kAttentionNvfp4Criterion;
    if (storage == KvCacheStorage::Fp8KeyNvfp4Value) return kAttentionK8V4Criterion;
    // rk4v4-e8 reaches the same int8 tensor-core arithmetic as int8-g64: the stored K/V codes
    // are decoded exactly and only the private Q int8 quantization and FP16 P/V staging remain.
    // On the 267 cases both storages ran on catchup4, its relative L2 error was 1.01x int8's
    // (median, 1.20x max), so it shares the int8 criterion.
    if (storage == KvCacheStorage::RK4V4E8) return kAttentionInt8Criterion;
    throw std::logic_error("unregistered causal-attention test storage");
}

int verify_attention(const std::string& label, const std::vector<double>& actual,
                     const std::vector<double>& reference, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), actual, reference, criterion);
}

} // namespace ninfer::test::causal_attention
