#pragma once

// Fork-local (PR #15): checks that each dense-body sizing site of a text profile reserves its
// INT8 activation staging under the int8 prefill rule, and nothing extra under the declared
// policies. Host only: the site functions query the Ops' capacity, they launch nothing. The
// expectations are those of the 27B body (paired Q4/Q5 projections, Q5 output adds, Q4 gate/up,
// Q5 down), where every lifted use has an INT8 route; a lifted use without one fails the check.

#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/dense_mlp.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/precision.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <variant>

namespace ninfer::test::qwen3_5_prefill {

namespace execution = models::qwen3_5::execution;

// The INT8 staging of one activation, restated from ops/linear/kernels/int8/int8_proj_launch.h:
// group-64 INT8 codes of the 256-padded K, then FP32 group scales, for one token tile of at most
// 4096 tokens.
inline std::size_t int8_staging_bytes(std::int32_t k, std::int32_t tokens) {
    const auto padded = (static_cast<std::size_t>(k) + 255) / 256 * 256;
    const auto tile   = static_cast<std::size_t>(std::min(tokens, 4096));
    const auto codes  = (padded * tile + 255) / 256 * 256;
    return codes + padded / 64 * tile * sizeof(float);
}

// Lifted sites per kind, so a caller can require that its profile exercises each one.
struct LiftedSites {
    int attention_projection = 0;
    int gdn_projection       = 0;
    int attention_output     = 0;
    int gdn_output           = 0;
    int dense_mlp            = 0;
};

struct SizingCheck {
    const execution::BodyActivations& int8;
    const execution::BodyActivations& declared;
    std::int32_t first;
    std::int32_t last;
    int failures = 0;

    void expect(bool condition, int layer, const char* site, std::size_t actual,
                std::size_t expected) {
        if (condition) return;
        std::cerr << "FAIL: layer " << layer << ' ' << site << " [" << first << ", " << last
                  << "] reserves " << actual << " bytes, expected " << expected << '\n';
        ++failures;
    }

    [[nodiscard]] std::size_t add_bytes(const ops::LinearProfile& p) const {
        return ops::linear_add_workspace_capacity_bytes(p.qtype, p.n, p.k, p.policy, first, last);
    }

    // A lifted paired projection and a lifted output add reserve exactly their staging; their
    // declared A16 routes need no workspace.
    template <class Mixer, class ProjectionBytes>
    void mixer(int layer, const Mixer& m, ProjectionBytes projection_bytes,
               const char* projection_site, const char* output_site, int& lifted_projection,
               int& lifted_output) {
        const execution::BodyActivations none;
        const auto none_bytes = projection_bytes(m, none);
        expect(projection_bytes(m, declared) == none_bytes, layer, projection_site,
               projection_bytes(m, declared), none_bytes);
        if (int8(m.projection) != m.projection) {
            ++lifted_projection;
            const auto staging = int8_staging_bytes(m.projection.first.k, last);
            expect(projection_bytes(m, int8) == staging && none_bytes == 0, layer,
                   projection_site, projection_bytes(m, int8), staging);
        }
        expect(add_bytes(declared(m.output)) == add_bytes(m.output), layer, output_site,
               add_bytes(declared(m.output)), add_bytes(m.output));
        if (int8(m.output) != m.output) {
            ++lifted_output;
            const auto staging = int8_staging_bytes(m.output.k, last);
            expect(add_bytes(int8(m.output)) == staging && add_bytes(m.output) == 0, layer,
                   output_site, add_bytes(int8(m.output)), staging);
        }
    }

    // A lifted dense MLP reserves its [gate_up / 2, T] BF16 product, then the larger of the
    // gate/up and down workspaces (dense_mlp_workspace_bytes).
    void dense(int layer, const execution::DenseProfile& profile, int& lifted_mlp) {
        const execution::BodyActivations none;
        const auto mlp_bytes = [&](const execution::BodyActivations& activations) {
            return execution::dense_mlp_workspace_bytes(profile, first, last,
                                                        execution::DenseMlpUse::Text, activations);
        };
        expect(mlp_bytes(declared) == mlp_bytes(none), layer, "dense MLP", mlp_bytes(declared),
               mlp_bytes(none));
        const auto& gu         = profile.gate_up;
        const auto& down       = profile.down;
        const bool gu_lifted   = int8(gu) != gu;
        const bool down_lifted = int8(down) != down;
        if (!gu_lifted && !down_lifted) return;
        ++lifted_mlp;
        const auto tokens   = static_cast<std::size_t>(last);
        const auto product  = (static_cast<std::size_t>(gu.n / 2) * tokens * 2 + 255) / 256 * 256;
        const auto gate_up  = gu_lifted ? int8_staging_bytes(gu.k, last)
                                        : ops::linear_swiglu_workspace_capacity_bytes(
                                             gu.qtype, gu.n, gu.k, gu.policy, first, last);
        const auto down_ws  = down_lifted ? int8_staging_bytes(down.k, last) : add_bytes(down);
        const auto expected = product + std::max(gate_up, down_ws);
        expect(mlp_bytes(int8) == expected, layer, "dense MLP", mlp_bytes(int8), expected);
    }
};

// Checks every text-body sizing site over [first, last]: under `int8` (a prefill pass with the
// int8 option) each lifted site reserves its INT8 staging; under `declared` (a16, or a pass that
// is not a prefill) every site is sized by its declared policies. Returns the failure count.
inline int check_text_sizing(const execution::TextProfile& text,
                             const execution::BodyActivations& int8,
                             const execution::BodyActivations& declared, std::int32_t first,
                             std::int32_t last, LiftedSites& lifted) {
    SizingCheck check{int8, declared, first, last};
    for (int layer = 0; layer < static_cast<int>(text.layers.size()); ++layer) {
        const auto& block = text.layers[static_cast<std::size_t>(layer)];
        if (const auto* attention = std::get_if<execution::AttentionProfile>(&block.mixer)) {
            check.mixer(
                layer, *attention,
                [&](const execution::AttentionProfile& p, const execution::BodyActivations& b) {
                    return execution::attention_projection_workspace_bytes(p, b, first, last);
                },
                "attention projection", "attention output", lifted.attention_projection,
                lifted.attention_output);
        } else {
            check.mixer(
                layer, std::get<execution::GdnProfile>(block.mixer),
                [&](const execution::GdnProfile& p, const execution::BodyActivations& b) {
                    return execution::gdn_projection_workspace_bytes(p, b, first, last);
                },
                "GDN projection", "GDN output", lifted.gdn_projection, lifted.gdn_output);
        }
        if (const auto* dense = std::get_if<execution::DenseProfile>(&block.ffn))
            check.dense(layer, *dense, lifted.dense_mlp);
    }
    return check.failures;
}

} // namespace ninfer::test::qwen3_5_prefill
