#include "ops/linear/add/runner.h"
#include "ops/linear_common.h"

#include <array>
#include <vector>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::QType;

constexpr std::array<std::int32_t, 10> kGraphTokens{13, 40, 80, 112, 128, 192, 513, 545, 609, 705};

void q5_a16_conformance(ninfer::test::linear_suite::Suite& suite) {
    // Exercise the production transitions, including the same transitions in
    // a composite tail and the 192-token limit on that tail.
    constexpr std::array<std::int32_t, 13> starts{5,  9,   17,  25,  33,  49, 65,
                                                  97, 129, 161, 193, 257, 513};
    constexpr std::array<std::int32_t, 13> interiors{1,  2,   3,   7,   13,  20,  40,
                                                     80, 112, 144, 224, 768, 1024};
    constexpr std::array<std::int32_t, 0> no_starts{};
    constexpr std::array<std::int32_t, 7> full{8, 13, 32, 40, 80, 112, 192};
    for (const auto k : {6144, 17408}) {
        ninfer::test::linear_add::add_shape(suite, QType::Q5_G64_FP16,
                                            ShapeCase{.n                = 5120,
                                                      .k                = k,
                                                      .seed             = 401U,
                                                      .route_starts     = starts,
                                                      .route_interiors  = interiors,
                                                      .graph_tokens     = kGraphTokens,
                                                      .composite_offset = 512});
        ninfer::test::linear_add::add_shape(suite, QType::Q5_G64_FP16,
                                            ShapeCase{.n               = 5120,
                                                      .k               = k,
                                                      .seed            = 401U,
                                                      .route_starts    = no_starts,
                                                      .route_interiors = full,
                                                      .full_output     = true});
    }
}

// Fork-local (PR #15): the INT8 group-64 route under AllowA8. Masked tiles, ordinary prefill and
// the bounded-workspace token tile (4097 = one full 4096 tile plus one token), in increasing
// order so every call also checks width invariance against the previous one. The graph tokens of
// the A16 conformance are captured here under AllowA8; their AllowA4 captures there keep the A16
// route, so both routes have graph-replay coverage at those widths.
void q5_a8_conformance(ninfer::test::linear_suite::Suite& suite) {
    using ninfer::ops::LinearPolicy;
    using ninfer::test::linear_add::GraphReplay;
    using ninfer::test::linear_add::Invocation;
    using ninfer::test::linear_add::InvocationCase;
    for (const auto k : {6144, 17408}) {
        std::vector<Invocation> calls;
        for (const std::int32_t t : {1, 2, 8, 14, 17, 24, 33, 40, 49, 96, 129, 192, 256, 512, 1024,
                                     4097})
            calls.push_back({.tokens = t, .policy = LinearPolicy::AllowA8});
        for (const std::int32_t t : {40, 128, 513})
            calls.push_back({.tokens = t,
                             .policy = LinearPolicy::AllowA8,
                             .graph  = GraphReplay::OriginalNegative});
        for (const std::int32_t t : kGraphTokens)
            calls.push_back({.tokens = t,
                             .policy = LinearPolicy::AllowA8,
                             .graph  = GraphReplay::OriginalZeroOriginal});
        ninfer::test::linear_add::add_cases(suite, QType::Q5_G64_FP16,
                                            InvocationCase{.n           = 5120,
                                                           .k           = k,
                                                           .seed        = k == 6144 ? 401U : 409U,
                                                           .invocations = calls});
    }
}

} // namespace

void ninfer::test::linear_add::add_q5_cases(Suite& suite) {
    q5_a16_conformance(suite);
    q5_a8_conformance(suite);
}
