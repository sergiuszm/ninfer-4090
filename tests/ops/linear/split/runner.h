#pragma once

#include "ops/linear/support.h"
#include "ops/quantized_weight.h"

namespace ninfer::test::linear_split {

using linear_suite::Suite;

struct WeightRecipe {
    QType type;
    int rows;
    std::uint32_t seed;
    quantized_weight::PatternedWeightOptions options{};
};

// A comparison region is contained in one output; its rows refer to concatenated parents.
struct Region {
    int first, rows, head_dim = 0;
};
enum class Columns { All, Seven, Projection, HeadBoundaries };
enum class Binding { Direct, ProjectionInputs, LinearInputs, Nvfp4Inputs };

struct Profile {
    int k;
    std::vector<WeightRecipe> weights;
    std::vector<int> outputs;
    std::vector<Region> regions;
    ReductionCriterion a16;
    double a4_absolute = 1.0 / 256.0;
    Binding binding    = Binding::Direct;
    std::vector<int> input_slices;
    // Fork-local (PR #15): the AllowA8 criterion. The default is upstream's FP8 A8 criterion,
    // which production also used for the attention Q4/Q5 INT8 route; the GDN Q4/Q5 form sets the
    // one production validated for it.
    ReductionCriterion a8{0.04, 1.0 / 256.0, 0.06};
};

struct Invocation {
    int t;
    ops::LinearPolicy policy   = ops::LinearPolicy::A16Only;
    bool graph                 = false;
    std::uint32_t input_seed   = 0; // Add T to this seed, independently of selection order.
    std::uint32_t changed_seed = 0; // Zero selects exact BF16 negation.
    Columns columns            = Columns::All;
    int sample_rows            = 7;
    bool full_rows             = false;
    bool convenience           = false;
    bool shared_output         = false;
    bool omit_divisors         = false;
    bool eager_before_capture  = false;
    bool validate_operands     = false;
    bool workspace_intervals   = false;
    bool reindex_invariance    = false;
    std::string id;
};

void add_profile(Suite& suite, const Profile& profile, std::vector<Invocation> calls);
void add_segmented_cases(Suite& suite);
void add_two_output_cases(Suite& suite);
void add_multi_output_cases(Suite& suite);

void add_bank_cases(Suite& suite);

} // namespace ninfer::test::linear_split
