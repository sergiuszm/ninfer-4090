# Sibling fork comparison: UDPSendToFailed/ninfer-4090

Date: 2026-08-17. Their branch: `feat/rtx-4090-sm89-native` at v0.9.0 (`8426c45a`).
Ours: `rtx4090-port` at `d78df936`.

Both forks descend from `Don-Chad/ninfer-3090` (merge base `c55c23bb`) and target
Qwen3.8-27B on a single RTX 4090. The work is largely complementary:

- **Ours**: attention-prefill retune for sm_89 (fp16-acc PV tiles, +30% kernel
  roofline), serve hardening (vision caps, 413 masking, swscale overflow,
  retained-slot `/slots`, `/metrics`, `context_window`), depth-sweep benchmarks.
- **Theirs**: KV-cache quantization (Hadamard-rotated 4/8-bit, E8 lattice 2/4-bit
  key codebooks), prompt-lookup draft speculation, L2 persistence for MTP weights,
  context envelope lift to 1M, Windows build support.

## Serve fix exchange

Their PR #1 describes the OpenAI tool-message content-part array bug (tool role
hard-required string content; agent clients such as Qwen Code send part arrays and
got 400). The same bug existed here; fixed in `d78df936` and deployed. Cross-link
posted on their PR.

## KV quantization evaluation (their branch, our 4090)

Method: their branch built unmodified for sm_89 (`-DCMAKE_CUDA_ARCHITECTURES=89`),
run against the official 16.96 GiB artifact on the same card that serves our
production config. NIAH: filler haystack, passphrase needle at a depth fraction,
exact-match answer, `temperature 0`, `enable_thinking false`; prefill rate
approximated as `prompt_tokens / wall` with `max_tokens 24`. Decode: greedy
512-token Go-code generation from a 45-token prompt (shallow), their
`--spec mtp --draft-tokens 4 --lm-head-draft`.

| Config | Max ctx loaded | KV runtime | NIAH (tokens@depth) | Prefill t/s | Decode t/s |
|---|---:|---:|---|---:|---:|
| their `int8` (control) | 96k | - | 59k@0.35 PASS | 1802 @59k | 139.2 |
| their `rk4v4-e8` | 200k | 4.92 GiB | 59k@0.35, 118k@0.8, 189k@0.35 all PASS | 1799 / 1499 / 1251 | 133.3 |
| their `rk2v4-e8` | 330k | 5.83 GiB | 118k@0.8, 314k@0.35 all PASS | 1404 @118k / 867 @314k | 130.3 |
| ours `int8` (deployed, reference) | 172k | - | exact 64k/128k | 1724 @88k / 1561 @128k | 148.6 (MTP3) / 152.6 (MTP4) |

Observations:

- **314k tokens on 24 GB with exact single-needle retrieval.** The 2-bit E8
  cylinder key codebook survives the easy quality bar at extreme depth.
- E8 modes cost ~4-6% decode vs int8 on their branch (139.2 -> 133.3 -> 130.3).
- At 59k depth, prefill is identical across int8 and rk4v4-e8 (1802 vs 1799) on
  their branch, and within ~5% of our retuned int8 at ~118k (1499 vs our 1561 at
  128k) despite their branch lacking the fp16-acc PV retune - the halved K
  traffic in attention roughly offsets it. Combining their KV modes with our
  retune should therefore beat our current depth prefill.
- Their decode is ~7-9% below ours at matched int8 (139.2 vs 148.6-152.6),
  consistent with their README ranges; their prompt-lookup only fires when MTP
  returns zero drafts, so it does not move MTP-on numbers.
- Runtime-per-token from load logs: rk4v4-e8 ~26.4 KB/token, rk2v4-e8
  ~19.0 KB/token, vs our measured int8 slope ~35.9 KB/token. Projected ceilings
  on our deployment: ~234k (rk4v4-e8) and ~325k (rk2v4-e8; needs their
  `kGqaAttentionMaximumVisibleKeys` lift `1da9ed5a` above 262k).

## Port assessment

Worth porting (in order):

1. `rk4v4-e8` (and the rk4v4/rk8v4 plumbing it builds on): commits `0a39efe3`,
   `0701b973`, `46116bf2`, `e4616151`, `9e774969`. The codec is isolated in
   `e8_lattice.cuh` + `e8_root_codec.cuh`; the integration touches
   `gqa_attention_prefill_i8.cuh` / `gqa_attention_decode_i8.cuh`, which conflict
   with our sm_89 retune (`ce50e995`) - the merge must re-apply the retune's
   maxnreg/producer-warp/PRMT/fp16-acc structure around their dequant hooks.
2. Context envelope lift `1da9ed5a` (only needed above 262k; small, clean).
3. Skip: prompt-lookup stack (`aee631c1`, `fc65ff96`, `bdf4a90b`), L2 policy
   (`3482bc46`), W8 double-buffer (`7c5970b2`) - measured decode with all of them
   is below our current numbers; revisit only for MTP-off serving.

Open questions before switching production KV dtype:

- Single-needle NIAH is a weak bar. Run multi-needle and code-repair-at-depth
  probes before trusting 2-bit keys for agent traffic; their README cosine
  similarities (98.7% rk4v4-e8, 96.2% rk2v4-e8) imply measurable degradation.
- MTP acceptance under E8 KV at depth is unmeasured (shallow decode only).
- Their bench matrix was Windows/CUDA 13.3; our numbers here are the Linux
  container stack.

## Port results (2026-08-17)

All six KV commits are merged onto `rtx4090-port` (`7c2ce91e..ec56f922` plus the
`2619adf7` request-log name fix). Merge notes:

- Their mode machinery never overlaps the `ce50e995` retune except at kernel
  signatures: fill kernels, staging lambda, and Q-quant loop hooks all landed
  automatically; every hook site was diffed against their tip. The ops layer is
  byte-identical to their branch except `gqa_attention_prefill_i8.cuh` (retune +
  modes interleaved) and the restored `int8-group64` request-log name.
- Their 1 GiB CUDA-graph allowance bump (`layouts_impl.h`) was NOT taken: with our
  allowances the INT8 172032 profile still loads (136 MiB slack); with theirs it
  cannot. No graph-allowance errors appeared in any E8 run.
- Upstream bug found: their `kv_cache_name` rename to "int8" breaks
  `test_request_log`, invisible on their side because the Windows flow never runs
  ctest. `ninfer_test_e8_codec` builds but is not registered with ctest.
- `ninfer_state_store_test` failed once under `ctest -j8` (GPU contention with the
  new codec test); serial and isolated runs pass 84/84 consistently.

Measured on the merged branch (same probes as the pre-port eval, same card):

| Mode | Ctx loaded | KV runtime / slack | NIAH | 5-needle @118k | Code detail @168k | Prefill t/s | Decode | MTP acc @111k |
|---|---:|---|---|---|---|---|---:|---:|
| `int8` | 172032 | 6.31 / 0.13 GiB | 59k, 118k pass | - | - | 1869 @59k / 1604 @118k | 134.2 | 78.4% |
| `rk4v4-e8` | 262144 | 5.08 / 1.37 GiB | 59k-260k all pass | 5/5 | 3/3 | 1846 / 1575 / 1347 @189k / 1172 @260k | 126.6 | 78.8% |
| `rk2v4-e8` | 262144 | 4.01 / 2.43 GiB | 260k pass | 5/5 | 3/3 | 1059 @260k | 120.5 | - |

The retune carries into E8: at matched depth the merged branch beats their branch by
+2.6% @59k growing to +7.7% @189k on rk4v4-e8 prefill. E8 keys cost nothing in MTP
acceptance at depth. **Deployed config since 2026-08-17: `rk4v4-e8` at the full
native 262,144 context** - the 24 GB card now serves the same context window as the
32 GB 5090. `rk2v4-e8` adds slack, not context (262,144 is the model's own limit);
it stays available for a future vision-plus-long-context profile.

## Second wave (their 2026-08-18 evening push, assessed 2026-08-18)

Sixteen commits (`0a925796..6d3fd165`), all with claimed bit-exact parity and a
green 84/84 suite on their side. Disposition per group:

- **`--vision-max-tokens` (6d3fd165): ported.** The vision scratchpad drops from a
  hardcoded 32768 tokens to a configurable default of 8192 and frees about 1.5 GiB.
  Cherry-picked clean. Their commit leaves the processor budget
  (`max_vision_tokens`, still 32768) out of sync with the shrunken workspace: a
  request with 8K-32K image tokens passes the budget check and reaches the
  undersized encoder. This fork wires the budget to the same limit
  (`fix(frontend)` follow-up commit), so the failure is a clean
  `media_budget_exceeded`. With the port, `rk4v4-e8` serves the full native
  262,144 context with `--vision` at 780 MiB slack - the 208K practical line and
  the vision-against-context tradeoff are gone. Correction (2026-09-23): upstream
  `fc5c4834` (2026-08-24) later made the tower encode one item at a time, so the
  limit now applies to each item and the aggregate budget is back at upstream's
  32768. The aggregate wiring had rejected agent conversations after about ten
  screenshots, because clients send every earlier image again with each turn.
- **CUDA-graph allowance tightening (c85db47a): skipped.** They replace their old
  1 GiB SM86/SM89 per-lane padding with flat 64 MiB (ordinary) / 256-320 MiB (MTP)
  allowances. This branch already carries the per-topology-class accounting, which
  measures 8 MiB / 86 MiB for the same profiles - tighter than their new flat
  values. Their commit is a catch-up, not a win.
- **Causal-tile partitioned prefill attention (c5f70526): ported 2026-08-19
  (commit 694e01f0), re-implemented inside the retuned kernel.** Our baseline
  already skipped masking on interior tiles (`full_score_tile`), so their
  headline could not transfer whole; what did transfer is the structural split:
  a FullTile-tagged instantiation of the key-block body with unconditional
  staging, no masking, and no softmax zero-selects, dispatched per block from a
  precomputed `n_full_blocks`. Their bf16 kernel and common-header changes were
  taken verbatim (identical base). Measured on the INT8 `d256-h24-kv4` append
  shape at W=1024: 144 to 165 TFLOP/s at 32K-224K context (-12 to -13%
  latency), 200 to 171 us shallow, registers stay at 128 with no spills,
  84/84 tests bit-exact. End-to-end serve prefill: +1.1% at 51K on INT8 KV
  (attention wall share of the hybrid-GDN model is only ~10% there and the
  share grows with depth), within noise on the deployed `rk4v4-e8` mode - E8
  staging time is lattice-decode compute, not the removed guards. The biggest
  serve-level beneficiary is the INT8-KV 5090 deployment; porting this to
  ninfer-5090 is queued.
- **Q4/Q5/Q6/W8 dequantization micro-optimizations (73f3d7be, 8f298555, b8ddda48,
  d9d701bc): rejected on measurement (2026-08-19).** All four cherry-pick clean
  and pass the 84-test suite, but on this Linux CUDA 13.1.2 `sm_89` build they
  regress the Q4/Q5 MMA rowsplit path hard. Measured medians at T=1024
  (`ninfer_q5_linear_add_bench --k 17408`, `ninfer_q4_linear_swiglu_bench`,
  suite shapes):

  | State | Q5 down k=17408 | Fused Q4 swiglu |
  |---|---:|---:|
  | pre-wave baseline | 1494.9 us (122.1 TFLOP/s) | 3166.2 us (115.3 TFLOP/s) |
  | + 73f3d7be (Q4/Q5 hoist) | 1631.2 us (-9.1%) | 3436.5 us (-8.5%) |
  | + 8f298555, b8ddda48 | 1618.9 us (flat) | 3394.6 us (flat) |
  | + d9d701bc (shuffle) | 2281.5 us (**-52.6%**) | 3123.1 us (+1.4% net) |

  The suite confirms the pattern on production shapes: `gdn_output_gate` Q5
  -38 to -49%, `draft_head` Q4 -5 to -9%, vision Q4/Q5 projections -13 to -54%,
  Q6/W8 and every small-T SIMT path neutral. The only net winner is the fused
  swiglu at +1.4% (~0.5% end-to-end), and it is inseparable from the losses
  without splitting the shared decode atoms per kernel. Suspected cause of the
  mismatch with their results: their WDDM and MSVC commits indicate a Windows
  toolchain, and the shuffle and `bfe.s32` patterns compile to different SASS
  there. The picks were dropped from this branch after the bisect.
- **GDN / conv1d / 2D-memcpy decode-tail work (52fc4aec, fa629318, b860bd5f,
  fa767237, d520f7bf, cf586d09, 4d79043e): low priority.** Decode on this card
  measures bandwidth-saturated end to end; expected gain is 1-3%.
- **Windows WDDM/D3D12 residency and MSVC flags (a35acf6a, aa8a1c98): not
  applicable.** All `_WIN32`-guarded.

## Third wave (their 2026-08-19 push, assessed 2026-08-20)

Eight commits (`5301c093..ed29978d`), their branch now tagged `v1.0.0-rtx4090`.
Ours is `rtx4090-port` at `c4d09b61`. Disposition per group:

- **E8 codec hardening (`b01692c6`): ported as `bc569eb8`.** The author is Daniel
  Parker, not the fork owner. The commit originates in upstream PR #35 and was
  taken back into their tree, so it arrives here third-hand. Cherry-picked with
  `-x` and the original authorship preserved. The defect is real: the early
  return on `rad_idx == 0` in `e8_encode_cylinder_8d_warp` leaves some 8-lane
  subgroups out of full-mask `__shfl*_sync` calls, which is undefined behaviour.
  The damage is latent here. That function is called only under the `E8Root`
  template flag, `kv_e8_root` is true only for `RK2V4E8`, and this box serves
  `rk4v4-e8`, which sets `kv_e8_lattice` instead. Measured on sm_89 with CUDA
  13.1.2, the pre-fix codec passes the new contract test and
  `compute-sanitizer --tool synccheck` reports no error, because the shuffles use
  XOR masks {1,2,4} that never cross an 8-lane boundary. A subgroup that returned
  early holds nothing its neighbours read. The same commit makes
  `verify_1m_retrieval` return `EXIT_FAILURE` on a missed needle instead of
  always exiting 0. `4fe6f7a4` (unused `kInvSqrt2`) followed as `a0e03d37`.
- **Q4/Q5 dequantization revert (`b0e926aa`): nothing to take, and it confirms
  the second-wave bisect.** They restore the direct shared-memory loads in the
  Q4/Q5 rowsplit GEMM and drop the warp-shuffle scale distribution, which is the
  commit this branch measured at -52.6% on Q5 down `k=17408`. What survives in
  their tree is the `bfe.s32` extraction and the `cta_row_base` address hoisting.
  Their headline of 1865 to 1929 tok/s at pp4096 is measured against their own
  regressed baseline, so it implies no gain over this branch.
- **GDN uniform loads and reductions (`ed29978d`): one hunk of three applies.**
  The `normalize_qk_lane` hunk was ported as `6e239351`. The `load_value_pack`
  hunk does not apply, because this tree has no such function. See the layout
  note below. The `apply_gdn_transition` `fmaf` hunk is a no-op, because nvcc
  contracts that expression by default.
- **State pool telemetry (`d9ce4436`): open.** A per-pool breakdown of the
  persistent arena (text KV, MTP KV, GDN state, DFlash KV, replay records) logged
  at serve start and in the CLI summary. This is useful for the VRAM ledger. It
  touches `program.h` and `program_impl.h`, which this branch has changed
  heavily, so the merge cost is real for a diagnostic-only gain.
- **KV cache test synchronization (`42f24afa`): not applicable.** The added
  `cudaDeviceSynchronize` calls address WDDM background DMA on Windows. This is
  not the `state_store` flake seen here under `ctest -j8`.
- **MSVC and PTX build flags (`60ff23d5`), README disclaimer (`7b686697`): not
  applicable.** The build changes sit entirely inside `if(MSVC)`.

### GDN value layout: why their fold gain cannot transfer

Their headline for `ed29978d` is 12.5 us off `recurrent_fold_kernel`. That gain
belongs to `load_value_pack`, and the two forks hold GDN values differently.
Their layout replicates all four `dv` values in every lane, so the pre-change
loader had lane 0 read a 64-bit pack and broadcast it with two shuffles. Their
commit removes that broadcast. This tree used `RawValueLane`: lane *i* held value
*i* under a `lane < kDvPerWarp` guard, with no shuffle at load time and 28 of 32
lanes idle. The broadcast was paid one level down instead, as a `__shfl_sync` per
`r` inside `apply_gdn_transition`.

Commit `c4d09b61` applies the idea in the form this tree needs. It is our own
change, not a port. The loader now reads a `Bf16x4Pack` uniformly in every lane
and relies on the L1 broadcast, the transition reads `v[r]` from registers, and
`store_value` writes one vector store from lane 0 rather than four scalar stores
from four lanes. The value path now matches the key path, which already used
`load_vec` and `store_vec`. SASS confirms the intent: 59 to 55 `SHFL` in the
affected kernels.

Both GDN changes are bit-exact and both measure near zero:

| Change | replay bench, all | fold | record | snapshot |
|---|---:|---:|---:|---:|
| `6e239351` XOR butterfly | +0.07% | +0.10% | -2.34% | -0.33% |
| `c4d09b61` uniform value pack | -0.09% | -0.08% | -0.45% | -0.29% |

Method: `ninfer_gdn_replay_bench --profile 27b --repeat 200`, `gpu_warm` medians
of three interleaved A/B rounds. For `c4d09b61`,
`ninfer_gated_delta_net_bench --snapshot --sweep` over five rounds shows -7.2% at
T=1, -7.7% at T=2, -3.3% at T=4 and -3.7% at T=5, then flat from T=8. That is
0.7 to 1.0 us per launch where the kernel is latency bound. The `--running`
sweep is flat. Bit-exactness was verified directly, not argued: 0 mismatches over
6.4M lanes for the butterfly reduction, and 0 mismatches over all 65536 bf16
patterns for the scalar-to-pair conversion.

The conclusion for future waves is that these paths are memory bound, not shuffle
bound. Two independent shuffle removals each moved the aggregate by about 0.1%.
The remaining GDN decode-tail commits in their tree are the same class of change.

### E8 test coverage gap (our own work, `94830b3f`)

The E8 codec shipped with no coverage. `ninfer_test_e8_codec` was built but never
registered with `add_test`, so ctest ignored it, and its verifier returned 0
unconditionally. Registering it is not enough: `tools/test_kv/test_e8_codec.cuh`
carries its own reference implementation of the E8 mathematics and never includes
`src/ops/kernel/e8_root_codec.cuh`, so no test compiled a line of the shipped
header. `tests/ops/test_e8_root_codec.cu` now drives it. The suite is 87 tests.

Writing that test surfaced a divergence between the two shipped encoders. When
the primary root is Type A it carries +-1 in exactly two coordinates, so the
residual is exactly symmetric across that pair and `|res|` ties to the last bit.
The scalar encoder keeps the first maximum and the warp argmax keeps the other,
which produces a different axis nibble for the same input. This is inert: only
`e8_encode_cylinder_8d_warp` writes the cache, and the scalar path reaches
production through no call site, because its only caller
`e8_encode_root_2stage_8d` is itself uncalled. The parity test excludes the axis
nibble and documents the tie. Normalising it would rewrite stored `rk2v4-e8`
codes.

## Upstream network (sweep of 2026-08-20)

Only the sibling fork moved in engine code. Upstream `Neroued/ninfer` added two
commits, both evaluation-only (a Qwen3.8 groupwise-int reasoning suite and its
published results). `Don-Chad/ninfer-3090` is unchanged since 2026-08-18,
`jram4/ninfer-4090` is still frozen at 2026-08-02, and
`shantanusingh16/ninfer-4090` is unchanged since the `timings` block was taken.

Tracing the author of `b01692c6` found a source worth tracking:
**`Neroued/ninfer` PR #35**, by `danielfparkernz`, open, titled *compressed-KV
cache (E8 lattice) for the Blackwell sm_120a path*. It ports the same E8
machinery this fork carries onto the official sm_120a tree, which is the base of
`ninfer-5090`. Six commits, 26 files, +2717/-125, on upstream master
`32c9881b`. The 5090 base `90059874` is an ancestor of that base. Nine of the 26
files overlap files changed on `nuntius-serve`. The branch is fetched in the
ninfer-5090 checkout as remote `dparker`.

PR #35 reopens the question settled in the port ledger, which declined E8 for the
5090 because 32 GB fits the full 262,144 context on `int8`. That reasoning holds
for the groupwise-int artifact. It does not hold for NVFP4. A validator on the PR
(`tiequan12345`, RTX 5090 32 GB) reports that the NVFP4 artifact cannot reach
262,144 with INT8 KV, because the runtime reservation is 13.02 GiB against
11.66 GiB available after weights. With `rk8v4` it boots at 31.15 GiB with MTP
k=3 and vision. Auto capacity still fails, because its 1 GiB headroom lands about
150 MiB short, so an explicit `--kv-capacity 262144` is required. Retrieval and a
paired quality battery are a statistical tie against groupwise-int with INT8 KV.
Note that his speed figures compare two configurations at once, weights and KV
together, so they are not a KV-only measurement.

The recommendation is to not hand-port PR #35. It is an unmerged branch that can
still change, `nuntius-serve` is 37 commits behind upstream master in any case,
and a merge delivers the sm_120a integration at no cost. Watch it, and decide on
E8 for the 5090 only if NVFP4 at full context becomes the goal.

Upstream has also gained a large volume of community contributions, several of
which duplicate work already carried here: PR #57 is the tool-message
content-part fix (`d78df936`), PR #24 advertises the model context in the models
API (`0f308358`), issue #49 asks for rewrite checkpoints that survive lane
eviction (the turn-checkpoint ring and auto-save on eviction), and issue #32 asks
whether attention accumulate is the prefill ceiling above 64K (answered by the
fp16-accumulate PV measurements). These are candidates to offer upstream, not
things to take.
