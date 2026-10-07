# Port ledger

The RTX 4090 fork (this repository, branch `rtx4090-port`) and the RTX 5090 fork
([sergiuszm/ninfer-5090](https://github.com/sergiuszm/ninfer-5090), branch
`nuntius-serve`) share almost all of their engine and serve code. Features are
born in one tree and cherry-picked into the other. This ledger records, per
feature, the commit hash in each tree, so coverage stays checkable without
archaeology.

Maintenance rules:

- Cherry-pick with `git cherry-pick -x`, so the destination commit records its
  source hash. Each repository holds the other as a local remote
  (`local-5090` here, `local-4090` there).
- Port in the session that ships the feature. Delayed ports pay a growing
  adaptation cost; `f640b404` on the 5090 side is the receipt.
- When a feature is deliberately not ported, record the decision here instead
  of leaving a silent gap.

## Feature rows

| Feature | 4090 (`rtx4090-port`) | 5090 (`nuntius-serve`) | Notes |
|---|---|---|---|
| `context_window` in `/v1/models` | `0f308358` | `ed606ed5` | |
| Prometheus `/metrics` | (commit series) | `fc582982`, `5fa5ffa1` | |
| Retained depth on idle `/slots` | `b0893e79` | `1a9640f1` | |
| Vision modality in `/v1/models` | `b6172f24` | `61013cf1` | Born on the 5090 side |
| fp16-accumulate PV tiles | part of `ce50e995` | `4483c820` | 4090 folds it into the sm_89 retune |
| 413 body fix + media prompt cap | `85f685a3`, `bde2765c` | `66423552`, `e9093c77` | Born on the 5090 side |
| sws_scale stride pad | `5a08683d` | `060bb320` | Born on the 5090 side |
| Tool content-part arrays | `d78df936` | `b0a0a6fe` | |
| llama.cpp-compatible `timings` | `0f95b32e` | `0834b6cb` | From the shantanusingh16 fork |
| Final-chunk usage param fix | `265011d9` | `51857983` | |
| Slot save/restore to disk | `beaeb70a` | `aeaf3f28` | 5090 needed `f640b404` (KV modes) |
| Session digests + `if_digest` | `8e478945` | `40504615` | |
| Cheapest-lane reuse tie-break | `1614ef54` | `59cb1afb` | |
| `/slots` snapshot publishing | `4a4fa92b` | `93fadf94` | |
| Live llamacpp `/metrics` counters | `656b0df7` | `b38ae92d` | |
| Turn checkpoint ring | `3a2e7f07`, `cba2c1f8`, `2cbe488d`, `3419fe43` | `6826b8b0`, `1ac61caf`, `8b28e502`, `66401303` | Picked with `-x` |
| Auto-save on eviction | `8093c640` | `cad0218e` | Picked with `-x` |
| Causal-tile key-block partition | `694e01f0` | `b5179823` | i8 body re-applied per schedule; bf16 and common taken verbatim |
| E8 codec hardening | `bc569eb8`, `a0e03d37` | not applicable | The 5090 tree carries no E8 code. Third-hand from upstream PR #35 through the sibling fork; authorship preserved |
| Production E8 codec test | `94830b3f` | not applicable | Same reason. Also registers the standalone oracle, which ctest had never run |
| GDN QK norm XOR butterfly | `6e239351` | open | Bit-exact over 6.4M lanes, measures near zero. Port is cheap; value is consistency, not throughput |
| GDN uniform value pack | `c4d09b61` | open | Bit-exact over all 65536 bf16 patterns, measures near zero. Born here, not a port |

## Inbound ports from downstream forks

Nine forks of `sergiuszm/ninfer-4090` now exist. This table records what each
one contributed and what was declined, so that a later sweep does not
re-examine the same commits. Survey date: 2026-08-29.

| Source commit | Here | Decision |
|---|---|---|
| jomcgi `a0d78215`, `chat_template_kwargs` aliases | `6affed2e` | Ported. llama.cpp and vLLM spell the Qwen thinking controls under `chat_template_kwargs`. Existing clients reach the effort knob without a change |
| Don-Chad `db076d67`, remove CUDA forward-compat libraries | `ff925039` | Ported. Our `Dockerfile` uses the same `nvidia/cuda:13.1.2` base and carried the same latent failure. The deployed `ninfer-dev:runtime` image is built by hand, so no rebuild is forced |
| Don-Chad `ccb20680`, qualify Qwen3.8 on SM89 | not applicable | `layouts_impl.h` already gates on `device.sm() != 89`. The upstream form admits 86 or 89 and would loosen our gate |
| jomcgi `1513de5a`, ghcr image workflow | declined | Hardwired to `ghcr.io/jomcgi` and to a `runAsNonRoot` cluster policy. We deploy hand-built local images |
| Don-Chad `7afc8e17`, resident-CTA budget from the runtime SM count | open | Not a cherry-pick. See the note below |

The `7afc8e17` principle applies to us, but its constants do not. That fix
separates per-SM occupancy from the device-wide budget for sm_86, where the
supported range is 82 to 84 SMs. Our `bf16_gdn_gating_proj_plan.cpp` hardcodes
the 128-SM budget of the RTX 4090 and feeds the same constants to a
`static_assert`. The number is correct for the RTX 4090 and wrong for every
other Ada device: the RTX 4080 has 76 SMs and the L40S has 142. On a card with
fewer SMs the budget is overstated. A cooperative launch that does not fit is
then accepted, and the driver rejects it with
`cudaErrorCooperativeLaunchTooLarge`. A port needs `DeviceContext::sm_count()`,
our own per-SM occupancy figures, and a `kMinSupportedSmCount` value for sm_89.

## Inbound sweep 2026-09-01 (UDP fork)

`udp/feat/rtx-4090-sm89-native` moved from `8bba5eb4` to `717479fe`: 25 commits,
almost all dated 2026-09-01. Four are correctness fixes; the rest are sm_89 kernel
and build tuning. Triage of the four, checked against this tree rather than read
from their messages:

| Source commit | Applies here | Decision |
|---|---|---|
| `05a88712`, transient admission shortfall bricks the executor | class real, trigger blocked here | CLOSED 2026-09-01 after a GPU-window repro attempt: three interleaved conversations deepening 49k->58k tokens, 30 requests, a save/restore loop at 0.5 s beside them. No latch, `/health` 200 throughout, and the only failures were `classification=timeout` (the graceful pending-deadline path). The external half of their trigger cannot occur here - every concurrent slot save/restore was refused `409 slot_busy`, because this catalog serialises slot operations against open resource transactions. The in-engine half DID occur: auto-save-on-eviction spilled 1.2-1.4 GB snapshots concurrently with admission seven times without ill effect. The failure CLASS stays real (`fail_all_locked` latches permanently; the worker catch-all converts any admission `logic_error` into it), which is why `dd5206f0` was worth porting on its own. |
| `dd5206f0`, `/health` reports the executor's real state | yes | PORTED as `60764d66` (adaptation, not a cherry-pick - their executor and service layers have diverged). `healthy()` on both cores, forwarded through Engine and GenerationService; 503 + `{"status":"error"}` when the engine has latched. NOT YET DEPLOYED: the running 8086 binary still answers a hardcoded ok |
| `8488278c`, publish snapshot saves whose write already finished | no | Not applicable. `src/core/disk_state_cache.*` exists only in the UDP tree - neither here nor in `neroued/master` |
| `e2556b50`, render mid-conversation system turns in place | no | Already covered by a different implementation. Our template folds only `messages[0]` (`chat_template.cpp:464-477`) and renders later instruction turns in place in the message loop, so the shape that threw for them returns 200 here - verified against the live 8086 server. Their fix also edits `anthropic_schema.cpp`, a file the upstream Anthropic rework replaced in our merge |

The ~15 perf commits are subject to the standing rule from `docs/udp-fork-comparison.md`:
kernel-bench before any perf pick, because their dequant micro-optimisations lost on
measurement here. Start with `45a5ae57` ("size CTA waves from the target SM count, not
an RTX 5090"): it may be the sm_89 form of the Don-Chad `7afc8e17` row still open above,
which needs `DeviceContext::sm_count()`, our own per-SM occupancy figures and a
`kMinSupportedSmCount`.

## Upstream catch-up 2026-09-05: `neroued/master` `ad0f3d38` merged

Merge commit on `recon/catchup-20260904` (worktree `ninfer-recon`), 20 upstream commits since
`5438b743`. Full inventory and every decision: `ninfer-recon-notes/CATCHUP-20260904.md`.

What rode along and how it landed:

| Item | Outcome |
|---|---|
| `a140e7ae` exact agent prefix reuse, `b8786751` aliased state ownership | Merged; `engine_core.h` auto-merged, `program_impl.h` two trivial hunks. `b8786751` removed `SequenceState::state_source_retained`; our restore path stopped assigning it. Default shared capacity is now `max(max_concurrency, 4)`; production pins `--max-shared-prefixes 1`, so its geometry is unchanged |
| `4ac73c47`, `21a0e85f`, `a2761ec1` KV-cache restructure | **Interface adopted, kernels kept.** `KvCacheStorage` + `PagedKVStorageLayout` replace the flag bag everywhere; the four fork modes are described in `paged_kv_storage.h` and `kv_fork_mode_flags()` feeds our unchanged int8/E8 kernels. fp16 V storage for the bf16 mode adopted (35 files; no production path). The fork's two-phase bf16 prompt kernel (`694e01f0`) was DROPPED for upstream's bf16 kernels - re-port only if bf16-mode prefill on the 4090 ever matters. nvfp4/k8v4 kernels are excluded on sm_89 (`cvt.e2m1x2`), stubbed, and the modes are rejected at startup |
| `550d0ac3` llama.cpp timings + prompt progress | Upstream's `timings` block replaces ours (superset minus `ttft_ms`, which nothing consumed); `id_slot`/`session_digest` kept |
| `5f6d44e4` health readiness | `/health` is 503 until the service attaches and after a latched failure; our latch check kept |
| `6e2786c5` readable operational logs | Upstream's prose capacity lines NOT used; our structured `engine capacity/context_cache/state_pools` boot lines kept in `apps/serve/main.cpp` (`quote_log_value` re-homed there). The request done line is upstream's prose (it now carries MTP acceptance and thinking accounting itself); the fork's 09-01 structured suffix (`speculative_*`, `host_exposed_ms`, `decode_*_us_per_round`, `thinking_*`) is DROPPED - every field is in the request JSONL. LOG-CONTRACT.md refresh is a phase-2 item |
| `e51b585c` cooperative launch capacity | Mechanism adopted (runtime SM count, tile partitioning); our sm_89 route bounds kept, our hand-rolled residency predicates deleted |
| `3b50962b`, `0c5d570c`, `719d56ef` tool-call frontend; `e3aeaf8c`; `f0eb3ac7` httplib 0.54.1; `863aa8a5`; perf and fixture commits | Merged clean |
| `5973313d` self-contained frontend fixtures | Our official-tokenizer test gate removed; `NINFER_QWEN3_6_27B_HF_DIR` no longer needed by the frontend test |

Also taken in the same pass: 3090 base `5820660d` (pairwise K reduction in the unsplit GDN
gating projection, cherry-picked clean; the numerics miss it fixes is the one our own
`test_gdn_gating_proj.cpp` comment documents at the T=2689 onset).

Deliberately NOT taken: the fork's two-phase bf16 prompt kernel (see above), the request-line
suffix (see above), `ttft_ms` in the chat `timings` block (nothing consumed it), the
200-before-attach `/health` behavior.

## Wave 1 (2026-09-07): small correctness picks from the sweep, branch `fix/wave1-20260907`

Base `6f327f49` (= production `catchup-6f327f49`). Cherry-picked with `-x`; every pick was read
against this tree, not just applied.

| Source | Landed as | Notes |
|---|---|---|
| upstream PR #211 `036511d8` (ranxianglei) | `e565fe50` | Clean. `activate()` and `commit_activation()` take the compute stream; both `program_impl.h` call sites pass `device.stream`. Closes the #210 crash class in our `logical_kv_store.h` |
| 0xrjman `cbf51152` stale-plan drop | `02d0976d` + `87230fd9` | Two trivial conflicts (our extra test fixtures; the `state_count` line). The regression scenario `shared-release-source` was written for the nvfp4 artifact; `87230fd9` runs it on the groupwise artifact and on `NINFER_PREFIX_REAL_KV_DTYPE` like the other fork fixtures |
| gzenz `839e5226` reasoning-effort tiers | `51bf3597` | Clean. `minimal -> low`, `high/max -> xhigh` instead of 400 |
| 0xrjman `54acc835` trigger-B footprint | **dropped** | Patches `state_footprint()` and `state_source_retained`, both removed by upstream `b8786751` (aliased state ownership), which we merged on 09-05. The double count it fixes cannot occur in the exclusive-resource accounting |
| gzenz `ff372161` checkpoint budget pre-check | **dropped** | Patches gzenz's own checkpoint-copy block (their `1b11452c`); this tree has neither the block nor `state_footprint()` |
| tensorninja `e3a129c3` restore diagnostics | **deferred** | Instruments `restore_could_be_hosted()` and the deferred-retry gate of their `c1e4eb1e`, which this tree does not have (`concurrent_executor.h` is gone since the 08-30 merge). A design port onto `engine_core.h`'s restore path, not a cherry-pick |

Validation: CPU build in `ninfer-catchup` (`-DNINFER_BUILD_BENCHMARKS=ON` now, for
`ninfer_context_cost_bench`), then the GPU window `ninfer-recon-notes/deploy-20260907/run-gpu-window-wave1.sh`
(full ctest, three real-model E2E scenarios on rk4v4-e8 including the new one, fresh-server A/B
vs the production binary, effort=high/minimal must be 200, restore of a production slot copy,
and a 4090 context-cost calibration run).

**Window 2026-09-07 18:39-18:47 UTC: all gates passed; DEPLOYED 18:48 UTC as `wave1-6f1399c9`.**
ctest 109/109 (99 ran, 10 skipped: 35B/score/load-plan without weights, nvfp4/k8v4, A4);
E2E on rk4v4-e8 `ok` x3 (default, automatic-private-anchors, shared-release-source with
`dropped=0` on every round); boot geometry identical; effort=high and effort=minimal 200 on the
new server; production slot copy restored `n_restored=32324 session=30447b168cab0f2c` = ref;
A/B vs the 6f327f49 binary on fresh servers: prefill +0.6..0.9%, cache hits identical
(15,168 / 15,190), 49k round-trip flat (save -1.3%, restore +0.3%), decode within content noise
(the probes carry a per-run nonce, so acceptance counts are not comparable across runs). One
outlier: the 16k save took 869 ms vs 602 ms; the 49k save was flat and the order was new-first
this time (ref-first on 09-05 showed the opposite sign), so it reads as page-cache order, not
the binary. 0 warnings. Rollback binary `ninfer-serve.pre-wave1-6f1399c9-20260907-1848`.

Context-cost calibration (`ninfer_context_cost_bench --suite all`, default reps): **prefill fit
accepted** (p95 relative error 1.4% training / 4.3% held-out, ordering 50/50 + 10/10), d2h and
h2d fits accepted (p95 19-27%), **d2d fit rejected** (p95 54% / 64%: 4-64 MB contiguous copies
in 1-8 operations measured 2x the model), so no preset file was written. Re-run queued with
`--transfer-warmup 6 --transfer-reps 41 --prefill-reps 7` (see the line below when it lands).
The re-run (`context_cost_4090_r2.json`, 41 samples) rejected d2d again with the same shape
(p95 56.7% / 64.1%, median 4.6% / 6.7%; the multi-operation 4-64 MB contiguous copies run ~2x the
`max(batch + ops*op_ns, bytes*ns_per_byte)` model), so the miss is the model shape on Ada, not
noise. **Hand-assembled preset DEPLOYED 2026-09-07 18:59 UTC**: `ninfer-recon-notes/deploy-20260907/
context_cost_presets_4090.json` (all four r2 fits, provenance notes carry the d2d caveat; the
compiled generic default priced 4090 d2d ~10x slower than measured and prefill at the 5090 rate).
Validated on a fresh :8087 server (`transfer_source=external prefill_source=external`, completion
ok, 0 warnings), then production recreated with `-v ~/ninfer-deploy/context-cost:/context-cost:ro
--context-cost-presets /context-cost/context_cost_presets_4090.json` (`recreate-containers.sh`
edited, backup `.pre-ccost-20260907`). Boot line on production reads external/external. Watch in
the soak: `materialization.predicted_*` and `prefix_reuse_path` in the JSONL - this is the first
time the planner prices restores and prefills with 4090 numbers.

## Upstream catch-up 2026-09-12: `neroued/master` `d4929686` merged (catch-up #3)

Merge commit on `recon/catchup-20260912` (worktree `ninfer-recon`), 93 upstream commits since
`ad0f3d38`. Full inventory and every decision: `ninfer-recon-notes/CATCHUP-20260912.md`.

| Item | Outcome |
|---|---|
| DFlash2 (`4df5e0b4`..`385b30ce`, ~75 commits: op set, converter, `dflash_impl.h`, corpus bench) | Merged as-is. Optional at runtime (`--spec dflash2 --draft-tokens 1..15`, weights bound only when the artifact carries `dflash2/*`); our artifact has none, production line unchanged. `kMaximumDFlashDraftTokens` 0 -> 15 for the 27B (upstream value; inert without flag + artifact). soohl's 4090 numbers (K5 vs MTP3: code +5%, prose -6%, JSON +23%) say bench pi's mix before fetching the 19 GiB artifact |
| `d4929686` materialization search budgets (closes #176) | Merged. `materialization_budget.h` (50 ms / 10 ms admission allowance, renewing search budget), `pressure_planner.h` rewrite, 10 new request-log `search_*` fields. **Supersedes `fix/d1-planner-search-budget` (`060b1b21`, deleted locally; delete `fork/fix/d1-planner-search-budget`).** Our `best_reuse_prompt_tokens` kept beside the new fields (different question: reuse on the table vs what the search did) |
| `a7818988` variable-width small-T attention | Merged. The int8 partial kernel now writes FP32 partials and runs FP16 P / FP16 V `mma_f16` (was bf16); the fork's PackedK/E8Root/PackedV paths decode into the int8 staging tiles upstream of that change, so every int8-family mode moves together. `tokens >= 6` split cases, the reduce `<Int8>` dispatch, the batch>1 `target_ctas` branch, `causal_attention_chunk_tokens` and the `resolve_route` switch all cover the int8 family (fork case labels added). **Numerics change on the production path - the window A/B at temp 0 is the gate** |
| `03177b91` kv coverage during speculative settlement | `materialize_to_tokens` -> `ensure_mapped_to_tokens`; our restore path renamed and given an explicit `mapped_pages == page_count` check (the old call threw on a mismatch, the new one treats coverage as a lower bound) |
| `b88c0f6f` H2D upload completion, `9f0575bb` bf16 include, `b158afe2` BPE flat table, `641ef3e7` ASCII NFC skip | Merged clean |
| DFlash2 W8 routes vs the sm_89 static shared-memory cap | **Fork fix.** CUDA 13.1 lets `sm_120a` allocate more than 48 KB of static `__shared__` (verified: 51,200 B compiles for 120a, fails for 89), and upstream's schedule assert guards only the 99 KB sm_120a limit. Three DFlash2 W8 instantiations overflowed at `nvlink` (`w8_small_t_mma_kernel<5120x4096, T=40, 8 K-warps>` from the dynamic grouped conv, 49,664 B; SwiGLU `r64_c96_k128`, 50,176 B, both Full variants). Under `NINFER_SM86` the conv keeps 8 warps only to T=32 and the c96 route runs the qualified 64x80 K128 tile; both schedule structs gained 48 KB `static_assert`s so the next overflow fails at compile time |
| MoE / nvfp4 perf (`487f8977`, `437e9f98`, `ce954918`, `7f14d963`, `ee9d5192`, `1c8f8acc`) | Merged; not exercised on this dense sm_89 target |
| Option tables, docs, bench usage | kv-dtype lists stay the fork's; `--spec` gains `dflash2`; the bench's `--mtp-draft-tokens` went with upstream's `--spec/--draft-tokens` |
| `tests/ops/softmax_attention/causal_cache.cpp` | Our unrotated `encode_key_row` oracle kept (the fork's Int8Group64 K is not D256-rotated); its `logical_k_quantized` output dropped because upstream's `cache_value()` now dequantizes `codes * scale`, which equals it for unrotated codes |

Also in the same pass: the dead `catch (const std::invalid_argument&)` our wave-1 pick `02d0976d`
left behind `catch (const std::logic_error&)` in `progress_materialization` was removed
(`-Wexceptions`).

**GPU window 2026-09-12 19:23-19:33 UTC (`ninfer-recon-notes/deploy-20260912/run-gpu-window-catchup3.sh`): all
gates passed.** ctest 119/120 (11 skipped: nvfp4/k8v4/A4, dflash2-real, 35B, score, load-plan) - the one
failure was the attention unit test's oracle, not the kernels: upstream's harness rewrite rotates the
reference query for every non-bf16 storage, which is wrong for the fork's unrotated Int8Group64 codec
(fixed, PASS on an idle GPU in a 47 s follow-up stop); the same run had lost the sm_86 skip on the two
new nvfp4/k8v4 case loops (abort -> guarded). Real-model E2E on rk4v4-e8 ok x3 (default,
automatic-private-anchors, shared-release-source). Production slot copy restored digest-exact
(n_restored=105383, session 5a96e7894feb885f, 0.79 s) through the renamed `ensure_mapped_to_tokens`
path. Boot geometry identical, effort high/minimal 200, 0 warnings. A/B vs `wave1-6f1399c9` on fresh
servers: prefill 2k/16k/49k 1968/2063/1883 vs 1919/2015/1844 tok/s (+2.1..2.5%), cache hits identical
(15,168 / 15,190), 49k save 1021 vs 1071 ms, 16k save 537 vs 581 ms, restores flat, decode 600-token
probes complete at normal acceptance (code 0.57 vs 0.58, prose 0.43 vs 0.36; the nonce makes counts
content-dependent). New planner fields present (`search_stop_phase`, `search_granted_ns` 5 ms,
`insufficient_expected_gain` on the unpressured probes).

**DEPLOYED 2026-09-12 19:55 UTC as `catchup3-9b26ae76`** (`deploy-catchup3.sh`: provenance ok, installed binary
byte-identical, health ok, gateway 200, boot `long_anchors_per_continuation=2 auto_long_anchors=2`, 0 warnings).
Rollback `~/ninfer-deploy/bin/ninfer-serve.pre-catchup3-9b26ae76-20260912-1955` (= wave1-6f1399c9).

**SOAK PASSED 2026-09-12 19:55 -> 2026-09-20 07:46 UTC (7.5 days); `rtx4090-port` FAST-FORWARDED
`1bd56c9a` -> `6505b7fc` on 2026-09-20.** Live binary sha `fff53605` == `bin/ninfer-serve.catchup3-9b26ae76`,
container RestartCount 0, one `server_start` (the deploy), no VM boot since 09-11. JSONL since the deploy:
129 `request_done` / 0 `request_error`; 748 stderr lines, 0 WARN/ERROR/crash signatures. Reuse paths
`private_endpoint` 119, `root` 8, `private_response_replay` 2; all 68 deep turns (>50k tokens, max
203,867) reused, ttft median 0.82 s, max 14.1 s (16-19k appended tokens at 73-118k depth, partial reuse).
The 8 roots are the 2 deploy probes and 6 first turns (`message_count` 3, 18-23k tokens at 2,033-2,059
tok/s; the wave-1 soak saw 1,890-1,950 at 29-39k); no `root` with `best_reuse_prompt_tokens` > 0. Planner
rework: `stop_reason` `insufficient_expected_gain` 127 / `no_pressure` 2, `budget_exhausted` 0 (32/81 on
wave-1), `selected_maximal_fallback` 0, `search_stop_phase` `expansion` 127, `search_granted_ns` median
5 ms max 10 ms (one renewal), `prefix_cache_hit_tokens` == `best_reuse_prompt_tokens` 129/129. Preset:
`predicted_now_ns` / measured prefill median 1.02 (p10 0.99, p90 1.05, n=60). Pressure was exercised for
the first time in a soak: `private_owners_evicted` 1, `checkpoints_dropped` 4, `spill_pages` 1,880
(7.2 GB d2h, 2.2 GB h2d), state `restores` 1, `search_budget_exhaustions` 0, host KV occupancy 2.3 GB at
the end - demotion and promotion ran under two resident sessions with no reuse loss. MTP acceptance
62.1% (98,378/158,511; temp 1.0 + tools n=127, like-for-like wave-1 53.4%). Effort tiers `xhigh`
127/127, `tool_call_parse.fallback_reason` none 129/129, 0 schema mismatches over 148 tool calls. Decode
(completion >= 200) median 105.8 tok/s. Slots 09-19/09-20: 9 saves, 3 auto-saves, 2 restores from disk
(92,798 tokens in 2.57 s, 24,726 in 1.50 s), 0 SKIPPED - restore-from-disk is production-verified on this
build. Caveat: light traffic (129 requests, 09-17 idle). Fast-forward per the 09-07 convention
(`git merge --ff-only`, `ninfer-recon` detached, `recon/catchup-20260912` deleted); the push deletes
`fork/fix/d1-planner-search-budget`.

Deliberately NOT taken: nothing dropped this time. Upstream PR #211 (the stream-ordered
membership publish we carry as `e565fe50`) was closed unmerged by its author on 09-10 and master
still publishes unordered - the patch stays fork-only.

## Vision budget per item, 2026-09-23 (`328d9aa8`)

A pi session on production failed every turn with `media_budget_exceeded` "vision raw patches
exceed processor budget" after its eleventh 1282x665 screenshot (request log: 10 images
admitted, 11 rejected). Agent clients send every earlier image again with each turn, and the
processor counts all of them. Cause: our `73b42127` (2026-08-18) tied the aggregate prompt
budget to `--vision-max-tokens` (8192), which was right while the encoder held a whole request.
Upstream `fc5c4834` (2026-08-24, catch-up #3) encodes one item at a time and splits the budgets
into 32768 per prompt and 16384 per item; the lockstep survived the merge. The fix restores the
upstream aggregate and applies `--vision-max-tokens` to the single-item budget, which is what
the encode workspace holds. Frontend test added (4 x 768 tokens admitted at a 1024 cap, one
1280-token item rejected). Deployed 2026-09-23 12:50Z as `visbudget-328d9aa8`, rollback
`bin/ninfer-serve.pre-visbudget-328d9aa8-20260923-1250`. Live check: 12 x 1282x665 (10,080
tokens) returned 200 with the correct count; one 3200x3200 image (10,000 tokens) returned 400
`media_budget_exceeded` "single media item raw patches exceed Vision execution capacity". The
remaining wall is the 32768 aggregate, about 39 such screenshots or 8 images at pi's
2000x2000 resize cap in one conversation.

## Community triage 2026-10-07, round 4 (PR #14)

T-Crypt answered the round-3 P1 with `e7495e6c` on 2026-10-06. The commit implements the rule that
the round-3 review proposed: a call end is rejected when another `</tool_call>` follows it before
the next `<tool_call>` with more than close markers between them. It was tested merged onto the tip
`b015b759`. A magnus GPU window ran on 2026-10-07 (18:28 to 18:51 UTC). Review, gate scripts and
evidence are in `ninfer-recon-notes/triage-20261004/round4/`.

| Item | Verdict | Notes |
|---|---|---|
| PR #14 `--tolerant-tool-calls` | MERGED (`91829c35`..`6d88143b`, rebase), default off | Every gate line passes in a16 and int8 prefill, see below. The rebased tree is identical to the gated merge. Not deployed: with the flag off, the PR changes no response |
| `docs/serving.md` | FIXED in `64911f31` | The PR did not update the tolerant-mode rules. Our commit documents the new rule, its two fallbacks and the 64-attempt budget |

- Harness: the same result as the round-3 prototype on every case. On the real model outputs of
  round 3, the quoting `write_file` now recovers the full 1,840 characters (round 3: 190), and the
  two-call variant is fixed. A call followed by prose that names `</tool_call>` falls back, as in
  strict mode. Timings are unchanged (2,000 real calls: 6.5 ms). A 1 MB worst case for the new
  check (close markers, prose only at the end) takes 11.5 ms because the attempt budget bounds the
  rescans. A value that quotes the closing tags 5,000 times exceeds the budget and falls back; 40
  times recovers in full.
- GPU gate, run twice with the round-3 method: a16 prefill (the round-3 configuration) and
  `--prefill-activations int8` (production since 2026-10-06). Flag off is identical to production
  in 117 of 117 (a16) and 73 of 73 (int8) responses. In tolerant mode, 109 and 67 responses stopped
  on the length limit and none returned a tool call. No recovered response left tool markup in
  its content. The llm-eval tool-use lane is 5/6 with and without the flag in both modes.
- The a16 base reproduced round 3 byte for byte (117 of 117), so the P1 output came back. Tolerant
  mode now returns the full file with no content left over. Under int8, the model does not write
  the real tags for that prompt, so the int8 run is a regression check only.
- PR #15 got an independent confirmation from gryknight9 on a second 4090 (Docker, 262K,
  `rk4v4-e8`, MTP 3, tip `b015b759`): cold prefill 1.67x at 5.3K and 1.71x at 10.4K tokens (a16
  about 2,100 tok/s, int8 about 3,600 tok/s).

## Community triage 2026-10-06, round 3 (PRs #14, #15)

T-Crypt answered both round-2 reviews on 2026-10-04. A magnus GPU window ran both gates on
2026-10-06 (20:11 to 20:50 UTC). Each PR was tested merged onto the tip `f4c67501`. Reviews,
gate scripts and evidence are in `ninfer-recon-notes/triage-20261004/round3/`.

| Item | Verdict | Notes |
|---|---|---|
| PR #15 INT8 group-64 activation prefill | MERGED (`7a507d1f`..`8adf1797`, rebase), default `a16` | Every gate line passes, see below. DEPLOYED 2026-10-06 21:17 UTC as `pr15int8-8adf1797` with `--prefill-activations int8` in the container line: cold 8.3K prefill 3,558 to 3,577 tok/s (a16 about 2,085) |
| PR #14 `--tolerant-tool-calls` | CHANGES REQUESTED, merged in round 4 | Round-2 items are fixed, see below. One new P1 from real model output: a complete call whose file content quotes the closing tags is recovered as a truncated file |

### PR #15 gate v3

- The round-2 "gating norm +8 to +10%" finding was our measurement error. The v2 gate ran the
  base binary first in the norm block, and the first run of a bench block is an outlier. The v3
  gate alternates base, PR, base, PR in both cache modes. Base run 2, PR run 1 and PR run 2 agree
  to 0.1 us at every T. No 27B gating MMA instance has `NormalizeInput`, and the 27B norm op is
  the fused SIMT kernel or rmsnorm plus the split kernels of `--op control`. Rule for later kernel
  A/B benches: alternate the binaries and report the first run of a block separately.
- Register counts of the 27B gating kernels on sm_89 (cuobjdump): split-1 82, split-2 and split-4
  (16 warps) 74, split-8 (8 warps) 65. Full and Predicated are equal. These match T-Crypt's log.
- `ninfer-perplexity --prefill-activations`: quick set A16 4.3429, INT8 4.3446 (+0.039% overall,
  worst domain +0.082%). INT8 scoring takes 142 s, A16 248 s.
- `prefix_real` with `NINFER_PREFIX_REAL_PREFILL_ACTIVATIONS=int8`: default, `e8-host-replay` and
  `automatic-private-anchors` pass. An invalid value aborts.
- Serve, production flags plus `--no-prefix-reuse`: the PR default is within 0.1% of base at
  6K, 50K and 128K, and greedy decode is byte-identical. INT8 prefill is +73%, +61% and +45%.
  MTP acceptance over 16 greedy prompts is 0.628 (a16) and 0.632 (int8). The llm-eval tool-use
  lane is 5/6 in both modes. `server_start` reports the effective mode and the flag, also with
  the `NINFER_A16_PREFILL=1` override.
- Open, optional: a `--q4q5-policy` switch on the attention-input and GDN-input benches.

### PR #14 round 3 (`5700b7d9`)

- Harness against `cdb8c102`: malformed openers (`<funtion=`, `<function =`, `<Function=`) fall
  back to text. The truncation guard works in the decoder. The strict path is byte-identical.
  Tolerant cost for 1 MB pathological outputs went from 8 to 12 s down to 1 to 2 ms. A 346 KB
  output of 2,000 real calls takes 5.5 ms (`cdb8c102`: stopped after 10 minutes).
- GPU gate with a `max_tokens` sweep over prompts that make the model quote tool markup inside
  `write_file`: with the flag off, 117 of 117 responses are identical to production. In tolerant
  mode, 109 responses stopped on the length limit and none returned a tool call. A replay of
  those texts without the guard recovers 26 of them as a truncated `write_file`.
- P1: the model wrote a complete call whose file content quotes `</parameter>`, `</function>` and
  `</tool_call>`. Tolerant mode closed the call at the quoted tags (190 of 1,840 characters) and
  returned the rest as content (`recovered_from=trailing_content`). Strict mode returns text. The
  review proposes a rule: a `</tool_call>` before the next `<tool_call>` means that the call ends
  later, unless that text holds only close markers. A prototype of the rule recovers the full
  file and changes no other harness case.

## Community triage 2026-10-04 (issue #12, PRs #10, #13, #14, #15)

One new issue and three new PRs arrived after 2026-09-22, and PR #10 had an unreviewed update.
All five items were answered on 2026-10-04 against tip `aeeba414`. A magnus GPU window ran the
gates for #13 and #15. Reviews, gate scripts and evidence are in
`ninfer-recon-notes/triage-20261004/`.

| Item | Verdict | Notes |
|---|---|---|
| PR #10 native Windows (MSVC) build | MERGED (`cb2df08a`..`b8c71f00`, rebase) | The `u128` shim is correct now. A differential test against native `unsigned __int128` (68M checks, UBSan on, all shift amounts 0..127) found 0 mismatches. The Linux path still selects the native type. Open P4 items: `make_prefill_work(0, UINT64_MAX)` wraps to 0 instead of saturating (also on Linux, unreachable with real token counts), a raw `__int128` in the context-cost bench fixture, and README wording |
| PR #13 capture-path entitlement credit | MERGED (`49deb207`, rebase) | Fixes a real engine latch, see below. Open gap: at `--max-concurrency 2` a shared prefix that ANOTHER lane releases is still not credited (the baseline is taken per capture transaction) |
| PR #15 INT8 group-64 activation prefill (port of soohl `19aa1337`) | CHANGES REQUESTED | Gate on the 4090, see below. P1: the tree does not build with tests and benchmarks on (soohl's `KvCacheStorage::E8Group64` in two tests, the Q5 bench update not taken). A gate-only build fix is in `pr15-gate-build-fixes.patch` |
| PR #14 `--tolerant-tool-calls` (re-implementation of gzenz `3c0b4dc5`) | CHANGES REQUESTED | Strict mode is byte-identical with the flag off. With the flag on, three adversarial cases make the parser emit a call that the model only quoted (for example `run_shell {"command":"rm -rf build"}` from a quoted example) |
| Issue #12 MTP-compatible structured output (UDP #12, XGrammar) | OPEN, after catch-up #4 | The MTP design of UDP #12 is correct (mask on the first target token, sequential draft validation, longest valid prefix). It patches files that no longer exist here, fetches XGrammar with git at configure time (fails in our image: CMake 3.28, no git), forces JSON inside an open `<think>` block, and lets special tokens be sampled inside JSON strings. Upstream has no constrained decoding yet (Neroued/ninfer#33) |

### PR #13 gate (magnus, base `aeeba414` against the PR, fresh server per run)

The driver is a Claude-Code-shaped Anthropic replay: `cache_control` on the system prompt and on
the last two user messages, microcompaction, edits two messages back, title calls. With the
default two-lane catalog (`--max-concurrency 2 --auto-long-anchors 2`, 4 private, 4 shared), the
base binary latched in 2 of 2 runs. Two requests failed with `capture transaction is not
reservable` (`program_impl.h:7850`), and `/health` stayed 503 until restart. The PR passed 40/40
in both runs, with 32 to 33 non-root reuses. The reporter geometries (one lane, anchors 2,
private continuations 8 and 2) and the production geometry (2/1/2 with vision and the 4090 cost
preset) passed on both binaries. Production never hit the latch because `--max-shared-prefixes 1`
keeps the shared catalog small. The issue #9 fast geometries passed 18/18, 18/18 and 19/19 on
the PR, and the `context_store` and `prefix_real` GPU tests passed. The merged tip `49deb207`
builds, and the planner, context-cost, materialization-budget and `u128` tests pass. The merge is
NOT deployed: production stays `visbudget-328d9aa8`.

### PR #15 gate (magnus, 400 W limit, production flags plus `--no-prefix-reuse`)

| Prompt | base A16 | PR INT8 | PR with `NINFER_A16_PREFILL=1` |
|---|---|---|---|
| ~6K | 2115 tok/s | 3622 (+71%) | 2089 (-1.2%) |
| ~25K | 2021 | 3389 (+68%) | 2006 (-0.7%) |
| ~50K | 1875 | 3012 (+61%) | 1866 (-0.5%) |
| ~128K | 1529 | 2221 (+45%) | 1525 (-0.3%) |

- Op tests 8/8 pass on sm_89. The real-model `prefix_real` scenarios (default, `e8-host-replay`,
  `automatic-private-anchors`) pass with INT8 prefill.
- Perplexity, quick set (4 of 16 streams, about 256K tokens, INT8 KV): +0.039% overall, worst
  domain `english_long_form` +0.082%.
- With `NINFER_A16_PREFILL=1` the greedy decode text is identical to base on both probes. Boot
  geometry, KV headroom and VRAM after a 128K prefill are identical for all three binaries.
- The PR replaces the constexpr cooperative residency of the GDN gating kernels with an uncached
  runtime occupancy query. This change is outside the stated scope and also affects the A16 path.
  The gating kernels are 8 to 9.5% slower than base, which is the 1% A16 prefill loss in the
  table. It relates to the Don-Chad `7afc8e17` note above.
- After an INT8 prefill, MTP acceptance on a greedy prose probe fell from 0.483 to 0.422 (decode
  -7.6%). The code probe did not change (0.808 against 0.810). Before INT8 becomes the default,
  run the temp-0 tool-call A/B, including MTP acceptance, and one vision probe.

### Other findings from this triage

- Claude Code 2.1.x always sends `thinking.display: "omitted"`. The Anthropic surface rejects it
  with a 400 (`requires encrypted hidden-reasoning restore semantics`, upstream `65f620a7`, still
  on `neroued/master`). Claude Code therefore works only with `MAX_THINKING_TOKENS=0`. Candidate
  upstream issue.
- The strict tool-call parser scans quadratically: a 1 MB pathological output takes 9.9 s and
  128 KB takes 158 ms. Real outputs do not reach this.
- Upstream is 83 commits past our merge base `d4929686` (41 at the 2026-09-22 handoff).

## Community triage 2026-09-22 (issues and PRs opened on this fork)

Six issues and four PRs had accumulated since 2026-08-30 without a reply; the fork was not
being watched (fixed the same day). Everything was answered on 2026-09-22 against tip `a889ce43`.

| Item | Verdict | Notes |
|---|---|---|
| PR #8 cancelled Anthropic stream rendered as a 500 | MERGED | Mirrors the existing `ClientDisconnected` pattern; upstream copy of the file is unchanged, worth offering upstream |
| PR #6 request accounting from `RequestCapacity::active` | MERGED | Metrics no longer lose accepted work during prepare; `ninfer_serve_metrics_test` passes; `ServeMetrics::active_snapshot` had no other users |
| PR #10 native Windows (MSVC) build | CHANGES REQUESTED | Linux path untouched; the MSVC `u128` shim is wrong (`!` for `~`, shifts >= 32 treat 64-bit limbs as 32-bit), so `q32_product_ns` returns 0 on Windows and the context-cost model runs blind there. Verified by compiling the emulated branch on GCC |
| PR #7 DFlash2 for sm_89 (78 commits) | CLOSED, superseded | 77 commits are upstream's DFlash2 series already in catch-up #3; the remaining commit reverts the `kv_storage_is_int8_family` widening in `small_t.cu` (E8 regression) and uses the pre-`d4929686` `post_mixer` signature. Its BM32/C96 SwiGLU tile versus our r64/c80 fallback (`87189c68`) is an open kernel-bench question |
| Issue #9 engine latches after a stale reuse candidate | OPEN, highest priority | Six invariants in `request_plan_impl.h` throw when a catalogued endpoint frontier disagrees with the live `execution_frontier`; lifts to HTTP 500 and a permanent 503. Both reporters run Claude Code on the Anthropic path; production (OpenAI chat only) has never hit it. Reporter offered a throw-to-logged-skip PR (accepted with logging requirements); gzenz `20213d4b` is a sibling fix plus an admission re-arm. A scripted Anthropic tool-call sequence did not reproduce it at 1.7K tokens |
| Issue #5 planner 5 ms budget | CLOSED | Superseded by `d4929686` in catch-up #3 (`budget_exhausted` 0 of 129 in the soak) |
| Issue #4 DFlash2 artifact fails to load | OPEN | The original error is fixed at tip. New problem: Hugging Face `main` is container v3 since 2026-09-15 and the download scripts pointed at it; pinned to `3526913004b1` with SHA-256. The DFlash2 revision `dc370fb6295a` is expected to load but is untested on this 4090 |
| Issue #2 dual 4090 / 1M | CLOSED | Single GPU by design; YaRN forks named |
| Issue #1 two times the output of llama.cpp | OPEN | Sampling defaults, effort mapping, `--preserve-thinking`; waiting for request-log lines |
| Issue #3 thanks | CLOSED | |

### Follow-up results, same day (magnus experiments)

- **Issue #9 root cause: device StateImage exhaustion.** Capacity is `max-concurrency +
  --device-state-slots` (default extra = concurrency); retained endpoints, up to two automatic long
  anchors per continuation, and shared prefixes all hold one. With the defaults a single retained
  tool-call turn exhausts the pool at one lane (`std::bad_alloc` on the first continuation) and the
  second conversation exhausts it at two lanes (`retained materialization source is unavailable`,
  thrown from inside `abort_transaction()` at `program_impl.h:5961` when `start_request`'s
  per-request invariant aborts publication; the abort handler asserts the consume-to-active source
  is still catalogued, throws out of the catch, masks the real invariant, and the engine latches).
  Production is safe only because it runs `--max-shared-prefixes 1`: 240/240 requests, 43K contexts
  under eviction, no failure. Verified workarounds keeping prefix reuse: `--device-state-slots 8`
  (two lanes) or `4` (one lane), `--auto-long-anchors 0`, or `--max-shared-prefixes 1` (two lanes).
  The six `inspect_lane` throws discussed on the issue never fire. Deterministic repro: two
  2K-token tool conversations fail within 30 s (`ninfer-recon-notes/issue9-20260922/`). Branch
  `fix/materialization-abort-invariant` (pushed) rethrows the original invariant and prints both
  sides of the mismatch: the invariant is `materialized sequence does not match its active
  entitlement` at `program_impl.h:7234`, and the only disagreeing field is `host.state_slots`
  (expected 1, actual 0): once a source's StateImage has been demoted to host, the restore drops the
  host replica while the planner's `active_entitlement` keeps it. The fix proper is (1) that
  entitlement (or `start_sequence`) for host-resident sources, (2) an abort path tolerant of a
  consumed source, (3) an exhausted device pool as a root fallback instead of `bad_alloc` at
  `reserve_destination()`.
- **Issue #9 FIXED 2026-09-22, merged as `539ccdcd` (tip `81b68a20`).** The entitlement
  counter above was the symptom; the cause is the admission planner's private-endpoint consume
  branch in `request_plan_impl.h`, which adds every long anchor of the source to
  `active_optional_resources` by residency without `state_exclusive_to_sequence`, while the
  sibling rewrite-restore branch and the actual-side `sequence_exclusive_state_resources` both
  filter. An anchor image another checkpoint owner also references (instrumented: `refs=2
  owned=1`) is counted as a slot the sequence does not own; the miscount became visible once the
  device pool filled and that anchor was demoted to host (`optional_host=1` vs actual 0), and at
  one lane as `std::bad_alloc` from the state reservation. Fix = one `continue` on the
  non-exclusive case. Gate on the merged tip: fast geometries (`4/4/2`, `4/1/2` at one lane,
  `2/1/2`) 14, 15, 18 requests, 0 errors; long run 3 x 43K tokens, two passes, 35 requests,
  0 errors, 23 private-endpoint + 3 shared-prefix reuses; `ctest -j1` 109 passed, 11 expected
  skips, 0 failed. Upstream has the same code at
  `src/models/qwen3_5/program/planning/request_plan.cpp` line 613 (filtered loop at 651);
  report draft in `ninfer-recon-notes/issue9-20260922/upstream-issue-draft.md`, not filed. Note
  that upstream never exercises that loop: automatic long anchors are this fork's feature
  (`9f63c77b`, 2026-09-02) and no OpenAI or Anthropic request can place an anchor marker
  upstream, so for them it is a latent defect; the trigger was ours.
  Also merged: the abort path now rethrows the ORIGINAL invariant when an acknowledgement fails
  (`2c046095`), and the entitlement mismatch names both sides and every StateImage (`81b68a20`).
  DEPLOYED 2026-09-22 18:44Z as `issue9-81b68a20` (binary sha `f2cf00e4`, the gate build;
  rollback `bin/ninfer-serve.pre-issue9-81b68a20-20260922-1844` = `community-01c22ab6`, the
  PRs #6/#8 build that served 18:14Z to 18:44Z). Open follow-ups, low priority now:
  an abort after `start_request` consumed the source still asserts on it (needs a
  ResourceManager contract for a source lost on abort), and an empty device pool still surfaces
  as an exception rather than a root fallback. Diagnostic instrumentation (pool tags, core
  `bad_alloc` tags, state dump) kept as `issue9-20260922/diag-instrumentation.patch`.
- **DFlash2 on the 4090** (`dc370fb6295a`, issue #4 closed): loads; MTP3 on it is unchanged
  (140/107 tok/s code/prose, bit-identical across boots). DFlash2 K=3 does not fit 262K on 24 GB
  (about 1.0 GiB short with vision, 0.53 GiB without): 224K without vision or 192K with vision,
  149.8/104.0 tok/s; K=5 190.3/94.2. At temperature 0 DFlash2 does not reproduce MTP3's text and
  K=3 differs from K=5, so the verify path is not exact on `rk4v4-e8`: quality gate before any
  default change.
- PR #8's bug was observed live at 13:42Z (a dropped client mid-generation logged as a response-render
  500); the merged #6/#8 binary (sha `58ae3ea0`) is built in `ninfer-catchup` and not deployed.

External data worth keeping: the PR #7 thread has a K sweep on a 4090D 48 GB (`xwfl15632`):
DFlash2 K=3 133.3/111.0 tok/s (code/prose) versus MTP3 122.0/97.5 on the same build, and K=7
collapses prose acceptance to 24%. It also pins the v3 artifact gate to upstream `98dada0e`
(jinja templates) plus the v3 loader (`4cde7ad0`, `04350ba9`).

### Recommended order after the 2026-09-22 triage

1. Catch-up #4, scoped before merging: upstream `9e163eee` is +41, and its v3 refactor moved 39 of the
   81 source files this fork changed since `d4929686` (runtime now under `src/models/qwen3_5/`), so it
   is a re-homing of the runtime delta. Payload = v3 artifact loader + jinja templates (unblocks new
   users; the download pin is a stopgap). Fold in upstream #297 and #294 if merged. The #9 filter must
   travel to `program/planning/request_plan.cpp`.
2. soohl INT8 dense prefill: kernel-bench + temp-0 quality gate (unchanged).
3. DFlash2 only after the quality gate (greedy text differs from MTP3) and a fit decision (224K no
   vision or 192K + vision on 24 GB).
4. Parked: the two #9 robustness follow-ups; gzenz tool-call pair and the UDP structured-JSON port,
   re-judged against upstream #294/#299 at catch-up #4.

Hand-off for the next session: `ninfer-recon-notes/HANDOFF.md` top (2026-09-22).

## Inbound sweep 2026-09-12 (all remotes, upstream issues, forks)

Counts vs `rtx4090-port` `1bd56c9a`. Upstream +93 (taken above). The rest, ranked:

| Source | What | Decision |
|---|---|---|
| **UDP `v1.2.0`** (`25c3099f`, `f34c3358`, `f589d684`) | Structured JSON generation (`response_format`) through a grammar mask, MTP preserved, mask scoped to the first MTP position | **Design port** (patches `concurrent_executor.h`, gone here since 08-30). Closes one of the two TEB feature gaps |
| UDP `912cbb56` | Validate-only stubs for the 66 `dflash2/*` tensors so the new HF artifact revision loads on a non-DFlash2 build | Not needed after this catch-up (upstream binds them optionally) |
| **gzenz `3c0b4dc5`** (9 files, +265, parser tests) | Tolerant text-form tool-call recovery (last-close parsing) | **Small port**; matches our measured 0.23% malformed calls |
| **gzenz `a8cdc1a6`** | Tool-call arguments typed by their declared JSON schema (boolean/int coercion, strings preserved) | **Small port**; matches `tool_call_parse.schema_mismatch_arguments` |
| gzenz `ae8eb23e` | Post-thinking sampler (second preset after the reasoning close, registered 0.2/0.95/20, HTTP `post_thinking`) + atomic {KV+state} cache units | Medium; the temp-0.6 tool-work mitigation could become a post-thinking preset |
| gzenz `6df4f011`, `cb535944` | Unified KV+state host demotion with state-only fallback and OOM recovery (production-verified on their 5090); vendored jinja chat-template engine replacing the hand-written renderer | The item-7 host-KV design choice now has a shipped candidate; the jinja engine is a large replatform - read before the next frontend work |
| soohl `aaec1533` | DFlash2 + adaptive MTP + host-staged BF16 vision on Ada, with the MTP3-vs-DFlash2 table | Evidence for the DFlash2 decision (above); the INT8 dense prefill (+68%) is still the top perf item |
| upstream #229 (Gene0Liu) | Duplicate of #176 filed 09-10; its JSON carries `best_reuse_prompt_tokens`, a field only this fork logs | Someone runs this fork or a derivative; #176 is closed by `d4929686` |
| upstream #224 (MichaelDementii, closed by Neroued) | Forced `tool_choice` by prompt opener; upstream wants constrained decoding later | The TEB `tool_choice=required` gap stays open (#223 tracks) |
| upstream #213/#215/#216 (ranxianglei, closed unmerged) | Groupwise-W8 variant; W8 linear_add split-K hardcodes 2048 rows ("silent corruption when routed to other row counts"); W8 short-K decode inefficiency | Probably not our path (our W8 is the embedding/head only, text body Q4/Q5/Q6) - verify which kernels the groupwise profile routes through `w8_linear_add_*` before dismissing #215 |
| 0xrjman +14, #152, #208, #231, 3090 base, shantanu, tensorninja | Upstream merges + Responses fixes; auto shared prefix still open; NVFP4 illegal address; abliterated NVFP4 artifact (5090); no new commits | Nothing to do |

### Recommended order after this catch-up

1. DONE 2026-09-20: catch-up #3 deployed, soaked and fast-forwarded; `fork/fix/d1-planner-search-budget` goes with the push.
2. soohl INT8 dense prefill: kernel-bench + temp-0 quality gate on the 4090.
3. gzenz `3c0b4dc5` + `a8cdc1a6` tool-call robustness.
4. UDP structured-JSON design port (TEB `response_format`).
5. Unchanged: #152 auto shared prefix, tensorninja `e3a129c3`, host-KV design (gzenz `6df4f011` vs xkeyC).
6. DFlash2 only after a pi-mix bench with the new artifact.

## Inbound sweep 2026-09-07 (all remotes, upstream issues, forks of this repo, active forks of upstream)

Counts are commits absent from `rtx4090-port` at `6f327f49` (= production `catchup-6f327f49`).
Bodies were read; applicability was checked against this tree. The 3090 base, UDP, shantanu and
the probe remotes have nothing new since 2026-09-03.

### Upstream `neroued/master`: 84 commits since `ad0f3d38` (all 2026-09-06/07)

- **DFlash2** (dominant: `src/ops` 144 files, `tests/ops` 46, converter, model cards): a new
  speculative backend with companion draft weights. Needs a NEW artifact (`qwen3_8_27b.ninfer`
  sha `0634abb0…`, 19.03 GiB, minimum runtime `385b30ce`, `--spec dflash2 --draft-tokens 7`);
  verified on the 5090 only (`docs/maintainer/qwen3.8-27b-dflash2.md`). The current artifact
  keeps working (dflash2 weights bind optionally). Value unknown on Ada until benched against
  MTP3; the next catch-up merge will be large but mostly additive under `src/ops`.
- `03177b91` fix(runtime) preserve kv coverage during speculative terminal settlement: DFlash
  and DFlash2 page-boundary state; the MTP hunks are a `materialize_sequence_kv` ->
  `ensure_sequence_kv_mapped` rename only. Low for production.
- Generic perf worth a kernel-bench: `6d1da9ce` q5 linear add aggregate cliff, `22d8a1d3` w8
  vocabulary t64 route, `487f8977` sparse_moe (n/a, dense). Bench-first rule stands.

### Upstream open issues and PRs in our area

| Item | What | Value |
|---|---|---|
| **#210 issue + #211 PR** (ranxianglei) | Hard crash (device-side assert, GPU lockup) on real agent workloads: `commit_activation()` ran the paged-cache membership publish on stream 0, unordered vs the compute stream. **Our `logical_kv_store.h:894-901` has the exact vulnerable pattern** (adopted with `a2761ec1`). Fix = thread `cudaStream_t` through `activate()`, pass `device.stream` at both `program_impl.h` call sites (+6/-4). | **High, small; 0 crashes here in 2 days but the race is real** |
| **#176-#180 issues** (splickz, 09-04/05) | The materialization cluster we fought as D1: 5 ms search ceiling (#176 = our `fix/d1-planner-search-budget`), private cache permanently saturated across conversations (#177 = the 2-cell thrash), planner charges transition loss for unreachable checkpoints (#178), infeasible shared captures (#179), rolling retention proposal (#180). #181 (closed) = small interleaved requests evict the conversation prefix. | Read before D1b; our automatic anchors (`--auto-long-anchors`) and the JSONL `best_reuse_prompt_tokens` are evidence worth posting there |
| **#175 issue** (closed) | 3090 ran on the 5090 cost profile, prefill predicted 2.3x too low. **We run `prefill_source=generic-default transfer_source=generic-default` for `hardware_class=nvidia-geforce-rtx-4090-sm89`** (boot line), so every planner cost prediction on the 4090 is uncalibrated. `context_cost.cpp` accepts an external preset file. | Medium: calibrate a 4090 preset, then re-read the D1 planner numbers |
| #195 PR | Fall back to a preset of the same weights format when no (model, weights) row matches | Low once we ship our own preset |
| **#152 PR** (+65/-7, serve only) | Automatic shared-prefix write at the system/developer frontier; closes #142 (agent siblings miss the shared head without `prompt_cache_breakpoint`). pi sends no breakpoint. | Medium for multi-session pi; small port |
| #173 PR (danielfparkernz, +4131) | rk2v4-e8 re-port onto upstream's paged-KV engine, 208 B/head-token | Watch: if merged, our E8 layer can converge with upstream |
| #162/#163 (hecrj), #197 ignore_eos, #183 `--chat-template FILE`, #148 Responses API | serve conveniences | Low |

### Forks of this repository (13)

tensorninja `+31` (09-03: board energy attribution; `e3a129c3` restore diagnostics still the
pick), pxzleo `+35` (UI themes, n/a), KasoLu and alin-o new at `+0`. Nothing else moved.

### Active forks of upstream with own commits (278 forks; 20 pushed after 09-03 checked)

| Fork | What | Decision |
|---|---|---|
| **soohl/ninfer** `+2` (09-05/06, +6k lines) | An independent RTX 4090 port of upstream with E8 KV, 262K, vision, MTP3: **INT8 group-64 activations for the dense prefill = 3,548-3,684 tok/s at 8K vs 2,111 A16 (+68%)**, decode/MTP verify stay A16, artifact unchanged; cooperative grid from measured Ada occupancy; rejected FP8 PV and larger E8 query tiles; perplexity evidence in `docs/ada.md`. Our production prefills at ~2,000 tok/s. | **High. Bench-first + quality gate** (llm-eval + tool-eval-bench, temp-0 A/B): lossy INT8 prefill is a product decision. Port the prefill route only, not their E8 (ours is qualified) |
| **gzenz/ninfer** `+50` (11 stars, 5090/NVFP4, 3 agent sessions at 555K) | Host-KV safety net (`--host-kv-mib`, spill evicted continuations to a pinned host arena, restore on reuse); rewrite checkpoint captured at the turn boundary; checkpoint retained when state-slot reservation fails; **pre-check slot budget before creating a checkpoint** (`ff372161`, 1 file); OOM recovery in the worker loop; reasoning-effort tier mapping (Claude Code sends `high` -> 400 today, `839e5226`, 1 file); NVTX ranges for MTP decode; monitor dashboard. | Medium: the checkpoint-budget and effort-mapping fixes are one-file ports; the safety net competes with the xkeyC design port (compare before choosing) |
| **0xrjman/ninfer** `+6` | `cbf51152` stale-plan requests are dropped instead of killing the worker (which latched the engine into permanent 503 until restart; +206, 3 files, with a real-request regression); `54acc835` `state_footprint()` double-counted the retained fork source when read==write (entitlement invariant throw); `15f07fa4` on-site diag markers; Codex Responses extensions. | **High for availability**, medium size; the footprint bug lives in `b8786751` code we merged |
| BenWu `+99` | Two-device layer pipeline; context-cost preset misses surfaced | n/a (single GPU); the preset-miss logging is the #175 theme |
| cometkim `+39` | Own DFlash2 line (superseded by upstream's), width-8 int8 verify tile, `meta.n_ctx` on /v1/models | Low |
| kaushikvira `+10` | Ports of PRs #61, #160, Responses items, DFlash2 graft tool | Low |
| Gevil `+272` | `ADOPTION.md`: a curated, tiered adoption record of the whole fork ecosystem (T-numbered) | Read as an index, port nothing |
| aljazceru `+18` (08-20) | sm_86 A5000 port, INT4-G64 KV, pinned-host embedding offload | Low |
| troubadour-hell, plugmind-dev, Xtravaganz, sunnyyangyangyang | Windows, WSL bridge, syncs | n/a |

### Recommended order

1. **#211** stream-ordered membership publish: cherry-pick, ctest, deploy in the next window
   (crash class, 3 lines).
2. **soohl INT8 dense prefill**: kernel-bench + temp-0 quality A/B on the 4090; ship only if
   the quality gate holds (+68% prefill would take the 131K TTFT from 89 s to ~53 s).
3. **0xrjman** stale-plan drop + footprint fix (availability), with their regression test.
4. tensorninja `e3a129c3` restore diagnostics (unchanged from the 09-04 order).
5. gzenz one-file fixes (`ff372161` checkpoint budget pre-check, `839e5226` effort tiers) and
   #152 auto shared prefix.
6. Calibrate a 4090 context-cost preset (#175 class), then D1b / the #176-#180 cluster with
   `best_reuse_prompt_tokens` in hand; post our findings on #176/#177.
7. xkeyC `14faf879` + the host prefix cache design port vs gzenz's safety net: pick one.
8. Next upstream catch-up (DFlash2, 84 commits) only with the new artifact and a bench plan.

## Inbound sweep 2026-09-04 (all remotes and forks)

Survey of `neroued/master` (upstream), `Don-Chad/ninfer-3090` (the 3090 base),
`UDPSendToFailed/ninfer-4090`, the 13 forks of this repository, and the recently active forks
of upstream and of the 3090 base. Counts are commits absent from `rtx4090-port` at `4565c832`.
Bodies were read from the commits, not inferred from subjects; applicability was checked
against this tree.

### Upstream `neroued/master`: 20 commits since the 2026-09-01 catch-up

`5438b743` to `ad0f3d38`. Ranked by value to this fork:

| Commit | What it does | Value | Merge risk |
|---|---|---|---|
| `a140e7ae` preserve exact agent prefix reuse (43 files) | Makes NInfer's own accepted output an exact endpoint for an unmodified replay: the Frontend detects the reconstruction boundary, the Engine carries accepted-prefix metadata, the Program commits identity atomically. Preserves JSON member order in tool schemas and tool arguments end to end. Consumes Claude Code's `x-anthropic-billing-header` System block before identity construction. Raises default shared capacity to `max(max_concurrency, 4)`. Fewer turns diverge at all, which complements the automatic anchors. | High | engine_core.h, anthropic_messages.h |
| `b8786751` correct aliased state ownership (program_impl.h, 356 lines; 264 test lines) | Separates global physical occupancy from owner-exclusive resources and fixes borrowed-read lifetime for a Fork from a retained source. That is the path every long-anchor restore takes. | High, correctness | program_impl.h, heavy |
| `3b50962b`, `0c5d570c`, `719d56ef` tool-call frontend | Schema-guided typed conversion of Qwen's untyped parameter text; embedded `<parameter=...>` markup preserved with fallback to content when unbalanced; structure recognition separated from normalization. Relevant to pi's tool loop. Not a repair for the `<function=command>` slip, which falls back to content by design today. | Medium | frontend, docs |
| `550d0ac3` llama.cpp timing and prompt progress; `5f6d44e4` health reports engine readiness; `6e2786c5` readable operational logs | Each collides with a fork-local feature: our `timings` block, our `/health` port `60764d66`, our LOG-CONTRACT. Reconcile by hand. | Medium | serve, conflict-heavy |
| `e51b585c` respect cooperative launch capacity | Sources the SM count from `DeviceContext` and keeps the 5090 route table. The generic form of the open `7afc8e17` row; our gating-proj plan hardcodes 128 SMs. | High for other Ada cards, low for the 4090 | gdn kernels |
| `4ac73c47`, `21a0e85f`, `a2761ec1` KV cache | nvfp4 and k8v4 modes, fp16 V storage and PV compute, centralized format contracts. Ada has no FP4 tensor cores. fp16 V may move numerics and speed of every mode. | Low; bench first | same layer as our E8 modes |
| the rest | httplib 0.54.1, dflash vision, media bench, rmsnorm and MoE perf (the 27B is dense), fixtures, funding | Low | none |

### 3090 base `origin/master`: 36 commits of its own

- `5820660d` sum the unsplit GDN gating projection's K reduction pairwise. Numerics:
  `ninfer_gdn_gating_proj_test` exceeded the fp32 relative-L2 bound at T=3457 and T=4097. Our
  `bf16_gdn_gating_proj_gemm_mma.cuh` has no pairwise reduction and differs from their post-fix
  file. **High.** Run our test at those two T values first; port if it fails.
- `7afc8e17` resident-CTA budget from the runtime SM count: still open. Take the upstream form
  `e51b585c` instead.
- `249d96c3` stop aborting startup on a device-wide memory reading: check whether our startup
  has the same abort. Low.
- `aea729f3` stream tool-call whitespace linearly: small. Low.
- Everything else is MSVC and Windows portability, a NixOS flake, 3090 bench cohorts, the ECC
  startup warning, and docs. Not applicable.

### UDP `feat/rtx-4090-sm89-native`: 127 commits of its own

- Already handled: `dd5206f0` (ported), `05a88712` (closed), `8488278c` and `e2556b50` (not
  applicable), and `8bba5eb4` malformed UTF-8 repair, which this tree already has
  (`consume_generated_utf8`, `kUtf8Replacement`).
- `c15e0e9e` chunk KV snapshot staging into bounded page batches. Their save and restore
  allocated one buffer the size of the whole snapshot and ran out of memory at 280K on 24 GB.
  Our v3 serializer does not use that staging code; peak memory of a 5 GB save here is
  unmeasured. Low. Measure before porting.
- `5e76d11a` MTP restore stride: fixes their staging code. Our restores reuse MTP correctly in
  production (96 to 99% reuse after restore). Not applicable unless it reproduces.
- `378e0ad8` scale default max tokens to context size: policy; pi sets `max_tokens`. Low.
- About 30 perf commits from 09-01 and 09-02 (small-T tensor-core routing, W8 and Q5 wave-tax
  removal, GDN conv staging, decode grid alignment). Bench-first rule stands. Start with
  `45a5ae57`.

### Forks of this repository (13, compared against `rtx4090-port`)

| Fork | Ahead | What is there | Decision |
|---|---:|---|---|
| xkeyC/ninfer-4090 | 5 | `69e6ae19` chunked host prefix reuse: `--host-prefix-cache-mib`, content-hashed 64-token KV page groups plus GDN state blocks stored once across branches, recency-and-frequency eviction, restore streamed to pinned staging. `14faf879` prefix cache hits in `usage`. `60a5c687` stream retained snapshot blocks. Measured: four agents rotating to 200K on a 4090 with a 20 GiB host cache, median TTFT 4.36 s against 150 s cold, 627 requests. | **High.** This addresses our "three sessions on two cells thrash" directly. About 2,000 lines on a base 177 commits behind ours: a design port, not a cherry-pick, after the upstream merge. `14faf879` alone is small and lets pi display cache hits. |
| tensorninja/ninfer-4090 | 31 | `e3a129c3` record why a deferred continuation restore never returns: four `ContinuationDiagnostics` fields in the JSONL around the restore gate (4 files). The rest is LoRA training and a dashboard. | **High, small.** Fills our "a failed restore logs nothing" gap. Builds on their `c1e4eb1e` deferral semantics; check we have the equivalent. |
| pxzleo/ninfer-4090-48g | 35 | A web UI (throughput charts, themes) and 48 GB card support. | Not applicable to a 24 GB card; a UI is a separate product decision. |
| jomcgi | 2 | `chat_template_kwargs` aliases (ported as `6affed2e`), ghcr CI (declined). | Done. |
| IronKinoko | 4 | Windows PowerShell packaging. | Not applicable. |
| shantanusingh16 | 3 | `timings` (ported), llama-swap image, docs. | Done. |
| pefman | 1 | A docker serve script. | No. |
| KasoLu, aakash-chaddha, mhux2000, NeuronsReact, HermiG, MohitBurkule | 0 | | |

### Siblings worth knowing about

- `iamwavecut/ninfer-3090` `feat/kv-content-cache-upstream` (17 commits, 72 files, Aug 21 to 24,
  164 behind the 3090 base): a content-addressed host KV cache with prefix and trajectory
  restore, and coalescing of identical in-flight prompts. The same idea as xkeyC's block cache
  on an older base. Read for design, do not port.
- Other-hardware ports of upstream (gfx906, V100, RTX Pro 4000, Windows, C#): not applicable.

### Recommended order for the next session

1. Upstream catch-up merge to `ad0f3d38`. Items `a140e7ae`, `b8786751`, the tool-call trio and
   `e51b585c` ride along; reconcile the three serve collisions by hand; bench the KV-cache
   trio before accepting it. Same procedure as 2026-09-01: compile early, expect cluster-A
   conflicts in engine_core.h and program_impl.h, where the A2 persistence, D2, D3 and the
   automatic anchors all live.
2. `5820660d`: run `ninfer_gdn_gating_proj_test` at T=3457 and T=4097 on our kernel; port if
   it fails.
3. tensorninja `e3a129c3` restore diagnostics.
4. xkeyC `14faf879` cached tokens in `usage`. Evaluate the host prefix block cache as a design
   port afterwards, with the four-agent 200K rotation as the acceptance test.
5. The pending `fix/d1-planner-search-budget` rebase (D1b). Re-measure with the diag field
   first: `a140e7ae` and `b8786751` may change the planner picture.

## Upstream catch-up backlog (as of 2026-09-01)

`neroued/master` is 16 commits ahead of the `6b94b8c5` merge target, touching 309 files,
33 of which this fork has modified since the merge. Two clusters matter:

- **Logging replatform** (`4a1a2188` spdlog foundation, `5438b743` unify product
  operational logs). `5438b743` touches `src/serve/console_log.cpp`, `apps/serve/main.cpp`
  and `src/serve/http_server.cpp` - the same three files the deprecation warning and the
  `/health` fix just edited, so expect conflicts there. The log-format contract it
  threatens is OURS, not the dashboard's: `fleet-probe` filters containers by
  `SERVER_HINT = llama|llm|vllm|ollama|tabby`, which `ninfer-qwen38` / `ninfer-dev:runtime`
  does not match, so magnus's logs are never parsed (its card is built from HTTP endpoints).
  What does depend on the formats is every diagnosis this project runs: the boot
  KV-capacity line, `[req N] done ... reuse= cache= ttft=`, and the
  `slot save`/`slot restore`/`slot auto-save` lines that are the only production evidence
  that persistence works.
- **Runtime and context-cache fixes** (`da49c0d6` materialization sources excluded from
  pressure, `3d9fda22` reuse under bounded pressure search, `5e4bf313` bounded shared
  capture expansion, `138d76ae` resource scheduling ownership). These land in the same
  cluster A files the A2 catalog work rewrote, so expect the merge to conflict there
  again.

Also new: `neroued/feat/kv-nvfp4-k8v4` (`1e7b5877`, nvfp4 and k8v4 KV modes). Relevant to
the E8 non-port row below, which says to revisit if NVFP4 becomes the goal on the 5090.

## Deliberate non-ports

| Feature | Lives in | Decision |
|---|---|---|
| sm_89 attention retune (`ce50e995`) | 4090 | Architecture-specific by design |
| E8 lattice KV modes (`c3a6e5c4`, `ec56f922`, series) | 4090 | Declined for the 5090 on 2026-08-19: 32 GB fits the full 262K context on `int8`, so E8 would buy only the decode-at-depth gain. **Revisit if NVFP4 becomes the goal**: upstream PR #35 ports E8 to sm_120a, and NVFP4 cannot reach 262K on `int8` at all. Wait for that PR to merge rather than hand-porting it. See `docs/udp-fork-comparison.md` |
| `--vision-max-tokens` (`0c3d2bee`, `73b42127`) | 4090 | Open: the 5090 fits the legacy 32K scratchpad next to 262K + vision, so nothing forces the port |
| Single-token W8 column-store fix (`68e2d0be`) | 4090 | Not applicable: the 5090 tree's `w8_linear_add_gemm_splitk.cu` is the upstream variant without the vulnerable tail dispatch |
| NVFP4 weights profile | 5090 (upstream) | Ada has no FP4 tensor cores; the 4090 gates the A4 tests off instead |

## Long-term direction

The measured divergence between the trees is about 40 files once in-flight
ports land: roughly half architecture-specific kernels, half platform
configuration. The plan of record is to converge on one repository with two
architecture profiles (`sm_89` and `sm_120a` behind a CMake switch) and retire
the second tree to a deploy configuration. Until then, this ledger is the
source of truth for coverage.
