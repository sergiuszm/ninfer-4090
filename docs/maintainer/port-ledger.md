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
The bench refuses a partial preset; if d2d never fits, hand-assemble one from the accepted
components in `context_cost_4090*.json` (schema: `context_cost.cpp` `parse_context_cost_presets`).

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
