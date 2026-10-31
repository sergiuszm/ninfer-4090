# NInfer against llama.cpp on one RTX 4090

Measured 2026-08-15. Both engines ran Qwen3.8-27B on the same card on the same day. This
document records the configurations, the method, and the raw numbers behind the summary
tables in the [README](../README.md).

## Environment

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GiB, stock power limit (450 W) |
| Driver | 595.84 |
| NInfer | branch `rtx4090-port`, kernel retune commit `ce50e995`, CUDA 13.1 build |
| llama.cpp | build 10358 (`030ebb558`), image `local/llama.cpp:b10358-cuda-89` |
| NInfer artifact | `qwen3_8_27b.ninfer`, groupwise Q4/Q5/Q6 + W8 endpoints, 16.96 GiB |
| llama.cpp model | `Qwen3.8-27B-UD-Q4_K_XL.gguf`, 16.68 GiB |

The artifacts differ by about 2% in size, which favors llama.cpp on every
bandwidth-bound measurement.

## Configurations

NInfer ran the deployed serve configuration:

```
ninfer-serve qwen3_8_27b.ninfer --max-context 172032 --kv-capacity 172032 \
  --max-concurrency 1 --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking
```

llama.cpp prefill and no-speculation decode ran through `llama bench`:

```
llama bench -m Qwen3.8-27B-UD-Q4_K_XL.gguf -fa 1 -ctk q8_0 -ctv q8_0 \
  -ngl 99 -ub 1024 -b 4096 -p 2048 -n 32 -d 0,32768,65536,131072 -r 3
```

llama.cpp MTP decode ran through `llama-server` with the same quantization and batch
options plus:

```
--spec-type draft-mtp --spec-draft-n-max 3 -c 131584
```

The 131,584-token context is the practical MTP ceiling on this card: VRAM reaches
23.8 of 24 GiB. The 64K variant (`-c 65536`) uses 20.8 GiB. The deployed 144K llama.cpp
configuration cannot fit the MTP draft buffers.

## Method

- `llama bench` reports marginal rates: the pp2048 row at depth d measures 2,048 tokens
  of prefill after a d-token prefix, and tg32 measures 32 decoded tokens after the same
  prefix. Three repetitions per cell.
- NInfer rates come from the `ninfer-serve` `/metrics` counters
  (`llamacpp:prompt_tokens_total`, `llamacpp:prompt_seconds_total`,
  `llamacpp:tokens_predicted_total`, `llamacpp:tokens_predicted_seconds_total`,
  `ninfer:draft_*`), differenced around each request. The counters exclude
  prefix-cache hits, so every prefill number is computed work.
- llama.cpp server-side rates for the MTP runs come from its `/metrics` counters, same
  names plus `llamacpp:spec_decode_num_draft_tokens_total` and
  `llamacpp:spec_decode_num_drafts_total`.
- Prefill prompts are synthetic needle-in-a-haystack journals: repeated
  subject-verb-object filler lines from a seeded generator, one needle sentence at 80%
  depth, one retrieval question at the end. Both engines answered the needle exactly on
  every run. Prompt sizes measured by the server tokenizer: 1,880 / 31,254 / 63,619 (63,633
  for the llama.cpp tokenizer) / 88,001 / 127,925 (127,939) tokens.
- Decode-at-depth requests append a 400-word story instruction to the journal and
  generate 512 tokens. Code decode uses four fixed prompts (red-black tree in Python,
  LRU cache in Go, JSON-RPC client in TypeScript, Dijkstra in Rust) at 1,500 tokens each.
- All requests use temperature 0 with thinking disabled. NInfer takes a top-level
  `enable_thinking: false`; llama.cpp takes `chat_template_kwargs: {"enable_thinking": false}`.
- Full-prompt llama.cpp prefill times below 64K are integrated from the marginal rates by
  trapezoid; the two server-measured points show that integration underestimates the true
  server time by 2-4%.

## Raw results

### llama.cpp `llama bench` sweep (marginal rates)

| Depth | pp2048 tok/s | tg32 tok/s |
|---:|---:|---:|
| 0 | 3,023.84 ± 2.72 | 45.90 ± 0.22 |
| 32,768 | 2,327.17 ± 1.34 | 42.03 ± 0.29 |
| 65,536 | 1,865.64 ± 1.36 | 38.58 ± 0.18 |
| 131,072 | 1,335.69 ± 0.41 | 33.14 ± 0.12 |

### NInfer serve prefill (full-prompt averages)

| Prompt tokens | Time | Rate |
|---:|---:|---:|
| 1,880 | 0.925 s | 2,032 tok/s |
| 31,254 | 15.54 s | 2,012 tok/s |
| 63,619 | 34.41 s | 1,849 tok/s |
| 88,001 | 51.05 s | 1,724 tok/s |
| 127,925 | 81.95 s | 1,561 tok/s |

Marginal band rates from finite differences: 2,010 tok/s across 2K-32K, 1,714 across
32K-64K, 1,465 across 64K-88K, 1,292 across 88K-128K.

### llama.cpp server prefill (from the MTP runs)

| Prompt tokens | Time | Rate |
|---:|---:|---:|
| 63,633 | 28.72 s | 2,216 tok/s |
| 127,939 | 71.59 s | 1,787 tok/s |

### Decode, no speculation

| Depth | llama.cpp tg32 | NInfer (CLI, greedy) |
|---:|---:|---:|
| 0 | 45.9 tok/s | 50.5 tok/s |
| 128K | 33.1 tok/s | 39.6 tok/s |

### Decode, MTP against MTP (server, greedy, draft depth 3)

| Workload | llama.cpp `draft-mtp` | NInfer MTP3 |
|---|---:|---:|
| Code, shallow | 118.8 tok/s, 85.9% acceptance | 148.6 tok/s, 81.0% acceptance |
| Code, 128K depth | 63.5 tok/s, 85.0% | 104.0 tok/s, 72.9% |
| Prose, 64K depth | 55.5 tok/s, 45.3% | 85.9 tok/s, 44.6% |
| Prose, 128K depth | 42.3 tok/s, 45.4% | 77.1 tok/s, 45.6% |

Acceptance is per drafted token. The llama.cpp shallow code row drafted 5,031 tokens over
1,678 steps for 6,000 decoded tokens; the NInfer row drafted 5,241 over 1,747 steps for the
same 6,000. The 128K code rows use one identical request: the 128K journal followed by a Go
code task, 1,200 generated tokens. On that request llama.cpp held 85.0% acceptance against
72.9% for NInfer, and NInfer still decoded 64% faster - the strongest evidence that the gap
is verify-step time rather than draft quality.

The matching sweep on the RTX 5090 lives in the
[ninfer-5090 repository](https://github.com/sergiuszm/ninfer-5090)
(`docs/qwen38-rtx5090-vs-llamacpp.md`).

## Findings

1. llama.cpp leads prefill; the lead narrows with depth. Server against server: 20% on a
   64K prompt, 15% on a 128K prompt.
2. NInfer leads decode everywhere, and the lead grows with depth: +10% shallow and +20% at
   128K without speculation; +25% on shallow code, +55% at 64K, and +82% at 128K with MTP
   on both sides.
3. Draft acceptance is engine-independent: about 81-86% on structured code and about 45%
   on prose, on both engines. The MTP decode gap is therefore step time, not draft
   quality.
4. MTP costs llama.cpp its long-context headroom on 24 GiB: 131,584 tokens against
   172,032 for NInfer with speculation active on both.
5. The often-quoted llama.cpp shallow prefill rate (about 2.8-3.0k tok/s) is not
   representative of long prompts: its own marginal rate falls to 1,336 tok/s at 128K
   depth under production settings.

## History note

An earlier measurement recorded llama.cpp prefill at 2,334 tok/s at 64K depth under
unmatched settings. Under the production configuration above the matched number is
1,866 tok/s. The table in this document supersedes the old figure.

## 2026-08-17 refresh: E8 KV default

The NInfer default moved from INT8 KV at 172,032 tokens to `rk4v4-e8` (E8
Conway-Sloane 4-bit keys, 4-bit values) at the full native 262,144 tokens. This
section re-measures the NInfer side on that configuration. The llama.cpp rows above
are reused unchanged: same card, same driver, and the llama.cpp configuration did
not change between the two dates.

Method for this section: `ninfer-serve` on the deployed E8 profile, greedy, thinking
off, one request at a time. Prefill rates come from differenced
`llamacpp:prompt_tokens_total` / `prompt_seconds_total` counters on a cold prompt.
Decode rates come from differenced `tokens_predicted` counters on a second request
over the same prompt; compatible-prefix reuse recomputed zero prompt tokens in every
run. The corpora are deterministic generated prose and Go code. They are not the
2026-08-15 payloads, so acceptance-sensitive MTP rows are close to, but not
identical with, the older corpus; the acceptance column carries that context.

Prefill, full prompt, computed tokens only:

| Prompt tokens | E8 prefill | E8 wall | INT8 (08-15) | llama.cpp server (08-15) |
|---:|---:|---:|---:|---:|
| 29,479 | 2,027 tok/s | 14.5 s | 2,012 tok/s | - |
| 58,924 | 1,857 | 31.7 | 1,849 | 2,216 tok/s (28.7 s) |
| 117,791 | 1,581 | 74.5 | 1,561 | 1,787 tok/s (71.6 s) |
| 176,652 | 1,381 | 127.9 | - | no entry |
| 235,399 | 1,228 | 191.7 | - | no entry |

E8 prefill is at parity with INT8 or up to 1% above it at every measured depth: the
halved key traffic in attention offsets the dequantization work. The llama.cpp
prefill lead is 19% at 64K and 13% at 128K, and llama.cpp has no entry beyond its
144K deployed ceiling.

Decode without speculation:

| Depth | E8 | INT8 (08-15) | llama.cpp tg32 (08-15) |
|---:|---:|---:|---:|
| ~2K | 50.4 tok/s | 50.5 tok/s | 45.9 tok/s |
| ~118K | 42.1 | 39.6 | 33.1 (at 128K) |
| ~235K | 36.6 | - | no entry |

The mode inverts with depth. Shallow decode is weights-bound, and E8 pays a small
dequantization cost. Decode at depth is KV-bandwidth-bound, and the 4-bit cache
reads half the bytes: E8 beats INT8 by 6% at 118K and extends the curve to 235K.
Against llama.cpp the no-speculation lead grows from +10% shallow to +27% at 128K.

Decode with MTP3, acceptance in parentheses:

| Depth | E8 prose | E8 code | llama.cpp `draft-mtp` (08-15) |
|---:|---:|---:|---:|
| shallow | 94.6 (43.9%) at 29K | 142.9 (78.0%) | code 118.8 (85.9%) |
| ~60K | 86.1 (42.3%) | 129.0 (80.6%) | prose 55.5 (45.3%) |
| ~120K | 77.5 (41.6%) | 114.2 (77.8%) | prose 42.3 (45.4%), code 63.5 (85.0%) |
| ~177K | 69.5 (39.8%) | - | no entry |
| ~240K | 65.4 (41.1%) | 91.2 (72.1%) | no entry |

The prose rows reproduce the INT8 numbers within noise (86.1 against 85.9 at 64K,
77.5 against 77.1 at 128K) at matched acceptance, so the E8 keys cost nothing in
MTP drafting on prose. The code rows use the generated Go corpus; the 2026-08-15
code rows used a real journal payload with lower acceptance, so compare shapes, not
cells.

Findings update:

1. The MTP long-context gap doubled: llama.cpp fits 131,584 tokens with MTP buffers
   on this card, NInfer now serves 262,144 - the model's own limit.
2. NInfer decode at 128K leads llama.cpp by +27% without speculation (was +20% at
   INT8) and by +83% on prose with MTP on both sides (was +82%).
3. Every rate from 144K to 262,144 tokens is NInfer-only territory on this card.
