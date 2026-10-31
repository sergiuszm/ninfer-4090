# RTX 4090 production serve line on `1882c6ad`

This page is the serve line for the production Qwen3.8-27B server on the RTX 4090 (24 GB) after
catch-up #4. It replaces the line of 2026-10-06 (binary `3a27f32d`), flag by flag. Values marked
**MEASURE** are estimates. Read them from the first GPU boot and update this page.

## The line

```bash
docker create --name ninfer-qwen38 --gpus all -p 8086:8080 --restart unless-stopped \
  -v /home/serchio/Models/neroued/Qwen3.8-27B-NInfer:/models \
  -v /home/serchio/ninfer-deploy/bin/ninfer-serve:/opt/ninfer/ninfer-serve:ro \
  -v /home/serchio/ninfer-deploy/logs:/logs \
  -v /home/serchio/ninfer-deploy/context-cost:/context-cost:ro \
  <runtime image> \
  /opt/ninfer/ninfer-serve /models/qwen3_8_27b.ninfer --host 0.0.0.0 --port 8080 \
  --model-id Qwen3.8-27B --max-context 262144 --context-capacity auto \
  --memory-headroom 256MiB \
  --max-concurrency 2 --max-pending-requests 16 --prefill-chunk 1024 \
  --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 3 --proposal-head optimized \
  --preserve-thinking --vision --vision-memory-budget 512MiB \
  --context-cost-presets /context-cost/context_cost_presets_4090.json \
  --prefill-activations int8 \
  --request-log-jsonl /logs/requests.jsonl
```

The `/slots` volume is gone. `--kv-dtype rk4v4-e8` needs the stage 2 KV modes and
`--prefill-activations int8` needs the stage 3 port. Until both land, a boot of the stage 6 tree
alone uses `--kv-dtype int8` and leaves out `--prefill-activations`. With `int8` KV, 262144 tokens
do not fit (see the measurements below), so such a boot also needs a smaller `--max-context`.

## Flag by flag

| Old flag | New flag and value | Reason |
|---|---|---|
| model path, `--host`, `--port`, `--model-id`, `--max-context 262144` | unchanged | The artifact must be v3 |
| `--kv-capacity 262144` (tokens) | `--context-capacity auto` | Upstream removed the token form (`2705a728`). The capacity is now bytes per rank, shared by KV and State. `auto` takes the largest capacity that fits after the headroom. **MEASURE**: the boot log shows the estimated token capacity (`e5e18b97`). It must be 262144 or more |
| (none) | `--memory-headroom 256MiB` | The default is 512 MiB and applies to auto and explicit capacity (`6dd2b1b4`). The catch-up #4 stage 2 boot (explicit capacity, no headroom) left 226 MiB free. Upstream's RTX 5090 example uses 256 MiB. **MEASURE**: free VRAM after warmup |
| `--max-concurrency 2`, `--max-pending-requests 16`, `--prefill-chunk 1024` | unchanged | |
| `--kv-dtype rk4v4-e8` | unchanged | The fork KV mode, re-homed in stage 2 |
| `--spec mtp --draft-tokens 3` | unchanged | |
| `--lm-head-draft` | `--proposal-head optimized` | Renamed in `4126ecf4` |
| `--preserve-thinking`, `--vision` | unchanged | |
| (none) | `--vision-memory-budget 512MiB` | Replaces the fork's Vision item cap of 8192 tokens (R12, `08ee5cff`). The default is 1 GiB, about the old 867 MB arena at 16384 tokens. Measured at 512 MiB: 8502 features per window, so the old limit of 8192 still fits |
| `--context-cost-presets FILE` | unchanged flag, new file | Schema 3 now requires `h2h` in `transfer` (`2705a728`), so the file of 2026-09-07 is refused. **MEASURE**: regenerate it on the RTX 4090 (R14) |
| `--prefill-activations int8` | unchanged | The fork INT8 prefill, re-homed in stage 3 |
| `--request-log-jsonl /logs/requests.jsonl` | unchanged | The request log schema is 31 now (20 before) |
| `--slot-save-path /slots`, `--auto-save-evicted` | removed | Slot persistence is dropped (plan decision 1). `GET /slots` is read-only and carries no `session_digest`, so the pi slot extension turns itself off |
| `--max-private-continuations 2`, `--max-shared-prefixes 1` | removed | Upstream removed them before `81c8ce09`. The context cache bounds retained sessions by bytes |
| `--auto-long-anchors 2` | removed | R4 is dropped (plan decision 4) |

## Defaults that the line keeps

| Flag | Default | Note |
|---|---|---|
| `--host-cache-capacity` | 8 GiB plus 8 State images | Same shape as the old host tier (8 GiB of host KV, 8 host State slots). The default is not keyed to the RTX 5090; it applies to every target with discrete memory. **MEASURE**: `ninfer_memory_host_reservation_bytes` against the host RAM |
| `--pending-timeout` | 30 s | A request that waits longer in the queue fails before its stream starts: 503 on Chat and Responses, 504 `timeout_error` on Messages (`45f74287`) |
| `--stats-interval` | 5 s | Was `--log-stats-interval-ms` |
| `--thinking-control` | model text | Not on the production line (fork PR #18) |

## Measurements

GPU window 1, 2026-10-31: the stage 1 tree on the RTX 4090, v3 artifact, `--kv-dtype int8`,
`--max-concurrency 2`, MTP 3, Vision.

| Configuration | Result |
|---|---|
| `--max-context 262144`, `auto`, default headroom (512 MiB) and Vision budget (1 GiB) | Startup fails: "Device budget cannot hold model weights and required execution resources" |
| `--max-context 131072`, `auto`, `--memory-headroom 256MiB`, `--vision-memory-budget 512MiB` | Ready in about 6 s. Budget 23.1 GiB, runtime 5.85 GiB, device context 5.31 GiB, KV ~146959 tokens (1.1 contexts of 131072). Startup free 285.7 MiB (`cuda_free_bytes` 299565056); `nvidia-smi` shows 286 MiB free. Host context 9.15 GiB pinned (the default; the host has 28 GiB RAM) |
| Vision at `--vision-memory-budget 512MiB` (same boot) | `budget_bytes` 536870912, `max_window_features` 8502, `encoder_workspace_bytes` 449756160, `output_capacity_bytes` 87060480 (10 KiB per feature), `input_staging_bytes` 105867264 |

The production values need the same readings with `rk4v4-e8` at 262144.

## Values to measure on the first boot

| Value | Where to read it | Decides |
|---|---|---|
| Estimated context token capacity | boot log line `context \| device ... \| KV ~N tokens`, or `engine.context_token_capacity_estimate` in the `server_start` record of the request log | `--context-capacity`: keep `auto` if 262144 or more |
| Free VRAM after warmup | `ninfer_cuda_memory_free_bytes` on `/metrics` | `--memory-headroom` |
| `max_window_features` | `resources.ranks[0].resources.vision.max_window_features` in the `server_start` record of the request log | `--vision-memory-budget`: 8502 at 512 MiB (window 1). Read it again if the budget changes |
| Host reservation | `ninfer_memory_host_reservation_bytes` on `/metrics` | `--host-cache-capacity` |
| Context cost preset | `bench/context_cost` on the v3 artifact | the `--context-cost-presets` file |

## Runtime image

The binary links libhwloc, liburing and libjpeg. The runtime image on the host needs
`libhwloc15`, `liburing2` and `libjpeg-turbo8` beside the packages of `ninfer-dev:runtime`. Do not
use the build image `ninfer-dev:c4b` as the runtime image. Its stub `libcuda.so.1` shadows the
driver.

The default seccomp profile of Docker blocks `io_uring`. The direct read queue then fails with
"initialize direct read queue: Operation not permitted" (the tests print it) and the loader falls
back without a message. Weights still load at 5.5 GB/s (3.1 s), so the line keeps the default
profile.
