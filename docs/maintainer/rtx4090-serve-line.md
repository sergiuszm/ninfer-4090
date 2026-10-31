# RTX 4090 production serve line on `1882c6ad`

This page is the serve line for the production Qwen3.8-27B server on the RTX 4090 (24 GB) after
catch-up #4. It replaces the line of 2026-10-06 (binary `3a27f32d`), flag by flag. It is deployed
since 2026-10-31 as package `c4b-fe4b4b724161` (port ledger, catch-up #4 "Deploy"). The values
that the line depends on are measured; see "Production readings".

## The line

```bash
docker create --name ninfer-qwen38 --label ninfer.deploy=c4b-fe4b4b724161 \
  --gpus all -p 8086:8080 --restart unless-stopped \
  -v /home/serchio/Models/neroued/Qwen3.8-27B-NInfer-v3:/models:ro \
  -v /home/serchio/ninfer-deploy/c4b-fe4b4b724161:/opt/ninfer:ro \
  -v /home/serchio/ninfer-deploy/logs:/logs \
  -v /home/serchio/ninfer-deploy/context-cost:/context-cost:ro \
  --entrypoint /opt/ninfer/apps/ninfer-serve ninfer-run:c4b \
  /models/qwen3_8_27b.ninfer --host 0.0.0.0 --port 8080 \
  --model-id Qwen3.8-27B --max-context 262144 --context-capacity auto \
  --memory-headroom 256MiB \
  --max-concurrency 2 --max-pending-requests 16 --prefill-chunk 1024 \
  --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 3 --proposal-head optimized \
  --preserve-thinking --vision --vision-memory-budget 512MiB \
  --context-cost-presets /context-cost/context_cost_presets_4090_int8.json \
  --prefill-activations int8 --tool-constraints auto \
  --request-log-jsonl /logs/requests.jsonl
```

The arguments after the image are the `serve-line.args` file of the package. `deploy-c4b.sh
switch` creates the container with `--restart no` and sets `unless-stopped` after all verify rows
pass. The `ninfer.deploy` label names the package that the container runs.

The model mount is the v3 directory (`qwen3_8_27b.ninfer`, 20.4 GB). The v2 file in
`Qwen3.8-27B-NInfer` is refused at artifact inspection.

`/home/serchio/ninfer-deploy/c4b-<sha12>` is the deploy package of one build: `apps/ninfer-serve`
and its libraries in `lib/` side by side, with `serve-line.args`, `deploy-c4b.sh`,
`rollback-c4b.sh`, `BUILD-INFO` and `SHA256SUMS`. Each build has its own directory. The old
binary `/home/serchio/ninfer-deploy/bin/ninfer-serve` stays unchanged for a rollback.
`ninfer-serve` loads `libninfer_core.so`, `libninfer_ops.so`, `libninfer_cpu.so`,
`libninfer_cpu_executor.so` and `libninfer_moe_cpu.so` from `$ORIGIN/../lib` (its RUNPATH). The
old swap of the single `ninfer-serve` file no longer works: without `../lib` the container exits
with "error while loading shared libraries". GPU window 1 mounted the whole build directory the
same way.

The `/slots` volume is gone. `--kv-dtype rk4v4-e8` needs the stage 2 KV modes and
`--prefill-activations int8` needs the stage 3 port. With `int8` KV, 262144 tokens do not fit (see
the measurements below).

## Flag by flag

| Old flag | New flag and value | Reason |
|---|---|---|
| model path, `--host`, `--port`, `--model-id`, `--max-context 262144` | unchanged | The artifact must be v3 |
| `--kv-capacity 262144` (tokens) | `--context-capacity auto` | Upstream removed the token form (`2705a728`). The capacity is now bytes per rank, shared by KV and State. `auto` takes the largest capacity that fits after the headroom. Measured: 285,274 tokens at the production line, more than 262144, so `auto` stays |
| (none) | `--memory-headroom 256MiB` | The default is 512 MiB and applies to auto and explicit capacity (`6dd2b1b4`). The catch-up #4 stage 2 boot (explicit capacity, no headroom) left 226 MiB free. Upstream's RTX 5090 example uses 256 MiB. Measured: 286 MiB free after boot and 284 MiB after a 128K prefill, so 256 MiB stays |
| `--max-concurrency 2`, `--max-pending-requests 16`, `--prefill-chunk 1024` | unchanged | |
| `--kv-dtype rk4v4-e8` | unchanged | The fork KV mode, re-homed in stage 2 |
| `--spec mtp --draft-tokens 3` | unchanged | |
| `--lm-head-draft` | `--proposal-head optimized` | Renamed in `4126ecf4` |
| `--preserve-thinking`, `--vision` | unchanged | |
| (none) | `--vision-memory-budget 512MiB` | Replaces the fork's Vision item cap of 8192 tokens (R12, `08ee5cff`). The default is 1 GiB, about the old 867 MB arena at 16384 tokens. Measured at 512 MiB: 8502 features per window, so the old limit of 8192 still fits |
| `--context-cost-presets FILE` | unchanged flag, new file `context_cost_presets_4090_int8.json` | Schema 3 now requires `h2h` in `transfer` (`2705a728`), so the schema 2 file of 2026-09-07 (`context_cost_presets_4090.json`) is refused. It stays next to the new file for a rollback. R14 assembled the new file from the window 3b calibration: one prefill entry for the INT8 signature `58642f67...b3c9` and the d2h, h2d, d2d and h2h transfer rows. The bench rejects the d2d fit on model shape; the file keeps that fit on purpose. h2h is upstream's generic value. Check: `engine.context_cost` in the `server_start` record shows external prefill and transfer sources |
| `--prefill-activations int8` | unchanged | The fork INT8 prefill, re-homed in stage 3 |
| (none) | `--tool-constraints auto` | The fork server default (see "Defaults that the line keeps"). The line writes it out, so the line documents plan decision 2 |
| `--request-log-jsonl /logs/requests.jsonl` | unchanged | The request log schema is 31 now (20 before) |
| `--slot-save-path /slots`, `--auto-save-evicted` | removed | Slot persistence is dropped (plan decision 1). `GET /slots` is read-only and carries no `session_digest`, so the pi slot extension turns itself off |
| `--max-private-continuations 2`, `--max-shared-prefixes 1` | removed | Upstream removed them before `81c8ce09`. The context cache bounds retained sessions by bytes |
| `--auto-long-anchors 2` | removed | R4 is dropped (plan decision 4) |
| (none) | `--tolerant-tool-calls` off, not on the line | Fork PR #14, re-ported to the streaming decoder (merged in `1fa19ad8`). It acts only on the free route: requests with `tool_constraints` `auto`, which pi gets from the fork server default `--tool-constraints auto` (plan decision 2). A file write that quotes `\n</parameter>` then completes exactly instead of failing with `invalid_tool_output`; text after the last call is dropped. Off at the first deploy (user decision). It goes on with a container recreate after one or two days of `invalid_tool_output` counts on the new line |

## Defaults that the line keeps

| Flag | Default | Note |
|---|---|---|
| `--host-cache-capacity` | 8 GiB plus 8 State images | Same shape as the old host tier (8 GiB of host KV, 8 host State slots). The default is not keyed to the RTX 5090; it applies to every target with discrete memory. Measured: 9821569024 bytes (8 GiB plus 8 State blocks of 153954304 bytes) of the 27.3 GiB host RAM of magnus, so the default stays |
| `--pending-timeout` | 30 s | A request that waits longer in the queue fails before its stream starts: 503 on Chat and Responses, 504 `timeout_error` on Messages (`45f74287`) |
| `--stats-interval` | 5 s | Was `--log-stats-interval-ms` |
| `--thinking-control` | model text | Not on the production line (fork PR #18) |
| `--tool-constraints` | `auto` (fork default; upstream `basic`) | Plan decision 2. pi and llm-eval send no `tool_constraints`, so the server default sets the mode of all their tool calls. `basic` cuts a raw string argument at the first `\n</parameter>` and still returns HTTP 200. Under `auto`, an ordinary non-strict call keeps the nested form exact, as production does. Check: `server.default_tool_constraints` in the `server_start` record is `auto` |

## Measurements

GPU window 1, 2026-10-31: the stage 1 tree on the RTX 4090, v3 artifact, `--kv-dtype int8`,
`--max-concurrency 2`, MTP 3, Vision.

| Configuration | Result |
|---|---|
| `--max-context 262144`, `auto`, default headroom (512 MiB) and Vision budget (1 GiB) | Startup fails: "Device budget cannot hold model weights and required execution resources" |
| `--max-context 131072`, `auto`, `--memory-headroom 256MiB`, `--vision-memory-budget 512MiB` | Ready in about 6 s. Budget 23.1 GiB, runtime 5.85 GiB, device context 5.31 GiB, KV ~146959 tokens (1.1 contexts of 131072). Startup free 285.7 MiB (`cuda_free_bytes` 299565056); `nvidia-smi` shows 286 MiB free. Host context 9.15 GiB pinned (the default; the host has 28 GiB RAM) |
| Vision at `--vision-memory-budget 512MiB` (same boot) | `budget_bytes` 536870912, `max_window_features` 8502, `encoder_workspace_bytes` 449756160, `output_capacity_bytes` 87060480 (10 KiB per feature), `input_staging_bytes` 105867264 |

## Production readings

The production line (`rk4v4-e8`, INT8 prefill, 262144, 2 lanes, MTP 3, Vision 512 MiB, headroom
256 MiB). Sources: GPU windows 3a and 3b on the merged tree `2c1ecd25`, the R14 check in window 4,
and the deploy switch (`~/ninfer-deploy/rollback/switch.log` on magnus, 2026-10-31 18:50 UTC)
with its `server_start` record. Read the values again when a flag of the line changes.

| Value | Where to read it | Reading | Source | Decides |
|---|---|---|---|---|
| Estimated context token capacity | boot log line `context \| device ... \| KV ~N tokens`, or `engine.context_token_capacity_estimate` in the `server_start` record of the request log | 285,274 tokens | windows 3a and 3b, switch.log | `--context-capacity auto` stays (262144 or more) |
| Free VRAM | `ninfer_cuda_memory_free_bytes` on `/metrics`, or `nvidia-smi` | 299565056 bytes at startup; `nvidia-smi` 286 MiB free after boot (23,798 MiB used) and 284 MiB after a 128K prefill. The old line had 640 MiB | windows 3a and 3b, switch.log | `--memory-headroom 256MiB` stays |
| `max_window_features` | `resources.ranks[0].resources.vision.max_window_features` in the `server_start` record of the request log | 8502 at 512 MiB, more than the old cap of 8192 | windows 1 and 3a, switch.log | `--vision-memory-budget 512MiB` stays |
| Host context cache | `engine.context_cache.host_capacity_bytes` in the `server_start` record of the request log | 9821569024 bytes (8 GiB plus 8 State blocks of 153954304 bytes); magnus has 27.3 GiB RAM | window 3a, the production `server_start` | `--host-cache-capacity` stays at the default |
| Host reservation | `ninfer_memory_host_reservation_bytes` on `/metrics`: the media live capacity when Vision is on, else 0 | 2147483648 bytes (the 2 GiB default) | window 3a | `--media-live-capacity` stays at the default |
| Context cost preset | `engine.context_cost` in the `server_start` record | prefill and transfer sources external, `prefill_signature` `58642f67...b3c9`, `hardware_class` `nvidia-geforce-rtx-4090-sm89` | window 4 (R14), switch.log | `context_cost_presets_4090_int8.json` (R14) |

## Runtime image

The deployed image is `ninfer-run:c4b`. Checked on 2026-10-31 against the `NEEDED` entries of `ninfer-serve` and its `lib/` libraries:
`ninfer-dev:runtime` resolves all of them except `libhwloc.so.15` and `liburing.so.2`. The FFmpeg
libraries, `libcurl.so.4`, `libjpeg.so.8` and `libcudart.so.13` are present. The runtime image
needs `libhwloc15` and `liburing2` in addition. Do not use the build image `ninfer-dev:c4b` as
the runtime image. Its stub `libcuda.so.1` shadows the driver.

The default seccomp profile of Docker blocks `io_uring`. The direct read queue then fails with
"initialize direct read queue: Operation not permitted" (the tests print it) and the loader falls
back without a message. Weights still load at 5.5 GB/s (3.1 s), so the line keeps the default
profile.
