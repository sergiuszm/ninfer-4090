#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
server="${NINFER_SERVER:-$root/ninfer-serve}"
if [[ ! -x "$server" && -x "$root/../build-sm86/apps/ninfer-serve" ]]; then
  server="$root/../build-sm86/apps/ninfer-serve"
fi
model="${1:-${NINFER_MODEL_DIR:-$root/models}/qwen3_6_35b_a3b.ninfer}"

if [[ ! -x "$server" ]]; then
  printf 'Missing executable: %s\n' "$server" >&2
  printf '%s\n' 'Set NINFER_SERVER or build build-sm86/apps/ninfer-serve.' >&2
  exit 1
fi
if [[ ! -f "$model" ]]; then
  printf 'Missing model: %s\n' "$model" >&2
  printf '%s\n' 'Run download-qwen36-35b-vision.sh or pass the model path.' >&2
  exit 1
fi

printf '%s\n' 'Starting Qwen3.6-35B-A3B Vision at http://127.0.0.1:8080/v1'
printf '%s\n' 'Safe RTX 3090 profile: one request, 32K context, vision enabled, MTP disabled'
# --context-capacity is left at auto: startup takes the largest KV + State capacity that fits
# after --memory-headroom (512MiB) and logs the estimated token capacity. It can be lower than
# --max-context times --max-concurrency; check the startup log.
# --vision-memory-budget 512MiB: measured as 8502 features per window for Qwen3.8-27B only; not
# measured for this 35B model. Add --request-log-jsonl FILE and read
# resources.ranks[0].resources.vision.max_window_features in its server_start record.
exec "$server" "$model" \
  --host 127.0.0.1 --port 8080 \
  --max-context 32768 \
  --max-concurrency 1 --max-pending-requests 8 \
  --prefill-chunk 512 --kv-dtype int8 --max-tokens 512 \
  --vision --vision-memory-budget 512MiB --no-thinking --temperature 0.2
