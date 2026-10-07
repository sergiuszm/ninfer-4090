#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${NINFER_MODEL_DIR:-$root/models}"
model="$model_dir/qwen3_8_27b.ninfer"

mkdir -p -- "$model_dir"
# Pinned to a container-v2 revision: the Hugging Face main revision moved to container v3 on
# 2026-09-15, which this engine's reader rejects (artifact magic is not NInfer v2).
revision='3526913004b1'
expected_sha256='eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e'

printf '%s\n' "Downloading Qwen3.8-27B NInfer model (revision $revision)..."
if ! curl -L -C - --fail --output "$model" \
  "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/$revision/qwen3_8_27b.ninfer"; then
  printf '%s\n' 'Download failed. Run this script again to resume.' >&2
  exit 1
fi
if [[ -z "${NINFER_SKIP_SHA256:-}" ]] && command -v sha256sum >/dev/null 2>&1; then
  printf '%s\n' 'Verifying SHA-256...'
  actual_sha256="$(sha256sum -- "$model" | cut -d' ' -f1)"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    printf 'SHA-256 mismatch: expected %s, got %s\n' "$expected_sha256" "$actual_sha256" >&2
    exit 1
  fi
fi
printf 'Model ready: %s\n' "$model"
