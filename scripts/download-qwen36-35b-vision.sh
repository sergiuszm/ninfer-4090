#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${NINFER_MODEL_DIR:-$root/models}"
model="$model_dir/qwen3_6_35b_a3b.ninfer"

mkdir -p -- "$model_dir"
# Pinned to the container-v3 revision of 2026-09-15 (v3 artifact with the updated built-in chat
# template). The engine rejects container v2; convert an older file with
# tools/upgrade_ninfer_v2_to_v3.py instead of downloading again.
revision='ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6'
expected_sha256='3e33297645dc33557751be1a3c407a74ed7c00f34909b5d4e8cfdce91b3dbe84'

printf '%s\n' 'Downloading the RTX 3090-compatible Qwen3.6-35B-A3B vision model...'
if ! curl -L -C - --fail --output "$model" \
  "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/$revision/qwen3_6_35b_a3b.ninfer"; then
  printf '%s\n' 'Download failed. Run this script again to resume.' >&2
  exit 1
fi
if [[ -z "${NINFER_SKIP_SHA256:-}" ]] && command -v sha256sum >/dev/null 2>&1; then
  printf '%s\n' 'Verifying SHA-256...'
  actual_sha256="$(sha256sum -- "$model" | cut -d' ' -f1)"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    printf 'SHA-256 mismatch: expected %s, got %s\n' "$expected_sha256" "$actual_sha256" >&2
    printf 'Delete %s and run this script again (a resumed older file does not match).\n' "$model" >&2
    exit 1
  fi
fi
printf 'Model ready: %s\n' "$model"
