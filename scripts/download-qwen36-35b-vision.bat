@echo off
setlocal
set "ROOT=%~dp0"
set "MODEL_DIR=%ROOT%models"
set "MODEL=%MODEL_DIR%\qwen3_6_35b_a3b.ninfer"
rem Pinned to the container-v3 revision of 2026-09-15 (v3 artifact with the updated built-in chat
rem template). The engine rejects container v2; convert an older file with
rem tools/upgrade_ninfer_v2_to_v3.py instead of downloading again.
set "REVISION=ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6"
set "EXPECTED_SHA256=3e33297645dc33557751be1a3c407a74ed7c00f34909b5d4e8cfdce91b3dbe84"

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
echo Downloading the RTX 3090-compatible Qwen3.6-35B-A3B vision model...
curl.exe -L -C - --fail --output "%MODEL%" "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/%REVISION%/qwen3_6_35b_a3b.ninfer"
if errorlevel 1 (
  echo Download failed. Run this file again to resume.
  exit /b 1
)
echo Expected SHA-256: %EXPECTED_SHA256%
echo Verify with: certutil -hashfile "%MODEL%" SHA256
echo Model ready: %MODEL%
