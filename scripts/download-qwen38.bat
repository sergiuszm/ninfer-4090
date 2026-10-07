@echo off
setlocal
set "ROOT=%~dp0"
set "MODEL_DIR=%ROOT%models"
set "MODEL=%MODEL_DIR%\qwen3_8_27b.ninfer"
rem Pinned to the container-v3 revision of 2026-09-15 (v3 artifact with the updated built-in chat
rem template). The engine rejects container v2; convert an older file with
rem tools/upgrade_ninfer_v2_to_v3.py instead of downloading again.
set "REVISION=1cbd84e7221e"
set "EXPECTED_SHA256=81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da"

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
echo Downloading Qwen3.8-27B NInfer model (revision %REVISION%)...
curl.exe -L -C - --fail --output "%MODEL%" "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/%REVISION%/qwen3_8_27b.ninfer"
if errorlevel 1 (
  echo Download failed. Run this file again to resume.
  exit /b 1
)
echo Expected SHA-256: %EXPECTED_SHA256%
echo Verify with: certutil -hashfile "%MODEL%" SHA256
echo Model ready: %MODEL%
