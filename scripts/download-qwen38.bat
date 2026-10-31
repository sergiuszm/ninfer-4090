@echo off
setlocal
set "ROOT=%~dp0"
set "MODEL_DIR=%ROOT%models"
set "MODEL=%MODEL_DIR%\qwen3_8_27b.ninfer"
rem Pinned to a container-v2 revision: the Hugging Face main revision moved to container v3 on
rem 2026-09-15, which this engine's reader rejects (artifact magic is not NInfer v2).
set "REVISION=3526913004b1"
set "EXPECTED_SHA256=eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e"

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
