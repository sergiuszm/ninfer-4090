@echo off
setlocal
set "ROOT=%~dp0"
set "SERVER=%ROOT%ninfer-serve.exe"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%ROOT%models\qwen3_8_27b.ninfer"

if not exist "%SERVER%" (
  echo Missing %SERVER%
  echo Put this launcher beside the release files.
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing model: %MODEL%
  echo Run download-qwen38.bat first, or drag the model onto this launcher.
  exit /b 1
)

echo Starting Qwen3.8-27B Vision at http://127.0.0.1:8080/v1
echo Tested RTX 3090 profile: one request, 32K context, ReplaySSM and MTP3
rem --context-capacity is left at auto: startup takes the largest KV + State capacity that fits
rem after --memory-headroom (512MiB) and logs the estimated token capacity. It can be lower than
rem --max-context times --max-concurrency; check the startup log.
rem --vision-memory-budget 512MiB: estimate for about 8K Vision features per window; measure
rem max_window_features at startup before relying on it.
"%SERVER%" "%MODEL%" --host 127.0.0.1 --port 8080 --max-context 32768 --max-concurrency 1 --max-pending-requests 8 --prefill-chunk 512 --kv-dtype int8 --max-tokens 1024 --vision --vision-memory-budget 512MiB --spec mtp --draft-tokens 3 --proposal-head optimized
