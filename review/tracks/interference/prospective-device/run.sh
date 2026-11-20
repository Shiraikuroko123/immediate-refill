#!/usr/bin/env bash
set -euo pipefail
export VLLM_USE_V1=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
here="$(cd -- "$(dirname -- "$0")" && pwd)"
py="${DEVICE_PYTHON:-/home/ubuntu/llm-research-20261117/pilot-round5/runtime/vllm_env/bin/python}"
model="${DEVICE_MODEL:-/home/ubuntu/llm-fixed-panel-20261120/model}"
out="${1:?supply a new output directory}"
mkdir -- "$out"
exec > >(tee -a "$out/console.log") 2>&1
timeout --kill-after=30s 10m "$py" -B "$here/experiment.py" calibrate --out "$out" --model "$model"
sha256sum "$out/prediction.json"
timeout --kill-after=30s 20m "$py" -B "$here/experiment.py" validate --out "$out" --model "$model"
"$py" -B "$here/experiment.py" analyze --out "$out" > "$out/summary.json"
