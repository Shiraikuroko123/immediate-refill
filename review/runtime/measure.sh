#!/usr/bin/env bash
set -euo pipefail
py="$(command -v -- "$1")"
tools="$2"
artifact="$3"
model="$4"
attempt="$5"
study="$artifact/tracks/interference/prospective-device"
export VLLM_USE_V1=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
"$py" -B "$tools/preflight.py" "$artifact" --model "$model" > "$attempt/environment.json"
"$py" -m pip list --format=freeze > "$attempt/packages.txt"
export DEVICE_PYTHON="$py" DEVICE_MODEL="$model"
bash "$study/run.sh" "$attempt/session"
"$py" -B "$study/audit.py" "$attempt/session" > "$attempt/session/audit.json"
(cd "$attempt/session" && sha256sum calibrate.jsonl prediction.json validate.jsonl summary.json audit.json > sha256.txt)
