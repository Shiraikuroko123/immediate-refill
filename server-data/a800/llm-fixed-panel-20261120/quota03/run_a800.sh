#!/usr/bin/env bash
set -eu
cd /home/ubuntu/llm-fixed-panel-20261120/quota03
export VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
VENV=/home/ubuntu/llm-research-20261117/pilot-round5/runtime/vllm_env/bin/python
for i in $(seq 1 180); do
    if test -f model/model.safetensors; then break; fi
    sleep 5
done
test -f model/model.safetensors
"$VENV" -u tracks/quota/live.py --model model --selection tracks/quota/selection.json --out raw --jobs fixed-exact,fixed-buffer,fixed-eager
printf '0\n' > primary.exit
"$VENV" -u tracks/quota/live.py --model model --selection tracks/quota/selection.json --out raw --jobs fixed-b0,natural-exact,natural-buffer
printf '0\n' > secondary.exit
