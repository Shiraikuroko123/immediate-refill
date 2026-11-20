#!/usr/bin/env bash
set -euo pipefail
if (( $# < 3 || $# > 4 )); then
    echo "Usage: bash run.sh EXTRACTED_ARTIFACT QWEN_DIRECTORY NEW_ATTEMPT [GPU_INDEX_OR_UUID]" >&2
    exit 2
fi
tools="$(cd -- "$(dirname -- "$0")" && pwd)"
artifact="$(realpath -- "$1")"
model="$(realpath -- "$2")"
attempt="$(realpath -m -- "$3")"
gpu="${4:-0}"
test -f "$artifact/tracks/interference/prospective-device/freeze.json"
test -r "$model/model.safetensors"
test ! -e "$attempt" || { echo "Choose a new attempt directory. Existing evidence is kept." >&2; exit 2; }
uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
gpu_name="$(nvidia-smi -i "$uuid" --query-gpu=name --format=csv,noheader)"
[[ "$gpu_name" == "NVIDIA A800-SXM4-80GB" ]] || { echo "Reference device: NVIDIA A800-SXM4-80GB. Selected: $gpu_name" >&2; exit 2; }
active="$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader)"
[[ -z "$active" ]] || { echo "Another compute process is using the selected GPU." >&2; exit 2; }
mkdir -- "$attempt"
nvidia-smi -i "$uuid" --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.mem --format=csv > "$attempt/gpu-before.csv"
if [[ -n "${DEVICE_PYTHON:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$uuid"
    bash "$tools/measure.sh" "$DEVICE_PYTHON" "$tools" "$artifact" "$model" "$attempt"
else
    image="docker.io/vllm/vllm-openai@sha256:6cf9808ca8810fc6c3fd0451c2e7784fb224590d81f7db338e7eaf3c02a33d33"
    image_id="sha256:d1a544bfefdcd51fb42a9cc64d29df092127a7f1706f7a284aac26e5c48c468d"
    if ! docker image inspect "$image_id" >/dev/null 2>&1; then
        docker pull --platform linux/amd64 "$image"
    fi
    [[ "$(docker image inspect -f '{{.Id}}' "$image_id")" == "$image_id" ]]
    printf '%s\n%s\n' "$image" "$image_id" > "$attempt/image.txt"
    docker run --rm --pull=never --platform linux/amd64 --network=none --gpus "device=$uuid" \
      --shm-size=8g --user "$(id -u):$(id -g)" -e USER=reviewer -e LOGNAME=reviewer -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/xdg-cache \
      -v "$tools:/tools:ro" -v "$artifact:/artifact:ro" -v "$model:/model:ro" -v "$attempt:/attempt" \
      --entrypoint /bin/bash "$image_id" /tools/measure.sh python3 /tools /artifact /model /attempt
fi
