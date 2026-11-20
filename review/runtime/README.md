# Rerun the prospective Qwen experiment on a new Linux A800

This recipe uses the frozen experiment, its independent analysis and a public vLLM image. The measured reference used Linux, CPython 3.10.12, NVIDIA A800-SXM4-80GB, driver 550.144.03, PyTorch 2.6.0+cu124, CUDA runtime 12.4, vLLM 0.8.5 V0 and the Flash Attention backend. The exact original operating-system image and complete GPU `pip freeze` are unavailable. Direct dependencies in the archived CPU requirements file describe the analysis environment. The public GPU image uses Python 3.12 and the same vLLM release. Other library versions and clocks on a new host may affect timings. [Original environment and calibration](../../../tracks/interference/prospective-device/session01/calibrate.jsonl)

Prepare an x86-64 Linux host with one idle A800-SXM4-80GB, a working CUDA 12.4-compatible NVIDIA driver, NVIDIA Container Toolkit, Docker, Bash and GNU coreutils. Allow space for the expanded container image in Docker storage. Its compressed layers total approximately 9.04 GB. The host running Docker must be able to pull the public image once or import a local copy. The measurement container itself runs offline. The project directory and checkpoint are mounted read-only. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) · [Official vLLM 0.8.5 install guide](https://docs.vllm.ai/en/v0.8.5/getting_started/installation/gpu.html)

Extract the author archive and obtain the verified Qwen checkpoint using the accompanying model instructions. Run the command below with the path containing the extracted `tracks/` directory, the checkpoint directory, and a new output directory whose parent exists. Set the last argument to the GPU index or UUID from `nvidia-smi -L`.

```bash
bash runtime/run.sh \
  /path/to/extracted-artifact /path/to/Qwen2.5-1.5B-Instruct /path/to/new-attempt 0
```

The runner checks the four frozen source files, checkpoint, GPU, engine, CUDA and PyTorch versions. It records the selected driver, clocks, Python and installed packages. The original experiment runner starts one engine process for calibration, writes the signed-direction prediction to `session/prediction.json`, then starts a second process for validation. The new measurements appear in `session/calibrate.jsonl` and `session/validate.jsonl`. `session/summary.json` and `session/audit.json` give the paired wall measurements and the prespecified `joint_transfer_passed` decision. Compare the new intervals and direction against the original results. A failed or inconclusive prospective prediction is a valid replication outcome. `session/sha256.txt` fingerprints the new files.

The reference experiment took around 144 seconds from calibration launch to completion of analysis after environment setup. The original runner allows up to ten minutes for calibration and twenty minutes for validation. Unpacking the container and downloading the checkpoint are separate. When Docker is unavailable but a suitable Linux environment already exists, set `DEVICE_PYTHON` to its Python executable and use the same command. The runner checks the recorded core versions before measurement.

```bash
DEVICE_PYTHON=/path/to/venv/bin/python bash runtime/run.sh \
  /path/to/extracted-artifact /path/to/Qwen2.5-1.5B-Instruct /path/to/new-native-attempt 0
```

The runner pulls `docker.io/vllm/vllm-openai@sha256:6cf9808ca8810fc6c3fd0451c2e7784fb224590d81f7db338e7eaf3c02a33d33` and verifies the immutable image configuration ID. To retain the software even if the registry becomes unavailable, an operator can obtain and export it ahead of time. After importing on the new host, the runner finds the image by ID.

```bash
docker pull vllm/vllm-openai@sha256:6cf9808ca8810fc6c3fd0451c2e7784fb224590d81f7db338e7eaf3c02a33d33
docker tag sha256:d1a544bfefdcd51fb42a9cc64d29df092127a7f1706f7a284aac26e5c48c468d llm-clock-review:v0.8.5
docker save -o vllm-0.8.5-amd64.tar llm-clock-review:v0.8.5
# On the new host
docker load -i vllm-0.8.5-amd64.tar
```

The supplied package includes the commands and content identifiers for the public image. We checked the Docker registry manifest, image configuration digest, source fingerprints and shell syntax on the packaging host. A second GPU host has yet to execute this container recipe. [Docker Hub tag](https://hub.docker.com/layers/vllm/vllm-openai/v0.8.5/images/sha256-6cf9808ca8810fc6c3fd0451c2e7784fb224590d81f7db338e7eaf3c02a33d33) · [vLLM 0.8.5 CUDA dependency records](https://github.com/vllm-project/vllm/blob/v0.8.5/requirements/cuda.txt) · [Registry record](registry.json)
