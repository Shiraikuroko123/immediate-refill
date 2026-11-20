"""Check the selected experiment sources and, optionally, a live deployment."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--model", type=Path, help="Verify checkpoint hashes and the live GPU runtime")
    args = parser.parse_args()
    folder = args.artifact / "tracks/interference/prospective-device"
    freeze = folder / "freeze.json"
    if sha256(freeze) != "05512fc7bc4c0d25156bffa4e6dce2fb988d154550f4517f9be439990256ff02":
        raise ValueError("Unexpected source manifest")
    hashes = json.loads(freeze.read_bytes())
    for name, digest in hashes.items():
        if sha256(args.artifact / name) != digest:
            raise ValueError(f"Source mismatch: {name}")
    report = {"frozen_sources_verified": sorted(hashes)}
    if args.model:
        import torch
        import vllm

        with (folder / "session01/calibrate.jsonl").open(encoding="utf-8") as fp:
            original = json.loads(fp.readline())
        for name, digest in original["model_files"].items():
            if sha256(args.model / name) != digest:
                raise ValueError(f"Checkpoint mismatch: {name}")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Select one CUDA device")
        observed = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                        vllm=vllm.__version__, cuda=torch.version.cuda)
        for key, value in observed.items():
            if original[key] != value:
                raise RuntimeError(f"{key}: measured {value}, expected {original[key]}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This device does not support BF16")
        report.update(observed, python=sys.version, platform=platform.platform(),
                      model_files=original["model_files"],
                      libraries={p: importlib.metadata.version(p)
                                 for p in ("transformers", "tokenizers", "safetensors", "xformers", "triton")})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
