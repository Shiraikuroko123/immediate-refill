"""Measure uniform-batch cached HF decoding with local model weights only."""
import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"))
parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16])
parser.add_argument("--prefix", type=int, default=128)
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--vendor", type=Path, help="Optional Python dependencies directory")
args = parser.parse_args()
if min(*args.batches, args.prefix, args.steps, args.repeats, args.threads) < 1:
    parser.error("Lengths, counts and batch sizes must be positive")
if len(set(args.batches)) != len(args.batches):
    parser.error("Duplicate batch sizes")
if args.vendor:
    sys.path.insert(0, str(args.vendor.resolve()))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = str(args.threads)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import psutil
import torch
import transformers
from transformers import AutoModelForCausalLM


def write_event(event, **fields):
    row = {"event": event, "utc": datetime.now(timezone.utc).isoformat(), **fields}
    with (args.out / "events.jsonl").open("a", encoding="utf8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync():
    if args.device == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def run_batch(model, ids, repeat):
    batch = ids.shape[0]
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync()
    start = time.perf_counter()
    # Build the cache from the transformer alone to avoid a B x L x V allocation.
    # The next query token is fixed. All timed decode calls use the full CausalLM.
    pref = model.base_model(input_ids=ids, use_cache=True, return_dict=True)
    past = pref.past_key_values
    if past is None:
        raise RuntimeError("Model returned no KV cache")
    del pref
    token = ids[:, :1].contiguous()
    sync()
    prefill_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    for _ in range(args.steps):
        result = model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True)
        token = result.logits[:, -1].argmax(dim=-1, keepdim=True)
        past = result.past_key_values
        del result
    sync()
    elapsed_ms = (time.perf_counter() - start) * 1000
    mem = psutil.Process().memory_info()
    return dict(batch=batch, repeat=repeat, prefill_backbone_ms=prefill_ms,
                decode_steps=args.steps, decode_elapsed_ms=elapsed_ms,
                mean_ms_per_step=elapsed_ms / args.steps,
                tokens_per_second=batch * args.steps * 1000 / elapsed_ms,
                rss_bytes=mem.rss, cpu_percent_system=psutil.cpu_percent(interval=None),
                available_ram_bytes=psutil.virtual_memory().available,
                process_peak_working_set=getattr(mem, "peak_wset", None),
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == "cuda" else None)


def main():
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Selected CUDA runtime is unavailable")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    dtype = args.dtype or ("bfloat16" if args.device == "cuda" else "float32")
    args.out.mkdir(parents=True, exist_ok=False)
    if args.device == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Choose --dtype float16 for a GPU without BF16 support")
    gpu = torch.cuda.get_device_properties(0) if args.device == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model.resolve(), local_files_only=True, torch_dtype=getattr(torch, dtype),
        trust_remote_code=False,
    ).eval().to(args.device)
    if args.prefix + args.steps > getattr(model.config, "max_position_embeddings", float("inf")):
        raise ValueError("Requested prefix plus decode steps exceeds model context")
    weight_manifest = {f.name: dict(size=f.stat().st_size, sha256=file_hash(f))
                       for f in sorted(args.model.glob("*"))
                       if f.suffix in {".safetensors", ".bin"} or f.name == "config.json"}
    write_event("metadata", model=str(args.model.resolve()), model_files=weight_manifest,
                dtype=dtype, device=args.device, gpu=(dict(name=gpu.name,
                memory=gpu.total_memory) if gpu else None), cuda=torch.version.cuda,
                torch=torch.__version__, transformers=transformers.__version__,
                cpu=platform.processor(), logical_cores=os.cpu_count(), threads=args.threads,
                interop_threads=1, host_ram=psutil.virtual_memory().total,
                available_ram_bytes=psutil.virtual_memory().available,
                cpuload_before_pct=psutil.cpu_percent(interval=0.5),
                script_hash=file_hash(Path(__file__)), args=vars(args) | {
                    "model": str(args.model), "out": str(args.out), "vendor": str(args.vendor)},
                parameters=sum(p.numel() for p in model.parameters()), python=sys.version)
    generator = torch.Generator(device="cpu").manual_seed(1701)
    largest = torch.randint(model.config.vocab_size, (max(args.batches), args.prefix),
                            generator=generator).to(args.device)
    results = {b: [] for b in args.batches}
    rng = random.Random(1701)
    for rep in range(-1, args.repeats):
        order = args.batches.copy()
        rng.shuffle(order)
        for batch in order:
            result = run_batch(model, largest[:batch], rep)
            write_event("warmup" if rep < 0 else "trial", **result)
            if rep >= 0:
                results[batch].append(result["mean_ms_per_step"])
    summary = {batch: {"median_ms_per_step": statistics.median(values),
                       "min": min(values), "max": max(values)} for batch, values in results.items()}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
