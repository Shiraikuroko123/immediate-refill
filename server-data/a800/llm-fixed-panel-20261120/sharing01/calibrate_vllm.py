"""Matched cached-decode windows for the project's single-card vLLM V0 environment."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event(path, kind, **data):
    with path.open("a", encoding="utf8") as stream:
        stream.write(json.dumps(dict(event=kind, utc=datetime.now(timezone.utc).isoformat(), **data))+"\n")


def run_window(engine, torch, params, uid, width, token_ids, args):
    requests = [f"{uid}-{i}" for i in range(width)]
    seen = {rid: [] for rid in requests}
    try:
        for i, rid in enumerate(requests):
            engine.add_request(rid, {"prompt_token_ids": token_ids}, params[i])

        def step():
            counts = {}
            for output in engine.step():
                rid = output.request_id
                if rid not in seen or rid in counts or len(output.outputs) != 1 or output.finished:
                    raise RuntimeError("Unexpected request, output count, or early termination")
                tokens = output.outputs[0].token_ids
                seen[rid].extend(tokens)
                counts[rid] = len(tokens)
            return counts

        for _ in range(len(token_ids)*width+8):
            step()
            if all(seen.values()):
                break
        else:
            raise RuntimeError("Prefill did not complete")
        if any(len(tokens) != 1 for tokens in seen.values()):
            raise RuntimeError(f"Decode positions differ after prefill: {[len(tokens) for tokens in seen.values()]}")
        for _ in range(args.prelude):
            if step() != dict.fromkeys(requests, 1):
                raise RuntimeError("Prelude failed to advance each request by exactly one")
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(args.steps):
            if step() != dict.fromkeys(requests, 1):
                raise RuntimeError("Timed window failed to advance each request by exactly one")
        torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - start) / 1e9
        return dict(width=width, context=len(token_ids), steps=args.steps, elapsed_s=elapsed,
                    ms_per_step=1000*elapsed/args.steps, first_request_tokens_sha256=hashlib.sha256(
                        json.dumps(seen[requests[0]], separators=(",",":")).encode()).hexdigest())
    finally:
        if engine.has_unfinished_requests():
            engine.abort_request(requests)
        torch.cuda.synchronize()
        if engine.has_unfinished_requests():
            raise RuntimeError("Cleanup retained running requests")


def summary(records, args):
    rng = random.Random(1702)
    result = []
    for context in args.contexts:
        groups = {}
        for row in records:
            if row["context"] == context:
                groups.setdefault(row["block"], {})[row["width"]] = row
        for width in args.widths:
            ratios = [group[1]["elapsed_s"]/group[width]["elapsed_s"] for group in groups.values()]
            distribution = sorted(median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
            result.append(dict(context=context, width=width, n_blocks=len(ratios),
                               median_block_r=median(ratios), min_block_r=min(ratios), max_block_r=max(ratios),
                               block_bootstrap_percentile_interval=[distribution[249],distribution[9749]],
                               first_request_token_hash_disagreements=sum(group[1]["first_request_tokens_sha256"]
                                 != group[width]["first_request_tokens_sha256"] for group in groups.values())))
    return result


def main(args):
    if not args.model.is_dir():
        raise ValueError("The pinned model must already exist locally")
    if (min(args.contexts + args.widths + [args.steps, args.prelude, args.blocks]) < 1
        or 1 not in args.widths or max(args.widths)>4 or len(set(args.widths))!=len(args.widths)
        or len(set(args.contexts))!=len(args.contexts)
        or max(args.contexts)+args.steps+args.prelude+2>args.max_model_len):
        raise ValueError("Invalid sweep dimensions or context cap")
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.out / "events.jsonl"
    event(path, "start", arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
          script_sha256=sha(Path(__file__)))
    try:
        os.environ["VLLM_USE_V1"] = "0"
        import torch
        import vllm
        from vllm import EngineArgs, LLMEngine, SamplingParams
        from vllm.sampling_params import RequestOutputKind
        engine_args = dict(model=str(args.model.resolve()), dtype="bfloat16", tensor_parallel_size=1,
                           gpu_memory_utilization=.75, max_model_len=args.max_model_len, max_num_seqs=4,
                           enable_prefix_caching=False, num_scheduler_steps=1, disable_async_output_proc=True,
                           disable_log_stats=True, enable_chunked_prefill=False,
                           max_num_batched_tokens=max(args.max_model_len,max(args.contexts)*max(args.widths)))
        event(path, "metadata", gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, torch=torch.__version__,
              vllm=vllm.__version__, engine_args=engine_args, config_sha256=sha(args.model/"config.json"),
              weights=[dict(file=p.name, size=p.stat().st_size) for p in sorted(args.model.glob("*.safetensors"))],
              nvidia_smi=subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,clocks.sm,clocks.mem",
                                         "--format=csv,noheader"],capture_output=True,text=True).stdout.strip())
        engine = LLMEngine.from_engine_args(EngineArgs(**engine_args))
        base = engine.get_tokenizer().encode(
            "Compute the following sum carefully, and give the final answer as an integer. 7 + 5 =",
            add_special_tokens=False)
        if not base:
            raise ValueError("Empty tokenized prompt")
        records = []
        rng = random.Random(4409)
        for block in range(-1, args.blocks):
            contexts = args.contexts.copy()
            rng.shuffle(contexts)
            for context in contexts:
                widths = args.widths.copy()
                rng.shuffle(widths)
                tokens = (base * (1+context//len(base)))[:context]
                for width in widths:
                    uid = f"b{block}-c{context}-w{width}"
                    params = [SamplingParams(temperature=args.temperature, top_p=args.top_p, seed=10000*(
                        block+2)+i, min_tokens=args.steps+args.prelude+2, max_tokens=args.steps+args.prelude+2,
                        ignore_eos=True, detokenize=False, output_kind=RequestOutputKind.DELTA) for i in range(width)]
                    event(path, "begin", block=block, uid=uid, context=context, width=width)
                    row = run_window(engine, torch, params, uid, width, tokens, args)
                    row.update(block=block)
                    event(path, "warmup" if block<0 else "trial", **row)
                    if block>=0:
                        records.append(row)
        result = summary(records, args)
        (args.out/"summary.json").write_text(json.dumps(result, indent=2)+"\n",encoding="utf8")
        event(path, "complete", trials=len(records))
    except BaseException:
        event(path, "failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--contexts",type=int,nargs="+",default=[512,2048,3968])
    parser.add_argument("--widths",type=int,nargs="+",default=[1,2,4])
    parser.add_argument("--steps",type=int,default=96)
    parser.add_argument("--prelude",type=int,default=8)
    parser.add_argument("--blocks",type=int,default=8)
    parser.add_argument("--max-model-len",type=int,default=8192)
    parser.add_argument("--temperature",type=float,default=.6)
    parser.add_argument("--top-p",type=float,default=.95)
    main(parser.parse_args())
