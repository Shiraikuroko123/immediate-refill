"""Quota-limited plurality voting on a vLLM 0.7.3 V0 engine."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from threading import Lock
import time
import traceback
import uuid

os.environ["VLLM_USE_V1"] = "0"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lookahead/live"))
from run_gpu import boxed, digest, minimum_prefix, store, tally
import frontier
import run_gpu


N, SLOTS, QUOTA = 16, 4, 14
COMPLETION_CAP = 4096
TEMPERATURE, TOP_P = .6, .95
JOBS = ("fixed-exact", "fixed-buffer", "fixed-b0", "fixed-eager", "natural-exact", "natural-buffer")
PROMPT = "\nSolve the problem and put your final answer in \\boxed{}."
WARMUP_PROMPT = "Ready."


def seed(uid, index):
    value = hashlib.sha256(f"quota-panel-v1/{uid}/{index}".encode()).digest()[:4]
    return int.from_bytes(value, "big") % (2**31-1)


class Ledger:
    """One commitment per successful reservation. Cancellations do not restore quota."""
    def __init__(self, quota):
        self.quota = quota
        self.used = 0
        self.lock = Lock()

    def reserve(self, index, step, events):
        with self.lock:
            if index != self.used or self.used >= self.quota:
                raise RuntimeError("Admission would violate the launch quota")
            self.used += 1
            events.append(dict(kind="reserve", index=index, step=step, before=index,
                               after=self.used, quota=self.quota))


def admission(policy, observed, quota):
    capacity = min(N, quota, len(observed) + SLOTS)
    if policy in ("full", "eager"):
        return capacity
    visible = observed
    if policy == "buffer":
        first_hole = next((i for i in range(N) if i not in observed), N)
        visible = {i:observed[i] for i in range(first_hole)}
    return min(capacity, minimum_prefix(N, visible) + int(policy != "b0"))


def drive(engine, tokenizer, sample, delta, cuda, q, qid, prompt_ids, condition,
          policy, reference, ref_sha, config_hash, position, out):
    run_id = uuid.uuid4().hex
    quota = N if policy in ("full", "eager") else QUOTA
    ledger = Ledger(quota)
    now = time.perf_counter

    def issue(request_id, index, length, forced, input_ids, sampling_seed):
        params = sample(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=length,
                        min_tokens=length if forced else 0, ignore_eos=forced,
                        seed=sampling_seed, detokenize=False, output_kind=delta)
        engine.add_request(request_id, {"prompt_token_ids":input_ids}, params)

    warm_prompt_ids = tokenizer.encode(WARMUP_PROMPT, add_special_tokens=False)
    warm_start = now()
    issue("engine-warmup-" + run_id + "-w", -1, 1, True, warm_prompt_ids, 0)
    warm = []
    while engine.has_unfinished_requests():
        for result in engine.step():
            for candidate in result.outputs:
                warm.extend(candidate.token_ids)
    if len(warm) != 1:
        raise RuntimeError("Warmup returned an unexpected token count")
    cuda.synchronize()
    warm_seconds = now() - warm_start
    cuda.reset_peak_memory_stats()
    started = now()
    paths, active, seen, events = [], {}, {}, []
    step = 0
    try:
        while True:
            winner, gap = tally(seen)
            if len(seen) == N:
                reason = "panel_complete"
                break
            if policy != "full" and gap > N-len(seen):
                reason = "strict_lock"
                break
            if ledger.used == quota and not active:
                reason = "quota_exhausted"
                break
            target = admission(policy, seen, quota)
            for index in range(ledger.used, target):
                request = f"{run_id}-{index}"
                forced = condition == "fixed"
                length = reference["paths"][index]["length"] if forced else COMPLETION_CAP
                ledger.reserve(index, step, events)
                issue(request, index, length, forced, prompt_ids, seed(q["unique_id"], index))
                active[request] = index
                paths.append(dict(index=index, seed=seed(q["unique_id"], index), finished=False,
                                  launch_step=step, launch_s=now()-started, token_ids=[]))
                events.append(dict(kind="launch", index=index, step=step))
            if not active:
                raise RuntimeError("Reachable nonterminal history has no active or admissible request")
            step += 1
            for result in engine.step():
                index = active[result.request_id]
                path = paths[index]
                if len(result.outputs) != 1:
                    raise RuntimeError("Expected one completion per path")
                candidate = result.outputs[0]
                path["token_ids"].extend(candidate.token_ids)
                if result.finished:
                    length = len(path["token_ids"])
                    if not 0 < length <= COMPLETION_CAP:
                        raise RuntimeError("Invalid completion length")
                    if condition == "fixed":
                        src = reference["paths"][index]
                        if length != src["length"]:
                            raise RuntimeError("Fixed path differs from reference length")
                        vote = src["vote"]
                    else:
                        vote = boxed(tokenizer.decode(path["token_ids"], skip_special_tokens=True)) if candidate.finish_reason == "stop" else None
                    path.update(finished=True, vote=vote, finish_reason=candidate.finish_reason,
                                finish_step=step, finish_s=now()-started)
                    del active[result.request_id]
                    seen[index] = vote
                    events.append(dict(kind="complete", index=index, step=step, tokens=length))
        if active:
            engine.abort_request(list(active))
        for index in active.values():
            paths[index].update(finish_reason="canceled", vote=None)
            events.append(dict(kind="cancel", index=index, step=step, tokens=len(paths[index]["token_ids"])))
        cuda.synchronize()
        elapsed = now() - started
        if engine.has_unfinished_requests():
            raise RuntimeError("Engine retained an active request after termination")
        events.append(dict(kind="stop", reason=reason, step=step, winner=winner, gap=gap))
        for path in paths:
            path["length"] = len(path["token_ids"])
            path["token_sha256"] = digest(json.dumps(path["token_ids"], separators=(",", ":")).encode())
            if condition == "fixed" and not path["finished"] and path["length"] >= reference["paths"][path["index"]]["length"]:
                raise RuntimeError("Canceled fixed path reached its terminal length")
        full = reference["paths"] if reference else paths
        ref_votes = {r["index"]:r["vote"] for r in full}
        serial = minimum_prefix(N, ref_votes)
        ref_winner = tally(ref_votes)[0]
        certified = reason in ("panel_complete", "strict_lock")
        if condition == "fixed":
            if certified and winner != ref_winner:
                raise RuntimeError("Certified winner differs from reference")
            if policy != "eager" and ledger.used > serial + int(policy != "b0"):
                raise RuntimeError("Relative start budget exceeded")
        counts = Counter(v for v in seen.values() if v is not None)
        return dict(qid=qid, unique_id=q["unique_id"], condition=condition, policy=policy,
                    position=position, seed_namespace="quota-panel-v1", config_hash=config_hash,
                    reference_sha256=ref_sha, quota=quota, quota_used=ledger.used, quota_reached=ledger.used==quota,
                    stop_reason=reason, certified=certified, winner=winner, final_gap=gap, vote_counts=dict(counts),
                    reference_winner=ref_winner, reference_serial_prefix=serial,
                    launch_delta_vs_reference=ledger.used-serial, launched=ledger.used,
                    completed=len(seen), canceled=len(active), generation_tokens=sum(r["length"] for r in paths),
                    canceled_tokens=sum(paths[i]["length"] for i in active.values()), steps=step,
                    wall_seconds=elapsed, warmup_seconds=warm_seconds, warmup_ids=warm, warmup_prompt_ids=warm_prompt_ids,
                    peak_allocated_bytes=cuda.max_memory_allocated(), paths=paths, events=events)
    except BaseException:
        store(out/f"failed-{qid:02d}-{condition}-{policy}-{run_id}.json",
              dict(config_hash=config_hash, reference_sha256=ref_sha, qid=qid, condition=condition, policy=policy,
                   step=step, quota=quota, quota_used=ledger.used, events=events, paths=paths,
                   error=traceback.format_exc()))
        raise


def jobs_for(qid, selected):
    # Balanced Latin square. Over six consecutive questions every comparison
    # occupies each of the six positions once and each ordered pair is adjacent once.
    order = [JOBS[(j+qid) % 6] for j in (0, 1, 5, 2, 4, 3)]
    return [x for x in order if x in selected]


def verify_reference(ref, qid, q, protocol):
    if (ref["qid"], ref["unique_id"], ref["policy"], ref["condition"], ref["completed"],
        ref["protocol_hash"]) != (qid, q["unique_id"], "full", "natural", N, protocol):
        raise ValueError("Incompatible reference panel")
    if [p["index"] for p in ref["paths"]] != list(range(N)):
        raise ValueError("Incomplete reference")


def run(args):
    selection = json.loads(args.selection.read_bytes())
    questions = selection["questions"]
    qids = list(range(len(questions))) if args.qids is None else list(map(int,args.qids.split(",")))
    if len(qids) != len(set(qids)) or any(i not in range(len(questions)) for i in qids):
        raise ValueError("Invalid or duplicate question indices")
    selected = set(args.jobs.split(","))
    if not selected <= set(JOBS) | {"natural-full"}:
        raise ValueError(f"Valid jobs: {JOBS}, natural-full")
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    import vllm
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    model = Path(args.model)
    model_files = {}
    if model.is_dir():
        for name in ("config.json","tokenizer_config.json","tokenizer.json","model.safetensors.index.json"):
            if (model/name).is_file():
                model_files[name] = digest((model/name).read_bytes())
    protocol = dict(model=args.model_identity, model_files=model_files, n=N, slots=SLOTS, quota=QUOTA,
                    completion_cap=COMPLETION_CAP, temperature=TEMPERATURE, top_p=TOP_P,
                    max_model_len=args.max_model_len, prompt=PROMPT, engine_warmup=WARMUP_PROMPT, seed_namespace="quota-panel-v1",
                    selection_sha256=digest(args.selection.read_bytes()),
                    code_sha256={p.name:digest(p.read_bytes()) for p in (Path(__file__), Path(run_gpu.__file__),Path(frontier.__file__))})
    protocol_hash = digest(json.dumps(protocol,sort_keys=True).encode())
    config = dict(protocol=protocol, protocol_hash=protocol_hash, model=str(model), gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, cuda=torch.version.cuda, vllm=vllm.__version__, dtype="bfloat16",
                  utilization=args.gpu_memory_utilization, v0=True)
    cfg = digest(json.dumps(config,sort_keys=True).encode())
    config_path = args.out/f"config-{cfg}.json"
    if not config_path.exists():
        store(config_path, config)
    print("Starting",config["gpu"],qids,sorted(selected),"protocol",protocol_hash, flush=True)
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=args.model, dtype="bfloat16", tensor_parallel_size=1, gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, max_num_seqs=SLOTS, enable_prefix_caching=False,
        num_scheduler_steps=1, disable_async_output_proc=True, disable_log_stats=True))
    tokenizer = engine.get_tokenizer()
    for qid in qids:
        question = questions[qid]
        ids = tokenizer.apply_chat_template([{"role":"user","content":question["problem"]+PROMPT}],
                                            add_generation_prompt=True, tokenize=True)
        if len(ids)+COMPLETION_CAP > args.max_model_len:
            raise ValueError(f"Prompt plus generation cap exceeds context on q{qid}")
        reference, ref_sha = None, None
        order = ["natural-full", *jobs_for(qid, selected)]
        for position, job in enumerate(order):
            condition, policy = job.split("-")
            name = f"q{qid:02d}-{job}.json"
            path = args.out/name
            if policy == "full" and args.references:
                source = args.references/name
                data = source.read_bytes()
                imported = json.loads(data)
                verify_reference(imported, qid, question, protocol_hash)
                if not path.exists():
                    store(path, imported)
                if path.read_bytes() != data:
                    raise ValueError("Imported and local reference bytes differ")
            if path.exists():
                row = json.loads(path.read_bytes())
                if policy != "full" or not args.references:
                    if row["config_hash"] != cfg or row["reference_sha256"] != ref_sha:
                        raise ValueError(f"Cannot resume incompatible run {path}")
            else:
                print("begin",qid,job,flush=True)
                row = drive(engine,tokenizer,SamplingParams,RequestOutputKind.DELTA,torch.cuda,question,qid,ids,
                            condition,policy,reference,ref_sha,cfg,position,args.out)
                row["protocol_hash"] = protocol_hash
                store(path,row)
                print("end",qid,job,row["stop_reason"],row["wall_seconds"],row["generation_tokens"],flush=True)
            if policy == "full":
                verify_reference(row,qid,question,protocol_hash)
                reference,ref_sha = row,digest(path.read_bytes())


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-identity", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--references", type=Path, help="Reuse full references generated by another shard or GPU")
    parser.add_argument("--qids", help="Comma-separated zero-based selection indices")
    parser.add_argument("--jobs", default=",".join(JOBS), help="Comma-separated jobs; natural-full alone generates references")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.75)
    return parser


if __name__ == "__main__":
    run(cli().parse_args())
