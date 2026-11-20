"""Post-hoc, paired H20 comparison of two one-extra-launch admission gates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from unittest.mock import patch

import run_gpu
from frontier import launch_limit, minimum_prefix


HERE = Path(__file__).resolve().parent


def contiguous_votes(observed):
    result = {}
    for index in range(len(observed)):
        if index not in observed:
            break
        result[index] = observed[index]
    return result


def buffered_limit(n, slots, budget, observed):
    # Only the serial-prefix certificate loses information. Every completion
    # still frees its physical slot and still counts in the stopping tally.
    prefix = contiguous_votes(observed)
    return min(n, minimum_prefix(n, prefix) + budget, len(observed) + slots)


def audit_gate(row, gate):
    """Recover every distinct admission history, including no-launch decisions."""
    completed = defaultdict(list)
    launches = {}
    for event in row["events"]:
        if event["kind"] == "complete":
            completed[event["step"]].append(event["index"])
        elif event["kind"] == "launch" and event["step"] not in launches:
            launches[event["step"]] = event["indices"]
        else:
            raise ValueError("Unexpected or duplicate engine event")
    observed = {}
    paths = row["paths"]
    launched = 0
    records = []
    for step in sorted({0, *completed, *launches}):
        for index in completed[step]:
            path = paths[index]
            if index in observed or index >= launched or path["finish_step"] != step:
                raise ValueError("Completion does not match the active path history")
            observed[index] = path["vote"]
        terminal = len(observed) == 16 or run_gpu.tally(observed)[1] > 16 - len(observed)
        if terminal != (step == row["steps"]):
            raise ValueError("Observed history should have stopped at a different step")
        exact = launch_limit(16, 4, 1, observed)
        buffer = buffered_limit(16, 4, 1, observed)
        if buffer > exact:
            raise ValueError("Buffered admission cannot exceed indexed admission")
        target = launched if terminal else (exact if gate == "exact" else buffer)
        new = launches.get(step, [])
        if new != list(range(launched, target)) or any(paths[i]["launch_step"] != step for i in new):
            raise ValueError("Recorded launches disagree with the specified gate")
        records.append(dict(step=step, observed_indices=sorted(observed),
                            contiguous_length=len(contiguous_votes(observed)),
                            exact_target=exact, buffer_target=buffer, launched_before=launched,
                            launched_after=launched+len(new), terminal=terminal))
        launched += len(new)
    if not records[-1]["terminal"] or launched != row["launched"] or len(observed) != row["completed"]:
        raise ValueError("Incomplete terminal launch or completion history")
    return records


def load_prerequisites(args):
    if args.after_exit_file:
        deadline = time.monotonic() + args.wait_hours * 3600
        while not args.after_exit_file.is_file() or not args.after_exit_file.read_text().strip():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Preceding job did not finish within {args.wait_hours} hours")
            time.sleep(10)
        status = int(args.after_exit_file.read_text().strip())
        if status != 0:
            raise RuntimeError(f"Preceding job exited {status}")
    selection = json.loads(args.selection.read_bytes())
    if len(selection["questions"]) != 12:
        raise ValueError("Expected the twelve fixed study questions")
    refs = []
    hashes = {}
    configs = set()
    for qid, question in enumerate(selection["questions"]):
        source = args.references / f"q{qid:02d}-natural-full.json"
        data = source.read_bytes()
        ref = json.loads(data)
        if (ref["qid"], ref["unique_id"], ref["condition"], ref["policy"],
            ref["completed"], ref["canceled"]) != (qid, question["unique_id"], "natural", "full", 16, 0):
            raise ValueError(f"Incomplete or mismatched full reference {source}")
        for condition in ("natural", "fixed"):
            for method in ("B0", "B1", "eager"):
                previous = args.references / f"q{qid:02d}-{condition}-{method}.json"
                if not previous.is_file():
                    raise FileNotFoundError(f"Preceding run incomplete: {previous}")
        refs.append(ref)
        hashes[source.name] = run_gpu.digest(data)
        configs.add(ref["config_hash"])
    if len(configs) != 1:
        raise ValueError("References use different configurations")
    original_hash = configs.pop()
    original = json.loads((args.references / f"config-{original_hash}.json").read_bytes())
    if original_hash != run_gpu.digest(json.dumps(original, sort_keys=True).encode()):
        raise ValueError("Original configuration digest mismatch")
    for path in (Path(run_gpu.__file__), run_gpu.HERE.parent / "theory/frontier.py", args.selection):
        if run_gpu.digest(path.read_bytes()) != original["files_sha256"][path.name]:
            raise ValueError(f"Original study source differs: {path}")
    return selection, refs, hashes, original_hash, original


def main(args):
    selection, refs, hashes, original_hash, original = load_prerequisites(args)
    import torch
    import vllm
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    for key, current in (("gpu", torch.cuda.get_device_name()), ("cuda", torch.version.cuda),
                         ("torch", torch.__version__), ("vllm", vllm.__version__)):
        if current != original[key]:
            raise ValueError(f"Expected {key}='{original[key]}', found '{current}'")
    config = dict(study="post-hoc matched indexed B1 vs contiguous B1",
                  original_config=original_hash, full_reference_sha256=hashes,
                  script_sha256=run_gpu.digest(Path(__file__).read_bytes()),
                  order="even qid exact first; odd qid buffer first")
    cfg = run_gpu.digest(json.dumps(config, sort_keys=True).encode())
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / f"config-{cfg}.json"
    if not manifest.exists():
        run_gpu.store(manifest, config)
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=original["model"], dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=original["utilization"], max_model_len=original["max_model_len"],
        max_num_seqs=4, enable_prefix_caching=False, num_scheduler_steps=1,
        disable_async_output_proc=True, disable_log_stats=True))
    tokenizer = engine.get_tokenizer()
    for qid, (question, reference) in enumerate(zip(selection["questions"], refs)):
        prompt = [{"role":"user","content":question["problem"]+"\nSolve the problem and put your final answer in \\boxed{}."}]
        ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        pair = ("exact", "buffer") if qid % 2 == 0 else ("buffer", "exact")
        ref_hash = hashes[f"q{qid:02d}-natural-full.json"]
        for position, gate in enumerate(pair, start=1):
            path = args.out / f"q{qid:02d}-fixed-{gate}.json"
            if path.exists():
                row = json.loads(path.read_bytes())
                if (row["config_hash"], row["reference_sha256"], row["order_position"], row["gate"]) != (
                        cfg, ref_hash, position, gate):
                    raise ValueError(f"Cannot resume mismatched record {path}")
                continue
            gate_function = launch_limit if gate == "exact" else buffered_limit
            with patch.object(run_gpu, "launch_limit", gate_function):
                row = run_gpu.run(engine, tokenizer, question, qid, ids, "fixed", "B1",
                                  reference, cfg, SamplingParams, RequestOutputKind.DELTA)
            row.update(gate=gate, order_position=position, reference_sha256=ref_hash,
                       gate_events=audit_gate(row, gate))
            run_gpu.store(path, row)
            print(qid, gate, row["wall_seconds"], row["generation_tokens"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True, help="Original completed 84-panel directory")
    parser.add_argument("--out", type=Path, required=True, help="Separate ablation output directory")
    parser.add_argument("--selection", type=Path, default=HERE/"selection.json")
    parser.add_argument("--after-exit-file", type=Path, help="Wait for the first job's numeric exit status before allocating GPU")
    parser.add_argument("--wait-hours", type=float, default=12)
    main(parser.parse_args())
