"""Frozen prospective device experiment. See PROTOCOL.md for decision rules."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WINDOW = ROOT / "tracks/sharing/hardware/calibrate_vllm.py"
spec = importlib.util.spec_from_file_location("cached_window", WINDOW)
cached_window = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cached_window)
FILES = [HERE / "PROTOCOL.md", HERE / "experiment.py", HERE / "run.sh", WINDOW]
MODEL = {
    "model.safetensors": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
    "config.json": "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
}
LENGTH = 128
ARMS = [("witness", "immediate"), ("witness", "staged"),
        ("control", "immediate"), ("control", "staged")]
ORDERS = [(0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1)]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, obj):
    with path.open("x", encoding="utf8") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in FILES}


def save_event(path, kind, **data):
    cached_window.event(path, kind, **data)


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def saving(immediate, staged):
    return 100 * (immediate - staged) / immediate


def predict(rows):
    by_width = {w: [r["ms_per_step"] for r in rows if r["width"] == w] for w in (1, 2, 3)}
    if any(len(v) != 8 for v in by_width.values()) or sorted((r["block"],r["width"]) for r in rows) != [
            (i,k) for i in range(8) for k in (1,2,3)]:
        raise ValueError("Expected eight complete calibration blocks")
    c1, c2, c3 = [median(by_width[w]) for w in (1, 2, 3)]
    result = {}
    for panel, immediate, staged in (("witness", LENGTH*(c1+c3), 2*LENGTH*c2),
                                     ("control", LENGTH*c2, 2*LENGTH*c1)):
        effect = saving(immediate, staged)
        decision = "tie" if abs(effect) < 1 else ("staged" if effect > 0 else "immediate")
        result[panel] = dict(immediate_ms=immediate, staged_ms=staged, effect_pct=effect, decision=decision)
    return dict(clock_ms=[c1, c2, c3], panels=result)


def setup(model, path):
    import torch
    import vllm
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    if {name: sha(model/name) for name in MODEL} != MODEL:
        raise ValueError("Model or tokenizer fingerprint mismatch")
    if vllm.__version__ != "0.8.5" or "A800" not in torch.cuda.get_device_name():
        raise ValueError("Expected A800 and vLLM 0.8.5")
    settings = dict(model=str(model.resolve()), dtype="bfloat16", tensor_parallel_size=1,
                    gpu_memory_utilization=.75, max_model_len=2048, max_num_seqs=4,
                    enable_prefix_caching=False, num_scheduler_steps=1, disable_async_output_proc=True,
                    disable_log_stats=True, enable_chunked_prefill=False, max_num_batched_tokens=2048)
    save_event(path, "metadata", gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
               vllm=vllm.__version__, python=sys.version, engine=settings, model_files=MODEL, pid=os.getpid(),
               gpu_query=subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,clocks.sm,clocks.mem",
                                         "--format=csv,noheader"],capture_output=True,text=True).stdout.strip())
    engine = LLMEngine.from_engine_args(EngineArgs(**settings))
    base = engine.get_tokenizer().encode(
        "Compute the following sum carefully, and give the final answer as an integer. 7 + 5 =",
        add_special_tokens=False)
    tokens = (base * (1 + 512 // len(base)))[:512]

    def params(n):
        return SamplingParams(temperature=0., min_tokens=n, max_tokens=n, ignore_eos=True, detokenize=False,
                              output_kind=RequestOutputKind.DELTA)
    return engine, torch, tokens, params


def calibrate(model, out):
    path = out / "calibrate.jsonl"
    engine, torch, tokens, params = setup(model, path)
    rng = random.Random(3721)
    for block in range(-1, 8):
        widths = [1,2,3]
        rng.shuffle(widths)
        for width in widths:
            uid = f"cal-{block}-{width}"
            row = cached_window.run_window(engine, torch, [params(162) for _ in range(width)], uid, width,
                                           tokens, SimpleNamespace(prelude=64, steps=96))
            save_event(path, "warmup" if block < 0 else "trial", block=block, **row)
    rows = [r for r in read_events(path) if r["event"] == "trial"]
    write_new(out/"prediction.json", dict(**predict(rows), calibration_sha256=sha(path),
              frozen_sources=source_hashes(), sealed_utc=datetime.now(timezone.utc).isoformat()))


def drive(engine, torch, tokens, params, uid, panel, arm, length=LENGTH):
    lengths = [2*length, length, length] if panel == "witness" else [length,length]
    limit = len(lengths) if arm == "immediate" else len(lengths)-1
    request_ids = [f"{uid}-{i}" for i in range(len(lengths))]
    indices = {rid:i for i,rid in enumerate(request_ids)}
    generated = [[] for _ in lengths]
    settings = [params(n) for n in lengths]
    live = set()
    events, steps = [], []
    next_lane = 0
    if engine.has_unfinished_requests():
        raise ValueError("Engine not drained at entry")
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    while next_lane < len(lengths) or live:
        while next_lane < len(lengths) and len(live) < limit:
            i = next_lane
            engine.add_request(request_ids[i], {"prompt_token_ids": tokens}, settings[i])
            events.append([len(steps), "admit", i])
            live.add(i)
            next_lane += 1
        if len(steps)>sum(lengths)*2:
            raise ValueError("Unexpected lack of progress")
        t0 = time.perf_counter_ns()
        outputs = engine.step()
        elapsed = time.perf_counter_ns()-t0
        advanced, first = [], []
        for o in outputs:
            i = indices[o.request_id]
            if i not in live or i in advanced or len(o.outputs) != 1 or len(o.outputs[0].token_ids) != 1:
                raise ValueError("Unexpected token event")
            if not generated[i]:
                first.append(i)
            advanced.append(i)
            generated[i].extend(o.outputs[0].token_ids)
            if o.finished != (len(generated[i]) == lengths[i]):
                raise ValueError("Finish event disagrees with fixed work")
            if o.finished:
                live.remove(i)
                events.append([len(steps), "finish", i])
        steps.append([elapsed, advanced, first])
    torch.cuda.synchronize()
    wall = time.perf_counter_ns()-start
    if engine.has_unfinished_requests() or [len(t) for t in generated] != lengths:
        raise ValueError("Unfinished panel or unequal work")
    return dict(panel=panel, arm=arm, wall_ns=wall, token_counts=lengths,
                token_hashes=[hashlib.sha256(json.dumps(t,separators=(",",":")).encode()).hexdigest()
                              for t in generated], transitions=events, steps=steps)


def validate(model, out):
    predfile = out / "prediction.json"
    seal = json.loads(predfile.read_bytes())
    if seal["frozen_sources"] != source_hashes() or seal["calibration_sha256"] != sha(out/"calibrate.jsonl"):
        raise ValueError("Calibration or sources changed")
    path = out / "validate.jsonl"
    save_event(path, "sealed_prediction", sha256=sha(predfile))
    engine, torch, tokens, params = setup(model, path)
    for block in range(-1,12):
        for pos, idx in enumerate(ORDERS[block % 4]):
            panel, arm = ARMS[idx]
            uid = f"val-{block}-{panel}-{arm}"
            save_event(path, "begin", block=block, panel=panel, arm=arm)
            row = drive(engine, torch, tokens, params, uid, panel, arm)
            save_event(path, "warmup" if block < 0 else "trial", block=block, pos=pos, **row)
    print("Completed device validation", flush=True)


def describe(rows, seal):
    result = {}
    for panel in ("witness","control"):
        selected = [r for r in rows if r["panel"] == panel]
        cells = {(r["block"],r["arm"]):r for r in selected}
        if len(selected) != 24 or set(cells) != {(i,a) for i in range(12) for a in ("immediate","staged")}:
            raise ValueError("Expected 12 complete paired validation blocks")
        pairs = []
        for i in range(12):
            im, st = [cells[i,arm] for arm in ("immediate","staged")]
            if im["token_counts"] != st["token_counts"]:
                raise ValueError("Different paired token budgets")
            cached = [sum(t for t,ids,first in r["steps"] if ids and not first) for r in (im,st)]
            firsts = [sum(t for t,ids,first in r["steps"] if first) for r in (im,st)]
            pairs.append(dict(block=i, effect_pct=saving(im["wall_ns"], st["wall_ns"]),
                              host_cached_step_effect_pct=saving(*cached), host_first_token_step_ns=firsts,
                              paired_token_hash_disagreements=sum(a!=b for a,b in zip(im["token_hashes"],
                                                                                    st["token_hashes"]))))
        effects = sorted(p["effect_pct"] for p in pairs)
        lo, hi = effects[2], effects[9]
        estimate = median(effects)
        prediction = seal["panels"][panel]
        agrees = {"tie":lo>=-2 and hi<=2,"staged":lo>0,"immediate":hi<0}[prediction["decision"]]
        result[panel] = dict(median_pct=estimate, interval_pct=[lo,hi], pairs=pairs,
                             passed=agrees and abs(estimate-prediction["effect_pct"])<=2,
                             prediction=prediction)
    return result


def analyze(out):
    seal = json.loads((out/"prediction.json").read_bytes())
    records = read_events(out/"validate.jsonl")
    if (records[0].get("sha256") != sha(out/"prediction.json")
            or seal["calibration_sha256"] != sha(out/"calibrate.jsonl")):
        raise ValueError("The prediction or its source changed")
    regenerated = predict([r for r in read_events(out/"calibrate.jsonl") if r["event"]=="trial"])
    if any(regenerated[k]!=seal[k] for k in regenerated):
        raise ValueError("Cannot reproduce sealed decision")
    results = describe([r for r in records if r["event"]=="trial"],seal)
    return dict(panels=results, joint_transfer_passed=all(v["passed"] for v in results.values()),
                inputs={n:sha(out/n) for n in ("prediction.json","calibrate.jsonl","validate.jsonl")})


def smoke():
    class Engine:
        def __init__(self):
            self.pending = {}
        def add_request(self,rid,prompt,maximum):
            self.pending[rid]=[maximum,0]
        def has_unfinished_requests(self):
            return bool(self.pending)
        def step(self):
            result=[]
            for rid,(n,i) in list(self.pending.items()):
                self.pending[rid][1]+=1
                done=i+1==n
                result.append(SimpleNamespace(request_id=rid,finished=done,outputs=[SimpleNamespace(token_ids=[i])]))
                if done:
                    del self.pending[rid]
            return result
    rows=[]
    torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None))
    for panel,arm in ARMS:
        t=drive(Engine(),torch,[2],lambda n:n,"fake",panel,arm,4)
        profile=Counter(len(ids) for _,ids,_ in t["steps"])
        expected={("witness","immediate"):{3:4,1:4},("witness","staged"):{2:8},
                  ("control","immediate"):{2:4},("control","staged"):{1:8}}
        assert profile==expected[panel,arm]
        for i in range(12):
            rows.append(dict(t,block=i,wall_ns=sum({1:10,2:11,3:16}[k]*v for k,v in profile.items())))
    pred=predict([dict(block=i,width=k,ms_per_step=v) for i in range(8) for k,v in enumerate([0,10,11,16]) if k])
    assert all(v["passed"] for v in describe(rows,pred).values())
    assert pred["panels"]["witness"]["decision"]=="staged" and pred["panels"]["control"]["decision"]=="immediate"
    assert len({(a,b) for order in ORDERS for a,b in zip(order,order[1:])})==12
    print("CPU mock verified both finite workload profiles, paired inference and opposite directions.")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["smoke","freeze","calibrate","validate","analyze"])
    p.add_argument("--out",type=Path)
    p.add_argument("--model",type=Path)
    args=p.parse_args()
    if args.stage=="freeze":
        write_new(HERE/"freeze.json",source_hashes())
    elif args.stage=="smoke":
        smoke()
    else:
        if not args.out or (args.stage!="analyze" and not args.model):
            p.error("Supply --out and, for device runs, --model")
        if source_hashes()!=json.loads((HERE/"freeze.json").read_bytes()):
            raise ValueError("Source fingerprint mismatch")
        os.environ["VLLM_USE_V1"]="0"
        if args.stage=="analyze":
            print(json.dumps(analyze(args.out),indent=2))
        else:
            path=args.out/f"{args.stage}.jsonl"
            with path.open("x",encoding="utf8") as stream:
                pass
            try:
                (calibrate if args.stage=="calibrate" else validate)(args.model,args.out)
            except BaseException:
                save_event(path,"failed",traceback=traceback.format_exc())
                raise
