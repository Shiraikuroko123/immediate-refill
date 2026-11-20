"""Run the paired 384/1024-token experiment on the original vLLM V0 driver."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import traceback
import uuid


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("quota_driver", ROOT / "tracks/quota/live.py")
quota = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quota)

SHORT, LONG = 384, 1024
NAMESPACE = "gsm-cap-reference-v1"
PROMPT = "\nSolve the problem. Show your reasoning briefly and put your final integer answer in \\boxed{}."
ARMS = ((SHORT, "exact"), (SHORT, "buffer"), (LONG, "exact"), (LONG, "buffer"))
ORDERS = ((0,1,3,2), (1,2,0,3), (2,3,1,0), (3,0,2,1))
MODEL = {
    "model.safetensors": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
    "config.json": "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen_sources():
    paths = (Path(__file__), Path(quota.__file__), Path(quota.run_gpu.__file__), Path(quota.frontier.__file__),
             HERE / "PROTOCOL.md", HERE / "selection.json", HERE / "selection.py")
    return {p.resolve().relative_to(ROOT).as_posix(): sha(p) for p in paths}


def path_seed(uid, index):
    token = hashlib.sha256(f"{NAMESPACE}/{uid}/{index}".encode()).digest()
    return int.from_bytes(token[:4], "big") % (2**31 - 1)


def integer_box(text):
    v = quota.run_gpu.boxed(text)
    if v is not None and re.fullmatch(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)", v):
        return str(int(v.replace(",", "")))
    return None


def project(reference, cap):
    return {"paths": [dict(index=p["index"], length=min(cap, p["length"]),
                           vote=p["vote"] if p["length"] < cap else None) for p in reference["paths"]]}


def configure():
    quota.seed = path_seed
    quota.boxed = integer_box
    quota.PROMPT = PROMPT
    quota.TEMPERATURE, quota.TOP_P = .7, .95
    quota.COMPLETION_CAP = LONG


class ReferenceTokenizer:
    """A full-reference completion that reaches the cap abstains even on an EOS tie."""
    def __init__(self, tokenizer):
        self.encode = tokenizer.encode
        self._decode = tokenizer.decode

    def decode(self, tokens, **kwargs):
        return "" if len(tokens) >= LONG else self._decode(tokens, **kwargs)


def prepare_questions():
    questions = json.loads((HERE / "selection.json").read_bytes())["questions"]
    ids = {q["unique_id"] for q in questions}
    assert len(questions) == len(ids) == 32
    assert all(hashlib.sha256(q["problem"].encode()).hexdigest() == q["unique_id"] for q in questions)
    return questions


def run(args, frozen):
    import torch
    import vllm
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    questions = prepare_questions()
    qids = list(range(len(questions))) if args.qids is None else list(map(int, args.qids.split(",")))
    if not qids or set(qids) - set(range(32)) or len(set(qids)) != len(qids):
        raise ValueError("Question indices must be unique and between 0 and 31.")
    if vllm.__version__ != "0.8.5" or "A800" not in torch.cuda.get_device_name():
        raise ValueError("Expected the A800 with vLLM 0.8.5.")
    if {name: sha(args.model / name) for name in MODEL} != MODEL:
        raise ValueError("The model or tokenizer differs from the previous pilot.")
    configure()
    engine_args = dict(dtype="bfloat16", tensor_parallel_size=1, gpu_memory_utilization=.75,
                       max_model_len=2048, max_num_seqs=4, enable_prefix_caching=False,
                       enable_chunked_prefill=False, num_scheduler_steps=1, disable_async_output_proc=True,
                       disable_log_stats=True)
    protocol = dict(model="Qwen/Qwen2.5-1.5B-Instruct", model_files=MODEL, n=16, slots=4, quota=14,
                    caps=[SHORT,LONG], allowance=1, temperature=.7, top_p=.95, prompt=PROMPT,
                    seed_namespace=NAMESPACE, engine_warmup=quota.WARMUP_PROMPT, code_sha256=frozen,
                    selection_sha256=sha(HERE/"selection.json"), order=ORDERS, v0=True, **engine_args)
    protocol_hash = quota.digest(json.dumps(protocol,sort_keys=True).encode())
    config = dict(protocol=protocol,protocol_hash=protocol_hash,gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__,cuda=torch.version.cuda,vllm=vllm.__version__,python=sys.version)
    cfg = quota.digest(json.dumps(config,sort_keys=True).encode())
    args.out.mkdir(exist_ok=False, parents=True)
    session = uuid.uuid4().hex
    quota.store(args.out/f"config-{cfg}.json",config)
    quota.store(args.out/"driver.json",dict(freeze=frozen,session_id=session,qids=qids,
                                           launch_utc=datetime.now(timezone.utc).isoformat()))
    engine = LLMEngine.from_engine_args(EngineArgs(model=str(args.model), **engine_args))
    tokenizer = engine.get_tokenizer()
    reference_tokenizer = ReferenceTokenizer(tokenizer)
    for qid in qids:
        question = questions[qid]
        ids = tokenizer.apply_chat_template([{"role":"user","content":question["problem"]+PROMPT}],
                                             add_generation_prompt=True, tokenize=True)
        if len(ids) + LONG > engine_args["max_model_len"]:
            raise ValueError(f"Prompt plus output limit exceeds context on question {qid}")
        full = None
        ref_sha = None
        for pos, (cap, policy) in enumerate([(LONG,"full"), *(ARMS[j] for j in ORDERS[qid % 4])]):
            condition = "natural" if policy == "full" else "fixed"
            label = "natural-full" if policy == "full" else f"{cap}-{policy}"
            path = args.out/f"q{qid:02d}-{label}.json"
            print("begin", qid, label, flush=True)
            try:
                reference = project(full, cap) if full else None
                row = quota.drive(engine,reference_tokenizer,SamplingParams,RequestOutputKind.DELTA,torch.cuda,question,
                                  qid,ids,condition,policy,reference,ref_sha,cfg,pos,args.out)
                row.update(protocol_hash=protocol_hash,cap=cap,session_id=session,seed_namespace=NAMESPACE)
                quota.store(path,row)
                if policy == "full":
                    quota.verify_reference(row,qid,question,protocol_hash)
                    full, ref_sha = row, sha(path)
                print("end", qid, label, row["steps"], row["wall_seconds"], flush=True)
            except BaseException:
                quota.store(args.out/f"failed-{qid:02d}-{label}-driver.json",
                            dict(qid=qid,cap=cap,policy=policy,session_id=session,error=traceback.format_exc()))
                raise


def smoke():
    sample = {"paths": [dict(index=k, length=w, vote=str(k)) for k,w in enumerate((1,383,384,385,1024))]}
    result = project(sample,SHORT)["paths"]
    assert [p["length"] for p in result] == [1,383,384,384,384]
    assert [p["vote"] for p in result] == ["0","1",None,None,None]
    assert [p["vote"] for p in project(sample,LONG)["paths"]] == ["0","1","2","3",None]
    assert all(len(set(order)) == 4 for order in ORDERS)
    assert all(len({order[i] for order in ORDERS}) == 4 for i in range(4))
    assert set(Counter((a,b) for order in ORDERS for a,b in zip(order,order[1:])).values()) == {1}
    assert integer_box(r"\boxed{2,011}") == "2011" and integer_box(r"\boxed{x}") is None
    assert len(prepare_questions()) == 32
    print("Checked question identities, cutoff boundary, shared votes and balanced condition order.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--model",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--qids")
    args = p.parse_args()
    if args.smoke:
        smoke()
    elif args.freeze:
        with (HERE/"freeze.json").open("x",encoding="utf8") as f:
            json.dump(frozen_sources(), f, indent=2)
            f.write("\n")
    else:
        if not (args.model and args.out):
            p.error("Supply --model and a new --out directory.")
        frozen = frozen_sources()
        if frozen != json.loads((HERE/"freeze.json").read_bytes()):
            raise ValueError("The operative code, protocol or selection differs from the frozen file.")
        run(args,frozen)
