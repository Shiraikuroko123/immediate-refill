"""Live V0 engine study of bounded excess path launches."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

os.environ["VLLM_USE_V1"] = "0"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "theory"))
from frontier import minimum_prefix, launch_limit


def digest(data):
    return hashlib.sha256(data).hexdigest()


def boxed(text):
    """Last complete, nonempty boxed expression, with balanced nested braces."""
    pos = len(text)
    while True:
        start = text.rfind("\\boxed{", 0, pos)
        if start < 0:
            return None
        begin = start + len("\\boxed{")
        depth = 1
        for idx in range(begin, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    return text[begin:idx].strip() or None
        pos = start


def tally(votes):
    c = Counter(x for x in votes.values() if x is not None)
    ranked = sorted(c,key=lambda x:(-c[x],x))
    return (ranked[0], c[ranked[0]]-(c[ranked[1]] if len(ranked)>1 else 0)) if ranked else (None,0)


def seed(uid,index):
    return int.from_bytes(hashlib.sha256(f"bounded-panel-v1/{uid}/{index}".encode()).digest()[:4],"big") % (2**31-1)


def store(path, record):
    """Do not replace any completed panel, even if two shards overlap."""
    partial=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with partial.open("x",encoding="utf-8") as f:
        json.dump(record,f,ensure_ascii=False,separators=(",",":"))
        f.flush()
        os.fsync(f.fileno())
    os.link(partial,path)
    partial.unlink()


def run(engine, tokenizer, question, number, ids, condition, method, reference, cfg, params_type, delta):
    import torch
    n, slots, cap = 16,4,1536
    run_id=uuid.uuid4().hex
    def issue(request_id,index,length,forced=False):
        params=params_type(temperature=.7,top_p=1.,max_tokens=length,min_tokens=length if forced else 0,
                           ignore_eos=forced,seed=seed(question["unique_id"],index),detokenize=False,
                           output_kind=delta)
        engine.add_request(request_id,{"prompt_token_ids":ids},params)
    issue(run_id+"-w",-1,1,True)
    warm_start=time.monotonic()
    warm_tokens=[]
    while engine.has_unfinished_requests():
        for out in engine.step():
            for candidate in out.outputs:
                warm_tokens.extend(candidate.token_ids)
    if len(warm_tokens)!=1:
        raise RuntimeError("Warmup did not emit exactly one token")
    torch.cuda.synchronize()
    warm_seconds=time.monotonic()-warm_start
    torch.cuda.reset_peak_memory_stats()
    paths, observed, active = {},{},{}
    events=[]
    started=time.monotonic()
    steps=0
    while True:
        winner,gap=tally(observed)
        if len(observed)==n or (method!="full" and gap>n-len(observed)):
            break
        if method in ("full","eager"):
            target=min(n,len(observed)+slots)
        else:
            target=launch_limit(n,slots,int(method[1:]),observed)
        new_indices=list(range(len(paths),target))
        if new_indices:
            events.append(dict(kind="launch",step=steps,elapsed=time.monotonic()-started,indices=new_indices))
        for index in new_indices:
            request_id=f"{run_id}-{index}"
            active[request_id]=index
            paths[index]=dict(index=index,seed=seed(question["unique_id"],index),token_ids=[],launch_step=steps,
                              launch_s=time.monotonic()-started,finished=False)
            length=reference["paths"][index]["length"] if condition=="fixed" else cap
            issue(request_id,index,length,condition=="fixed")
        if not active:
            raise RuntimeError("No runnable path before lock")
        steps+=1
        for out in engine.step():
            index=active[out.request_id]
            row=paths[index]
            if len(out.outputs)!=1:
                raise RuntimeError("Expected a single completion per request")
            candidate=out.outputs[0]
            row["token_ids"].extend(candidate.token_ids)
            if out.finished:
                length=len(row["token_ids"])
                if not 0 < length <= cap:
                    raise RuntimeError("Invalid completed length")
                if condition=="fixed":
                    archived=reference["paths"][index]
                    if length!=archived["length"]:
                        raise RuntimeError("Controlled path emitted a different length")
                    label=archived["vote"]
                else:
                    label=boxed(tokenizer.decode(row["token_ids"],skip_special_tokens=True)) if candidate.finish_reason=="stop" else None
                row.update(finished=True,finish_reason=candidate.finish_reason,vote=label,finish_step=steps,
                           finish_s=time.monotonic()-started)
                observed[index]=label
                del active[out.request_id]
                events.append(dict(kind="complete",index=index,tokens=length,step=steps))
    canceled=list(active.values())
    if active:
        engine.abort_request(list(active))
    torch.cuda.synchronize()
    elapsed=time.monotonic()-started
    if engine.has_unfinished_requests():
        raise RuntimeError("Engine retained a live path after cancellation")
    for index,row in paths.items():
        row["length"]=len(row["token_ids"])
        row["token_sha256"]=digest(json.dumps(row["token_ids"],separators=(",",":")).encode())
        if not row["finished"]:
            row["finish_reason"]="canceled"
            row["vote"]=None
            if condition=="fixed" and row["length"]>=reference["paths"][index]["length"]:
                raise RuntimeError("Controlled canceled path passed its terminal length")
    ref=reference["paths"] if reference else list(paths.values())
    ref_votes={p["index"]:p["vote"] for p in ref}
    ref_winner,_=tally(ref_votes)
    prefix=minimum_prefix(n,ref_votes)
    excess=len(paths)-prefix
    if condition=="fixed" and (ref_winner!=winner or method in ("B0","B1") and excess>int(method[1:])):
        raise RuntimeError("Fixed workload violated vote or path-launch bound")
    return dict(qid=number,unique_id=question["unique_id"],condition=condition,policy=method,
                wall_seconds=elapsed,warmup_seconds=warm_seconds,warmup_ids=warm_tokens,
                generation_tokens=sum(row["length"] for row in paths.values()),
                canceled_tokens=sum(paths[i]["length"] for i in canceled),launched=len(paths),
                completed=len(observed),canceled=len(canceled),winner=winner,reference_winner=ref_winner,
                reference_serial_prefix=prefix,launch_delta_vs_reference=excess,steps=steps,
                paths=list(paths.values()),events=events,config_hash=cfg,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())


def main(args):
    import torch
    import vllm
    from vllm import LLMEngine, EngineArgs, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    selection=json.loads(args.selection.read_text(encoding="utf-8"))
    qids=[int(x) for x in args.qids.split(",")] if args.qids else list(range(len(selection["questions"])))
    if len(qids)!=len(set(qids)) or any(i<0 or i>=len(selection["questions"]) for i in qids):
        raise ValueError("Invalid or duplicate qids")
    args.out.mkdir(parents=True,exist_ok=True)
    config=dict(model=str(args.model),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                vllm=vllm.__version__,n=16,slots=4,max_tokens=1536,temperature=.7,top_p=1.,
                max_model_len=args.max_model_len,utilization=args.gpu_memory_utilization,
                parser="last complete boxed expression at natural stop",v0=True,
                files_sha256={p.name:digest(p.read_bytes()) for p in
                              (Path(__file__),Path(sys.modules["frontier"].__file__),args.selection)})
    cfg=digest(json.dumps(config,sort_keys=True).encode())
    manifest=args.out/("config-"+cfg+".json")
    if not manifest.exists():
        store(manifest,config)
    engine=LLMEngine.from_engine_args(EngineArgs(model=str(args.model),dtype="bfloat16",tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,max_model_len=args.max_model_len,max_num_seqs=4,
            enable_prefix_caching=False,num_scheduler_steps=1,disable_async_output_proc=True,disable_log_stats=True))
    tokenizer=engine.get_tokenizer()
    for qid in qids:
        question=selection["questions"][qid]
        prompt=[{"role":"user","content":question["problem"]+"\nSolve the problem and put your final answer in \\boxed{}."}]
        ids=tokenizer.apply_chat_template(prompt,add_generation_prompt=True,tokenize=True)
        if len(ids)+1536>args.max_model_len:
            raise ValueError(f"Prompt too long {qid}")
        order=["B0","B1","eager"]
        offset=qid%len(order)
        order=order[offset:]+order[:offset]
        jobs=[("natural","full")]+[("natural",m) for m in order]+[("fixed",m) for m in reversed(order)]
        reference=None
        for condition,method in jobs:
            path=args.out/f"q{qid:02d}-{condition}-{method}.json"
            if path.exists():
                result=json.loads(path.read_text(encoding="utf-8"))
                if result["config_hash"]!=cfg:
                    raise ValueError("Saved panel uses a different configuration")
            else:
                result=run(engine,tokenizer,question,qid,ids,condition,method,reference,cfg,SamplingParams,RequestOutputKind.DELTA)
                store(path,result)
                print(qid,condition,method,result["wall_seconds"],result["generation_tokens"],flush=True)
            if method=="full":
                reference=result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--selection",type=Path,default=HERE/"selection.json")
    parser.add_argument("--qids",help="Comma-separated zero-based selection indices; defaults to all 12")
    parser.add_argument("--max-model-len",type=int,default=4096)
    parser.add_argument("--gpu-memory-utilization",type=float,default=.75)
    main(parser.parse_args())
