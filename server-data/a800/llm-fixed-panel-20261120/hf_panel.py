"""Continuous HF decode of fixed-seed panels. Tested interface: transformers 4.40.x."""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import random
import time
import traceback

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from workload import parse_answer, path_seed, questions


def gap_and_winner(votes):
    counts = Counter(v for v in votes if v is not None)
    ordered = sorted(counts, key=lambda v: (-counts[v], v))
    if not ordered:
        return 0, None
    second = counts[ordered[1]] if len(ordered) > 1 else 0
    return counts[ordered[0]] - second, ordered[0]


def safe(remaining, gap):
    return min(remaining, max(0, (remaining-gap)//2+1))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def as_legacy(cache):
    return cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache


def align_cache(previous_ids, active_ids, previous_cache, prompt_cache, past_lengths, device):
    """Left-align the masks at the right edge. Copy stored rows only on a membership change."""
    if previous_ids == active_ids:
        return previous_cache
    position = {idx: row for row, idx in enumerate(previous_ids)}
    keep_positions = [position[idx] for idx in active_ids if idx in position]
    new_count = len(active_ids)-len(keep_positions)
    # Surviving old paths always precede new admissions in the seed order.
    assert active_ids[:len(keep_positions)] == [previous_ids[k] for k in keep_positions]
    target_length = max(past_lengths)
    row_ids = torch.tensor(keep_positions, device=device, dtype=torch.long)
    layers = []
    for layer_index, (pk, pv) in enumerate(prompt_cache):
        merged = []
        for component, prefix in enumerate((pk,pv)):
            pieces = []
            if keep_positions:
                previous = previous_cache[layer_index][component]
                selected = previous[:, :, -target_length:, :].index_select(0,row_ids)
                pieces.append(selected)
            if new_count:
                padded = F.pad(prefix, (0,0,target_length-prefix.shape[-2],0))
                pieces.append(padded.expand(new_count,-1,-1,-1))
            merged.append(torch.cat(pieces,dim=0) if len(pieces)>1 else pieces[0])
        layers.append(tuple(merged))
    return tuple(layers)


def step(model, cache, pending_tokens, past_lengths, device):
    length = cache[0][0].shape[-2]
    positions = torch.tensor(past_lengths,device=device,dtype=torch.long).unsqueeze(1)
    columns = torch.arange(length+1,device=device).unsqueeze(0)
    mask = (columns >= length-positions).long()
    ids = torch.tensor(pending_tokens,device=device,dtype=torch.long).unsqueeze(1)
    out = model(input_ids=ids,position_ids=positions,attention_mask=mask,past_key_values=cache,use_cache=True)
    return out.logits[:, -1], as_legacy(out.past_key_values)


def sample(logits, temperature, uniforms):
    if temperature == 0:
        return logits.argmax(-1).tolist()
    # The shared protocol uses top-p=1. Independent per-path uniforms index each exact CDF.
    probabilities = (logits.float()/temperature).softmax(-1)
    cumulative = probabilities.cumsum(-1)
    cumulative[:, -1] = 1.
    u = torch.as_tensor(uniforms,device=logits.device,dtype=torch.float32)[:,None].clamp(max=1.-2.**-24)
    return torch.searchsorted(cumulative.contiguous(),u.contiguous(),right=True).squeeze(1).tolist()


def panel(model, tokenizer, prompt, q, repeat, policy, options, device, stops):
    indices = list(range(options.n))
    rngs = [random.Random(path_seed(q["qid"],repeat,i)) for i in indices]
    history = [[] for _ in indices]
    votes, paths, events = {}, {}, []
    active, previous_ids = [], []
    previous_cache = None
    launched, committed, ticks = 0, 0, 0
    # Clear references to the previous trial's cache before measuring current allocations.
    sync(device)
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    prefix = model(input_ids=prompt[:,:-1],use_cache=True)
    prompt_cache = as_legacy(prefix.past_key_values)
    del prefix
    prompt_last = int(prompt[0,-1])
    while True:
        remaining = options.n-len(votes)
        margin, winner = gap_and_winner(votes.values())
        if not remaining or (policy!="full" and margin>remaining):
            assert not active
            break
        if policy == "rolling" and committed == len(votes):
            committed += safe(remaining,margin)
        if policy == "rolling":
            width = min(options.slots,committed-len(votes))
        elif policy == "refill":
            width = min(options.slots,safe(remaining,margin))
        else:
            width = min(options.slots,remaining)
        admit = list(range(launched,launched+width-len(active)))
        if admit:
            events.append({"kind":"launch","tick":ticks,"time_s":time.perf_counter()-start,"ids":admit})
        launched += len(admit)
        active.extend(admit)
        assert len(active) <= options.slots
        past_lengths=[prompt.shape[1]-1+len(history[i]) for i in active]
        cached=align_cache(previous_ids,active,previous_cache,prompt_cache,past_lengths,device)
        previous_cache = None
        pending=[history[i][-1] if history[i] else prompt_last for i in active]
        logits, previous_cache=step(model,cached,pending,past_lengths,device)
        del cached
        next_tokens=sample(logits,options.temperature,[rngs[i].random() for i in active])
        ticks += 1
        previous_ids=active.copy()
        for i,tok in zip(previous_ids,next_tokens):
            history[i].append(tok)
            terminal = tok in stops
            if terminal or len(history[i])==options.max_tokens:
                reason="stop" if terminal else "length"
                text=tokenizer.decode(history[i],skip_special_tokens=True)
                votes[i]=parse_answer(text,reason)
                # Keep timing active through online answer decoding and vote parsing.
                paths[i]=dict(index=i,seed=path_seed(q["qid"],repeat,i),text=text,answer=votes[i],
                              finish_reason=reason,length=len(history[i]),finish_tick=ticks,
                              finish_s=time.perf_counter()-start,token_ids=history[i])
        active=[i for i in previous_ids if i not in votes]
    sync(device)
    elapsed=time.perf_counter()-start
    for path in paths.values():
        path["token_hash"]=hashlib.sha256(json.dumps(path["token_ids"],separators=(",",":")).encode()).hexdigest()
    return dict(qid=q["qid"],repeat=repeat,policy=policy,winner=winner,correct=winner==q["truth"],
                elapsed_s=elapsed,main_tokens=sum(len(t) for t in history),launched=launched,ticks=ticks,
                prompt_tokens=prompt.shape[1],events=events,paths=[paths[i] for i in sorted(paths)],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None)


def self_check():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(43)
    torch.set_num_threads(1)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=80,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
                                      num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)).eval()
    device=torch.device("cpu")
    prompt=[1,3,4,8,2]
    pk=as_legacy(model(torch.tensor([prompt[:-1]]),use_cache=True).past_key_values)
    cache=None
    old=[]
    tokens={i:[] for i in range(6)}
    comparisons=0
    max_error=0.
    cases=[[0,1],[0,1],[1,2,3],[2,3,4],[3,4],[4,5]]
    for tick,current in enumerate(cases):
        lengths=[len(prompt)-1+len(tokens[i]) for i in current]
        cache=align_cache(old,current,cache,pk,lengths,device)
        pending=[tokens[i][-1] if tokens[i] else prompt[-1] for i in current]
        logits,cache=step(model,cache,pending,lengths,device)
        for row,i in enumerate(current):
            expected=model(torch.tensor([prompt+tokens[i]]),use_cache=False).logits[0,-1]
            error=(logits[row]-expected).abs().max().item()
            max_error=max(max_error,error)
            torch.testing.assert_close(logits[row],expected,atol=2e-5,rtol=1e-5)
            tokens[i].append(20+i+tick)
            comparisons+=1
        old=current.copy()
    print(json.dumps(dict(check="Qwen2 row mixing against unpadded full-context logits",comparisons=comparisons,max_abs_error=max_error)))


def summarize(rows, out):
    refs={(r["qid"],r["repeat"]):r for r in rows if r["policy"]=="full"}
    summary=[]
    for r in rows:
        reference=refs[(r["qid"],r["repeat"])]
        rp={p["index"]:p for p in reference["paths"]}
        expected=[]
        for i in range(len(rp)):
            expected.append(rp[i]["answer"])
            if gap_and_winner(expected)[0] > len(rp)-i-1:
                break
        summary.append({**{k:r[k] for k in ("qid","repeat","policy","winner","correct","elapsed_s","warmup_s",
                            "main_tokens","launched","ticks","peak_allocated_bytes","peak_reserved_bytes")},
                        "full_winner_match":r["winner"]==reference["winner"],
                        "length_mismatches":sum(p["length"]!=rp[p["index"]]["length"] for p in r["paths"]),
                        "token_hash_mismatches":sum(p["token_hash"]!=rp[p["index"]]["token_hash"] for p in r["paths"]),
                        "vote_mismatches":sum(p["answer"]!=rp[p["index"]]["answer"] for p in r["paths"]),
                        "terminal_reason_mismatches":sum(p["finish_reason"]!=rp[p["index"]]["finish_reason"] for p in r["paths"]),
                        "serial_reference_prefix":len(expected),
                        "prefix_match":r["launched"]==len(expected) if r["policy"]!="full" else None})
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
        w=csv.DictWriter(stream,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    by={(r["qid"],r["repeat"],r["policy"]):r for r in rows}
    gains=[1-by[q,rep,"refill"]["elapsed_s"]/by[q,rep,"rolling"]["elapsed_s"] for q,rep in refs]
    aggregate=dict(paired_macro_latency_reduction=sum(gains)/len(gains),trial_pairs=len(gains),
                   hashed_path_mismatches=sum(r["token_hash_mismatches"] for r in summary if r["policy"]!="full"))
    (out/"aggregate.json").write_text(json.dumps(aggregate,indent=2))
    print(json.dumps(aggregate))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-check",action="store_true")
    p.add_argument("--model",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--dtype",choices=["bfloat16","float16","float32"],default="bfloat16")
    p.add_argument("--attn",choices=["sdpa","eager"],default="sdpa")
    p.add_argument("--n",type=int,default=8)
    p.add_argument("--slots","--capacity",dest="slots",type=int,default=4)
    p.add_argument("--questions",type=int,default=6)
    p.add_argument("--repeats",type=int,default=1)
    p.add_argument("--max-tokens",type=int,default=160)
    p.add_argument("--temperature",type=float,default=.7)
    opts=p.parse_args()
    if opts.self_check:
        self_check()
        return
    if opts.model is None or opts.out is None or min(opts.n,opts.slots,opts.questions,opts.repeats,opts.max_tokens)<1 or opts.slots>opts.n or opts.temperature<0:
        p.error("Supply --model, --out, positive panel counts and a nonnegative temperature.")
    opts.out.mkdir(parents=True,exist_ok=False)
    try:
        device=torch.device(opts.device)
        tokenizer=AutoTokenizer.from_pretrained(opts.model,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(opts.model,torch_dtype=getattr(torch,opts.dtype),
                    attn_implementation=opts.attn,local_files_only=True).to(device).eval()
        eos=model.generation_config.eos_token_id
        stops=set(eos if isinstance(eos,list) else [eos])
        if tokenizer.eos_token_id is not None: stops.add(tokenizer.eos_token_id)
        metadata=dict(args=vars(opts),python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
                      cuda=torch.version.cuda,device=torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu",
                      eos_ids=sorted(x for x in stops if x is not None),attention=model.config._attn_implementation,
                      code_hashes={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in
                                   ["hf_panel.py","workload.py"]},
                      hashes={f:hashlib.sha256((opts.model/f).read_bytes()).hexdigest() for f in
                              ["config.json","generation_config.json","tokenizer_config.json","model.safetensors.index.json"] if (opts.model/f).is_file()})
        (opts.out/"manifest.json").write_text(json.dumps(metadata,indent=2,default=str))
        policies=["rolling","refill","full"]
        queries=questions(opts.questions)
        random.Random(20261120).shuffle(queries)
        records=[]
        with (opts.out/"trials.jsonl").open("w",encoding="utf-8") as stream:
            for repeat in range(opts.repeats):
                for order,q in enumerate(queries):
                    prompt=tokenizer.apply_chat_template([{"role":"user","content":q["prompt"]}],add_generation_prompt=True,
                                                          return_tensors="pt").to(device)
                    if prompt.shape[1]<2: raise ValueError("Chat template must produce at least two tokens.")
                    cycle=repeat*len(queries)+order
                    offset=cycle%len(policies)
                    base=policies if (cycle//len(policies))%2==0 else policies[::-1]
                    for policy in base[offset:]+base[:offset]:
                        sync(device);warm=time.perf_counter()
                        warm_logits=model(input_ids=prompt,use_cache=True).logits[0,-1:]
                        warm_token=sample(warm_logits,opts.temperature,[random.Random(0).random()])[0]
                        del warm_logits
                        sync(device);warm_s=time.perf_counter()-warm
                        r=panel(model,tokenizer,prompt,q,repeat,policy,opts,device,stops)
                        r.update(prompt_text=q["prompt"],truth=q["truth"],warmup_s=warm_s,warmup_token_id=warm_token)
                        stream.write(json.dumps(r,ensure_ascii=False)+"\n");stream.flush()
                        records.append(r)
                        print(q["qid"],repeat,policy,r["elapsed_s"],r["main_tokens"],r["winner"],flush=True)
        summarize(records,opts.out)
    except BaseException:
        (opts.out/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()
