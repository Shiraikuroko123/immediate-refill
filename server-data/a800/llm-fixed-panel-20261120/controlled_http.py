"""Measure HTTP admission on a previously frozen length-and-vote workload."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import http.client
import json
from pathlib import Path
import random
import statistics
import time
import traceback
from urllib.parse import urlsplit

from run import counts, dump, log, sha, stopping_prefix


def load_profile(path):
    panels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["policy"] != "full":
            continue
        key = row["qid"], row["repeat"]
        paths = sorted(row["paths"], key=lambda p: p["index"])
        if key in panels or [p["index"] for p in paths] != list(range(row["launched"])):
            raise ValueError(f"Invalid full reference panel {key}")
        for p in paths:
            if p["length"] < 1 or p["length"] != len(p["token_ids"]):
                raise ValueError(f"Invalid cached length in {key}")
            encoded = json.dumps(p["token_ids"], separators=(",",":"))
            if sha(encoded) != p["token_hash"]:
                raise ValueError(f"Invalid cached token hash in {key}")
        if sum(p["length"] for p in paths) != row["main_tokens"]:
            raise ValueError(f"Invalid corpus token accounting in {key}")
        panels[key] = {"prompt": row["prompt_text"], "paths":paths, "winner":row["winner"]}
    if not panels:
        raise ValueError("No full-budget panels")
    return panels


def generate(options, prompt, seed, length):
    address = urlsplit(options.base_url)
    payload = dict(model=options.model, temperature=options.temperature, top_p=1.0,
                   seed=seed, min_tokens=length, max_tokens=length, ignore_eos=True, stream=False)
    if options.api_mode == "chat":
        route="/chat/completions"
        payload["messages"]=[{"role":"user","content":prompt}]
    else:
        route="/completions"
        payload["prompt"]=prompt
    conn = http.client.HTTPConnection(address.hostname, address.port, timeout=options.timeout)
    started=time.monotonic()
    try:
        conn.request("POST",address.path.rstrip("/")+route,body=json.dumps(payload),
                     headers={"Content-Type":"application/json"})
        response=conn.getresponse()
        body=response.read()
        arrived=time.monotonic()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body[:1000]!r}")
        raw=json.loads(body)
        choice=raw["choices"][0]
        used=raw["usage"]["completion_tokens"]
        if used != length:
            raise RuntimeError(f"Requested {length} controlled tokens, reported {used}: {body[:300]!r}")
        generated = choice.get("text") if options.api_mode=="text" else choice["message"].get("content")
        message = choice.get("message",{})
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        return dict(seed=seed,required_tokens=length,usage=raw["usage"],finish_reason=choice["finish_reason"],
                    start_clock=started,arrival_clock=arrived,request_s=arrived-started,text_hash=sha(generated or ""),
                    generated_text_hash=sha(json.dumps([reasoning,generated or ""],ensure_ascii=False)),
                    raw_response=raw)
    finally:
        conn.close()


def evaluate(options, pool, key, source, policy):
    qid,rep=key
    identity=dict(qid=qid,repeat=rep,policy=policy)
    archived=source["paths"]
    n=len(archived)
    warm=generate(options,source["prompt"],archived[0]["seed"],1)
    log(options.out/"warmups.jsonl",{**identity,**warm})
    active, completed, votes, events = {}, {}, [], []
    committed=launched=0
    start=time.monotonic()
    try:
        while True:
            winner,gap=counts(votes)
            remaining=n-len(votes)
            if remaining==0 or gap>remaining:
                break
            minimum=min(remaining,(remaining-gap)//2+1)
            if policy=="rolling":
                if committed==len(votes):
                    committed+=minimum
                width=min(options.capacity,committed-len(votes))
            else:
                width=min(options.capacity,minimum)
            first=launched
            while len(active)<width and launched<n:
                path=archived[launched]
                task=pool.submit(generate,options,source["prompt"],path["seed"],path["length"])
                active[task]=launched
                launched+=1
            events.append(dict(t_s=time.monotonic()-start,event="decision",completed=len(votes),gap=gap,
                               width=width,committed=committed if policy=="rolling" else None,
                               admitted=list(range(first,launched))))
            ready,_=wait(active,return_when=FIRST_COMPLETED)
            finished=[]
            for task in ready:
                idx=active.pop(task)
                finished.append((idx,task.result()))
            for idx,row in sorted(finished,key=lambda item:(item[1]["arrival_clock"],item[0])):
                # Only a successful length-verified completion releases its archived vote.
                vote=archived[idx]["answer"]
                votes.append(vote)
                completed[idx]={**row,"index":idx,"released_vote":vote,"arrival_s":row["arrival_clock"]-start}
                events.append(dict(t_s=time.monotonic()-start,event="release",index=idx,answer=vote))
        elapsed=time.monotonic()-start
        expected=stopping_prefix([p["answer"] for p in archived])
        if active or sorted(completed)!=list(range(expected)) or winner!=source["winner"]:
            raise RuntimeError(f"Controlled task broke the work or vote invariant for {identity}")
        for idx in sorted(completed):
            log(options.out/"paths.jsonl",{**identity,**completed[idx]})
        for event in events:
            log(options.out/"events.jsonl",{**identity,**event})
        record=dict(**identity,decision_seconds=elapsed,launched=launched,reference_prefix=expected,winner=winner,
                    generation_tokens=sum(r["usage"]["completion_tokens"] for r in completed.values()))
        log(options.out/"panels.jsonl",record)
        print(qid,rep,policy,round(elapsed,4),record["generation_tokens"],flush=True)
        return record,completed
    except BaseException:
        log(options.out/"partial_trials.jsonl",dict(**identity,events=events,completed=list(completed.values()),
                                                   pending=sorted(active.values()),error=traceback.format_exc()))
        raise


def main(options):
    panels=load_profile(options.profile)
    qids=sorted({q for q,rep in panels})
    if options.questions is not None:
        qids=qids[:options.questions]
    repeats=sorted({rep for q,rep in panels})[:options.repeats]
    if not repeats or not qids or any((q,r) not in panels for q in qids for r in repeats):
        raise ValueError("Incomplete reference qid/repetition grid")
    random.Random(20261120).shuffle(qids)
    options.out.mkdir(parents=True,exist_ok=False)
    dump(options.out/"manifest.json",dict(profile_sha256=hashlib.sha256(options.profile.read_bytes()).hexdigest(),
         client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qids=qids,repeats=repeats,
         args={**vars(options),"profile":str(options.profile),"out":str(options.out)},
         workload="real model decoding to frozen per-path lengths, followed by release of the archived vote"))
    pairs=[]
    try:
        with ThreadPoolExecutor(options.capacity) as pool:
            for j,rep in enumerate(repeats):
                for number,q in enumerate(qids):
                    # Each prompt alternates which policy executes first across repetitions.
                    order=["rolling","refill"] if (j+number)%2==0 else ["refill","rolling"]
                    log(options.out/"order.jsonl",dict(qid=q,repeat=rep,order=order))
                    group={policy:evaluate(options,pool,(q,rep),panels[q,rep],policy) for policy in order}
                    a,ap=group["refill"]
                    b,bp=group["rolling"]
                    assert sorted(ap)==sorted(bp) and a["generation_tokens"]==b["generation_tokens"]
                    changes=sum(ap[i]["generated_text_hash"]!=bp[i]["generated_text_hash"] for i in ap)
                    prompt_tokens_differ=sum(ap[i]["usage"].get("prompt_tokens")!=bp[i]["usage"].get("prompt_tokens") for i in ap)
                    if prompt_tokens_differ:
                        raise RuntimeError("Matched requests reported different prompt lengths")
                    row=dict(qid=q,repeat=rep,order=order,rolling_seconds=b["decision_seconds"],
                             refill_seconds=a["decision_seconds"],paired_reduction=1-a["decision_seconds"]/b["decision_seconds"],
                             generated_tokens_each=a["generation_tokens"],matched_paths=len(ap),
                             generated_text_hash_mismatches=changes,prompt_tokens_mismatches=prompt_tokens_differ)
                    log(options.out/"pairs.jsonl",row)
                    pairs.append(row)
        per_q={qid:statistics.mean(row["paired_reduction"] for row in pairs if row["qid"]==qid) for qid in qids}
        dump(options.out/"summary.json",dict(questions=len(qids),pairs=len(pairs),per_question=per_q,
             macro_paired_wall_reduction=statistics.mean(per_q.values()),
             pooled_wall_reduction=1-sum(r["refill_seconds"] for r in pairs)/sum(r["rolling_seconds"] for r in pairs),
             equal_generated_tokens_per_policy=sum(r["generated_tokens_each"] for r in pairs),
             matched_paths=sum(r["matched_paths"] for r in pairs),
             generated_text_hash_mismatches=sum(r["generated_text_hash_mismatches"] for r in pairs)))
    except BaseException:
        (options.out/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--profile",required=True,type=Path,help="HF full-panel reference trials.jsonl")
    parser.add_argument("--model",required=True,help="Served model name from /v1/models")
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--base-url",default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-mode",choices=["chat","text"],default="chat")
    parser.add_argument("--capacity",type=int,default=4)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--questions",type=int)
    parser.add_argument("--temperature",type=float,default=.7)
    parser.add_argument("--timeout",type=int,default=180)
    args=parser.parse_args()
    if min(args.capacity,args.repeats,args.timeout)<1 or (args.questions is not None and args.questions<1):
        parser.error("Counts must be positive")
    main(args)
