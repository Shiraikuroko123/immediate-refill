"""Retrospective three-request fixed-length replay, with a two-action oracle."""
import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from math import floor
from pathlib import Path
import random
from statistics import mean, median


HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PREDICTION = STUDY / "prospective-device/session01/prediction.json"
MANIFEST = STUDY / "holdout/freeze.json"
FROZEN = HERE / "freeze.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes():
    return {p.relative_to(STUDY).as_posix():sha(p) for p in
            (HERE/"PROTOCOL.md", HERE/"replay.py", MANIFEST, PREDICTION)}


def write_json(path,obj):
    with path.open("xb") as out:
        out.write((json.dumps(obj,indent=2,allow_nan=False)+"\n").encode("utf8"))


def inputs():
    manifest = json.loads(MANIFEST.read_bytes())
    integrity, questions, shards = {}, defaultdict(dict), {}
    for entry in manifest["selected"]:
        name = Path(entry["path"]).name
        content = (STUDY/"holdout/raw"/name).read_bytes()
        git_id = hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest()
        assert len(content)==entry["size"] and git_id==entry["oid"]
        integrity[name] = hashlib.sha256(content).hexdigest()
        for raw in content.splitlines():
            r = json.loads(raw)
            key, k = r["problem_key"], r["rollout_index"]
            assert k not in questions[key]
            assert type(r["output_tokens"]) is int and r["output_tokens"]>0 and type(r["hit_token_cap"]) is bool
            questions[key][k] = (r["output_tokens"],r["hit_token_cap"])
            assert shards.setdefault(key,name)==name
    assert len(questions)==200 and all(set(paths)==set(range(16)) for paths in questions.values())
    return questions, shards, integrity


def occupancy(lengths,slots):
    active = []
    next_lane = 0
    occupancy = Counter()
    while active or next_lane<len(lengths):
        while len(active)<slots and next_lane<len(lengths):
            active.append(lengths[next_lane])
            next_lane+=1
        increment=min(active)
        occupancy[len(active)]+=increment
        active=[w-increment for w in active if w>increment]
    return [occupancy[k] for k in (1,2,3)]


def measure(key,shard,paths,clock):
    lengths=[paths[i][0] for i in range(3)]
    caps=[paths[i][1] for i in range(3)]
    im=occupancy(lengths,3)
    st=occupancy(lengths,2)
    a,b,c=sorted(lengths)
    assert im==[c-b,b-a,a]
    head=min(lengths[:2])
    remaining=abs(lengths[0]-lengths[1])
    last=lengths[2]
    assert st==[abs(remaining-last),head+min(remaining,last),0]
    assert sum((i+1)*v for i,v in enumerate(st))==sum((i+1)*v for i,v in enumerate(im))==sum(lengths)
    immediate=sum(k*w for k,w in zip(im,clock))
    staged=sum(k*w for k,w in zip(st,clock))
    delta=immediate-staged
    row=dict(question=key,shard=shard)
    for i in range(3):
        row.update({f"length_{i}":lengths[i],f"hit_cap_{i}":int(caps[i]),
                    f"immediate_width_{i+1}_steps":im[i],f"staged_width_{i+1}_steps":st[i]})
    row.update(immediate_ms=immediate,staged_ms=staged,gain_ms=delta,gain_pct=100*delta/immediate,
               oracle_ms=min(immediate,staged),oracle_gain_pct=100*max(0,delta)/immediate)
    return row


def interval(values):
    ordered=sorted(values)
    def quantile(p):
        target=(len(ordered)-1)*p
        i=floor(target)
        return ordered[i]+(ordered[i+1]-ordered[i])*(target-i)
    return [quantile(.025),quantile(.975)]


def summarize(rows):
    n=len(rows)
    effects=[r["gain_pct"] for r in rows]
    upper=[r["oracle_gain_pct"] for r in rows]
    rng=random.Random(20261120)
    samples,oracle_samples=[],[]
    for _ in range(10000):
        idx=rng.choices(range(n),k=n)
        samples.append(sum(effects[i] for i in idx)/n)
        oracle_samples.append(sum(upper[i] for i in idx)/n)
    all_immediate=sum(r["immediate_ms"] for r in rows)
    all_staged=sum(r["staged_ms"] for r in rows)
    all_opt=sum(r["oracle_ms"] for r in rows)
    caps=[sum(r[f"hit_cap_{i}"] for i in range(3)) for r in rows]
    lengths=[r[f"length_{i}"] for r in rows for i in range(3)]
    return dict(questions=n,paths=len(lengths),min_length=min(lengths),max_length=max(lengths),
                capped_paths=sum(caps),questions_with_capped_paths=sum(v>0 for v in caps),
                mean_gain_pct=mean(effects),median_gain_pct=median(effects),
                mean_gain_pct_descriptive_interval=interval(samples),
                modeled_total_ms=dict(immediate=all_immediate,staged=all_staged,oracle=all_opt),
                modeled_time_difference_ms=all_immediate-all_staged,
                pooled_gain_pct=100*(all_immediate-all_staged)/all_immediate,
                positive=sum(e>0 for e in effects),negative=sum(e<0 for e in effects),equal=sum(e==0 for e in effects),
                at_least_one_pct=sum(e>=1 for e in effects),at_least_two_pct=sum(e>=2 for e in effects),
                oracle_mean_gain_pct=mean(upper),oracle_pooled_gain_pct=100*(all_immediate-all_opt)/all_immediate,
                oracle_mean_gain_pct_descriptive_interval=interval(oracle_samples))


def run(out):
    frozen=json.loads(FROZEN.read_bytes())
    assert frozen["sha256"]==hashes()
    clock=json.loads(PREDICTION.read_bytes())["clock_ms"]
    questions,shards,integrity=inputs()
    rows=[measure(key,shards[key],questions[key],clock) for key in sorted(questions)]
    out.mkdir(parents=True,exist_ok=False)
    with (out/"questions.csv").open("x",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result=dict(frozen_sha256=sha(FROZEN),csv_sha256=sha(out/"questions.csv"),clock_ms=clock,
                raw_sha256=integrity,retrospective=True,machine_timings_on_these_questions=False,
                **summarize(rows))
    write_json(out/"summary.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze",action="store_true")
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    if args.freeze:
        write_json(FROZEN,dict(utc=datetime.now(timezone.utc).isoformat(),sha256=hashes()))
    elif args.out:
        run(args.out)
    else:
        p.error("Supply --out with a new directory")
