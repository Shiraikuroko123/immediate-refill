"""Freeze GSM8K shard selection, fetch verified trajectory inputs, and price one-slot interventions."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median
import sys

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/"tracks/sharing/applicability"))
from audit import load_all, simulate, tally, target
from amc import boxed
sys.path.insert(0, str(ROOT/"tracks/sharing/global"))
from replay import read_curves

REV = "b8c33b85a1a6cdfb23b06a5f9d696f52950780e4"
DATASET = "reasoning-cues/rollouts-qwen4b"
TREE = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REV}/boxed/gsm8k/run"
N, Q = 16, 14


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    with path.open("x", encoding="utf8") as out:
        out.write(json.dumps(obj, indent=2)+"\n")


def freeze():
    url = TREE + "?limit=1000"
    files = []
    while url:
        response = requests.get(url, timeout=(10, 35))
        response.raise_for_status()
        files.extend(v for v in response.json() if re.fullmatch(
            r"boxed/gsm8k/run/rollouts_base_none_p[0-9]{5}\.jsonl", v["path"]))
        url = response.links.get("next", {}).get("url")
    assert len(files) >= 8 and len({f["path"] for f in files}) == len(files)
    selected = sorted(files, key=lambda f: digest(("four-gsm-v1/"+f["path"]).encode()))[:8]
    write_json(HERE/"freeze.json", dict(utc=datetime.now(timezone.utc).isoformat(), tree=TREE,
        candidates=len(files), selected=selected, protocol=digest((HERE/"PROTOCOL.md").read_bytes()),
        code=digest(Path(__file__).read_bytes())))


def inputs(frozen):
    groups = defaultdict(list)
    raw = HERE/"raw"
    raw.mkdir(exist_ok=True)
    hashes = {}
    for f in frozen["selected"]:
        name = Path(f["path"]).name
        file = raw/name
        if file.exists():
            content = file.read_bytes()
        else:
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REV}/"+f["path"]
            response = requests.get(url, timeout=(10, 60))
            response.raise_for_status()
            content = response.content
        assert len(content) == f["size"]
        assert hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest() == f["oid"]
        if not file.exists():
            file.write_bytes(content)
        hashes[name] = digest(content)
        for line in content.splitlines():
            row = json.loads(line)
            work = row["output_tokens"]
            assert type(work) is int and work > 0
            vote = None if row["hit_token_cap"] else boxed(row["completion"])
            groups[row["problem_key"]].append((row["rollout_index"], work, vote))
    return groups, hashes


def schedule(lengths, votes, arm, intervene=None):
    ends, seen, starts = {}, {}, []
    occupancy = [0]*4
    now = 0
    expiry = None
    used = False
    epochs = []
    while True:
        winner, gap = tally(seen.values())
        certified = gap > N-len(seen)
        if certified or (len(starts) == Q and not ends):
            break
        limit = target("buffer" if arm=="buffer" else "exact", Q, seen)
        active = len(ends)
        eligible = now > 0 and limit > len(starts) and active+limit-len(starts) >= 2
        if eligible:
            epochs.append(now)
        gap_trigger = eligible and active > 0 and gap+2*active > N-len(seen)
        capped = (not used and (arm=="first_gap_8" and gap_trigger or arm=="first_any_8" and eligible))
        if capped:
            used = True
            expiry = now+8
        if capped or arm=="repeat_gap" and gap_trigger or arm=="one" and intervene==now:
            limit -= 1
        for i in range(len(starts),limit):
            starts.append(now)
            ends[i] = now+lengths[i]
        assert 0 < len(ends) <= 4
        next_time = min(ends.values())
        if expiry is not None:
            next_time = min(next_time, expiry)
            expiry = None
        occupancy[len(ends)-1] += next_time-now
        now = next_time
        for i in [i for i, end in ends.items() if end==now]:
            seen[i] = votes[i]
            del ends[i]
    return dict(occ=occupancy, starts=starts, epochs=epochs, ticks=now,
                tokens=sum((i+1)*d for i,d in enumerate(occupancy)),
                winner=winner,certified=certified)


def outcomes(key, group, curves, cohort="GSM"):
    group = sorted(group)
    lengths = tuple(v[1] for v in group)
    votes = tuple(v[2] for v in group)
    baseline = schedule(lengths,votes,"exact")
    ref = simulate(lengths,votes,"exact")
    assert all(baseline[k] == ref[j] for k,j in (("starts","starts"),("tokens","tokens"),
            ("ticks","steps"),("certified","certified"),("winner","winner")))
    choices = {arm:schedule(lengths,votes,arm) for arm in ("buffer","repeat_gap","first_gap_8","first_any_8")}
    one = [baseline]+[schedule(lengths,votes,"one",e) for e in baseline["epochs"]]
    for alt in [*choices.values(),*one]:
        assert alt["ticks"]>=baseline["ticks"]
        assert (alt["certified"],alt["winner"]) == (baseline["certified"],baseline["winner"])
    robust = 0
    for alt in one[1:]:
        diff = [sum(alt["occ"][i:])-sum(baseline["occ"][i:]) for i in range(4)]
        robust += int(max(diff)<=0 and min(diff)<0)
    for arm in ("first_gap_8","first_any_8"):
        assert choices[arm]["ticks"]-baseline["ticks"]<=8
    for context,c in curves.items():
        def price(s):
            return sum(v*t for v,t in zip(c,s["occ"]))
        times = {**choices,"oracle_one":min(one,key=price)}
        for name, alt in times.items():
            other = (name=="buffer")
            base, alt = (alt,baseline) if other else (baseline,alt)
            yield dict(cohort=cohort, question=key, arm="indexed_vs_buffer" if other else name,
                       context=context,base_time=price(base),time=price(alt),
                       base_tokens=base["tokens"],tokens=alt["tokens"],
                       changed=int(base["starts"]!=alt["starts"]),
                       first_changed_index=next((i for i,(a,b) in enumerate(zip(base["starts"],alt["starts"])) if a!=b),
                                                min(len(base["starts"]),len(alt["starts"])) if len(base["starts"])!=len(alt["starts"]) else None),
                       eligible=len(baseline["epochs"]),dominant_events=robust)


def compute(out, clock):
    frozen = json.loads((HERE/"freeze.json").read_bytes())
    assert frozen["code"]==digest(Path(__file__).read_bytes()) and frozen["protocol"]==digest((HERE/"PROTOCOL.md").read_bytes())
    groups, hashes = inputs(frozen)
    curves = {n:list(map(float,v["median"])) for n,v in read_curves(clock,4).items()}
    curves[0] = [1.]*4
    detail, excluded = [], []
    for key, records in sorted(groups.items()):
        if len(records)!=N or len({r[0] for r in records})!=N:
            excluded.append([key,len(records),len({r[0] for r in records})])
            continue
        detail.extend(outcomes(key,records,curves))
    if not detail:
        raise ValueError("No complete 16-path groups")
    out.mkdir(parents=True,exist_ok=False)
    with (out/"questions.csv").open("x",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0])); w.writeheader(); w.writerows(detail)
    strata=defaultdict(list)
    for v in detail:
        strata[v["context"],v["arm"]].append(v)
    summary=[]
    for (context,arm), rs in strata.items():
        gain=np.array([1-v["time"]/v["base_time"] for v in rs])
        rng=np.random.default_rng(19431)
        bs=np.mean(rng.choice(gain,(10000,len(rs)),replace=True),axis=1)
        summary.append(dict(context=context,arm=arm,n=len(rs),mean=float(np.mean(gain)),median=float(np.median(gain)),
                            positive=int(np.sum(gain>1e-12)),negative=int(np.sum(gain< -1e-12)),
                            changed=sum(v["changed"] for v in rs),interval=np.quantile(bs,[.025,.975]).tolist(),
                            pooled=1-sum(v["time"] for v in rs)/sum(v["base_time"] for v in rs),
                            mean_token_saving=mean(1-v["tokens"]/v["base_tokens"] for v in rs)))
    write_json(out/"summary.json",dict(summary=summary,source_hashes=hashes,excluded=excluded,
        file_records=sum(len(x) for x in groups.values()),num_questions=len(groups),clock_sha256=digest(clock.read_bytes()),
        robust_questions=sum(x["dominant_events"]>0 for x in strata[3968,"oracle_one"]),
        robust_events=sum(x["dominant_events"] for x in strata[3968,"oracle_one"])))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("freeze","run"))
    p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--clock",type=Path,default=ROOT/"tracks/interference/hardware/a800-live02/events.jsonl")
    args=p.parse_args()
    if args.mode=="freeze":
        freeze()
    else:
        compute(args.out,args.clock)
