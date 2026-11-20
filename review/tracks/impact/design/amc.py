"""Two-stage, length-only forecast on separate public AMC23 generation batches."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, pstdev
import sys

import numpy as np
import requests

BASE = Path(__file__).resolve().parent
QUOTA = BASE.parents[1] / "quota"
sys.path.insert(0, str(QUOTA / "transfer_diag"))
from run import simulate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf8", newline="\n") as out:
        out.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    with path.open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boxed(text):
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for idx in range(start, len(text)):
        depth += (text[idx] == "{") - (text[idx] == "}")
        if depth == 0:
            return re.sub(r"\s+", "", text[start:idx]) or None
    return None


def fetch(batch, cache):
    pool = defaultdict(list)
    records = []
    cache.mkdir(exist_ok=True, parents=True)
    for src in json.loads((BASE / "source.json").read_bytes())["files"]:
        if src["path"].endswith("_b2.jsonl") != (batch == 2):
            continue
        path = cache / Path(src["path"]).name
        if not path.exists():
            url = ("https://huggingface.co/datasets/reasoning-cues/rollouts-qwen4b/resolve/"
                   "b8c33b85a1a6cdfb23b06a5f9d696f52950780e4/" + src["path"])
            response = requests.get(url, timeout=(15, 60))
            response.raise_for_status()
            content = response.content
        else:
            content = path.read_bytes()
        git_oid = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if git_oid != src["oid"] or len(content) != src["size"]:
            raise ValueError(f"Corrupt source {path.name}")
        if not path.exists():
            path.write_bytes(content)
        records.append({"file": path.name, "sha256": digest(path), "git_oid": git_oid})
        for line in content.splitlines():
            row = json.loads(line)
            length = row["output_tokens"]
            if type(length) is not int or not 0 < length <= 31744:
                raise ValueError("Invalid reported output token count")
            vote = None if batch == 1 or row["hit_token_cap"] else boxed(row["completion"])
            pool[row["problem_key"]].append((row["rollout_index"], length, vote))
    for key, rows in pool.items():
        rows.sort()
        if len({r[0] for r in rows}) != 16 or len(rows) != 16:
            raise ValueError(f"Incomplete problem {key}")
    if not pool or len(records) != 2:
        raise ValueError("Missing input shard")
    return pool, records


def run(phase, out, cache):
    frozen = json.loads((BASE / "freeze.json").read_bytes())
    for rel, expected in frozen["sha256"].items():
        if digest(BASE / rel) != expected:
            raise ValueError(f"Frozen source changed: {rel}")
    out.mkdir(parents=True, exist_ok=True)
    if phase == "predict":
        pool, sources = fetch(1, cache)
        rows = []
        for key, panel in sorted(pool.items()):
            lengths = [r[1] for r in panel]
            rows.append(dict(question=key, paths=len(panel), pilot_tokens=sum(lengths),
                             cv=pstdev(lengths) / mean(lengths)))
        write_csv(out / "predictions.csv", rows)
        write_json(out / "forecast.json", dict(utc=datetime.now(timezone.utc).isoformat(),
                   sha256=digest(out / "predictions.csv"), sources=sources))
        return
    forecast = json.loads((out / "forecast.json").read_bytes())
    if digest(out / "predictions.csv") != forecast["sha256"]:
        raise ValueError("Forecast changed")
    with (out / "predictions.csv").open(encoding="utf8", newline="") as stream:
        predictions = {r["question"]: float(r["cv"]) for r in csv.DictReader(stream)}
    pool, sources = fetch(2, cache)
    if predictions.keys() != pool.keys():
        raise ValueError("Question sets differ")
    rows = []
    for key, panel in sorted(pool.items()):
        lengths = tuple(r[1] for r in panel)
        votes = tuple(r[2] for r in panel)
        exact, buffer = (simulate(lengths, votes, gate) for gate in ("exact", "buffer"))
        rows.append(dict(question=key, cv=predictions[key], changed=int(exact["starts"] != buffer["starts"]),
                         exact=exact["steps"], buffer=buffer["steps"], saving=1-exact["steps"]/buffer["steps"],
                         abstentions=sum(v is None for v in votes), output_tokens=sum(lengths)))
    yes = np.array([r["cv"] for r in rows if r["changed"]])
    no = np.array([r["cv"] for r in rows if not r["changed"]])
    auroc, interval, decision = None, None, "unidentifiable"
    if len(yes) and len(no):
        def auc(y, n):
            diff = y[:, None] - n[None, :]
            return float(((diff > 0).sum() + .5 * (diff == 0).sum()) / diff.size)
        auroc = auc(yes, no)
        rng = np.random.default_rng(24601)
        bootstrap = [auc(rng.choice(yes, size=len(yes)), rng.choice(no, size=len(no))) for _ in range(20000)]
        interval = np.quantile(bootstrap, [.025, .975]).tolist()
        decision = ("replicate" if auroc >= .65 and interval[0] > .50 else
                    "reject-target" if interval[1] < .65 else "inconclusive")
    write_csv(out / "questions.csv", rows)
    write_json(out / "result.json", dict(utc=datetime.now(timezone.utc).isoformat(),
               questions=len(rows), changed=len(yes), unchanged=len(no), auroc=auroc, interval=interval,
               decision=decision, mean_modeled_saving=mean(x["saving"] for x in rows),
               faster=sum(x["saving"] > 0 for x in rows), forecast_sha256=forecast["sha256"],
               sources=sources, abstentions=sum(x["abstentions"] for x in rows)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("predict", "evaluate"))
    cli.add_argument("--out", type=Path, default=BASE/"complete")
    cli.add_argument("--cache", type=Path, default=BASE/"raw")
    args = cli.parse_args()
    run(args.phase, args.out, args.cache)
