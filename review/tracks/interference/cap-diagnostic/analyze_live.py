"""Audit 32 complete five-run blocks before estimating the paired cap interaction."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import runpy
from statistics import mean, median
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT / "tracks/quota"))
from summarize import audit_events, earliest, grade_integer, path_checks, tally


CONDITIONS = ((384, "exact"), (384, "buffer"), (1024, "exact"), (1024, "buffer"))
ORDERS = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
NAMESPACE = "gsm-cap-reference-v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), sha(raw)


def checked_row(row, *, qid, item, cap, arm, position, condition, ref_hash):
    assert row["qid"] == qid and row["unique_id"] == sha(item["problem"].encode()) == item["unique_id"]
    assert (row["policy"], row["cap"], row["position"], row["condition"]) == (arm, cap, position, condition)
    assert row["reference_sha256"] == ref_hash and row["seed_namespace"] == NAMESPACE
    assert row["warmup_seconds"] > 0 and len(row["warmup_ids"]) == 1
    for i, path in enumerate(row["paths"]):
        key = hashlib.sha256(f"{NAMESPACE}/{row['unique_id']}/{i}".encode()).digest()
        assert path["seed"] == int.from_bytes(key[:4], "big") % (2**31 - 1)
    audit_events(row)


def project(full, cap):
    projected = []
    for path in full["paths"]:
        length = min(path["length"], cap)
        vote = path["vote"] if length < cap else None
        projected.append({**path, "length": length, "token_ids": path["token_ids"][:length], "vote": vote})
    return {"paths": projected}


def median_costs():
    path = STUDY / "hardware/a800-live02/events.jsonl"
    cells = defaultdict(list)
    for row in map(json.loads, path.read_bytes().splitlines()):
        if row["event"] == "trial" and row["context"] == 3968:
            cells[row["width"]].append(row["elapsed_s"] / row["steps"])
    assert set(cells) == {1, 2, 3, 4} and all(len(v) == 8 for v in cells.values())
    return [median(cells[w]) for w in range(1, 5)], sha(path.read_bytes())


def summarize(values, draws):
    vals = np.asarray(values, dtype=float)
    interval = np.quantile(vals[draws].mean(axis=1), [.025, .975])
    return dict(mean=float(vals.mean()), median=float(np.median(vals)), ci=interval.tolist())


def inspect(raw, selection):
    items = selection["questions"]
    assert len(items) == len({q["unique_id"] for q in items}) == 32
    assert selection["protocol_sha256"] == sha((HERE / "PROTOCOL.md").read_bytes())
    costs, clock_sha = median_costs()
    schedule = runpy.run_path(str(STUDY / "holdout/run.py"))["schedule"]
    records, summary_rows, pairs = [], [], []
    configs, found, live_hashes = set(), set(), {}
    for qid, item in enumerate(items):
        def load(name):
            path = raw / f"q{qid:02d}-{name}.json"
            content, file_hash = read(path)
            found.add(path.name)
            live_hashes[path.name] = file_hash
            configs.add((content["protocol_hash"], content["config_hash"]))
            return content, file_hash

        full, ref_hash = load("natural-full")
        checked_row(full, qid=qid, item=item, cap=1024, arm="full", position=0, condition="natural", ref_hash=None)
        path_checks(full, full)
        assert full["completed"] == len(full["paths"]) == 16 and full["canceled"] == 0
        assert all(p["length"] <= 1024 and (p["length"] < 1024 or p["vote"] is None) for p in full["paths"])
        records.append(full)
        by_cap = {}
        order = [CONDITIONS[j] for j in ORDERS[qid % 4]]
        for cap in (384, 1024):
            ref = project(full, cap)
            lengths = [p["length"] for p in ref["paths"]]
            votes = [p["vote"] for p in ref["paths"]]
            serial = earliest(dict(enumerate(votes)))
            full_winner = tally(votes)[0]
            endpoints = {}
            mod = {}
            for arm in ("exact", "buffer"):
                row, _ = load(f"{cap}-{arm}")
                checked_row(row, qid=qid, item=item, cap=cap, arm=arm, position=1 + order.index((cap, arm)),
                            condition="fixed", ref_hash=ref_hash)
                path_checks(row, ref)
                assert row["reference_serial_prefix"] == serial and row["launched"] <= min(14, serial + 1)
                assert row["reference_winner"] == full_winner
                assert not row["certified"] or row["winner"] == full_winner
                records.append(row)
                endpoints[arm] = row
                mod[arm] = schedule(lengths, votes, arm)
                assert (row["winner"], row["certified"]) == (mod[arm]["winner"], mod[arm]["certified"])
            exact, buffered = endpoints["exact"], endpoints["buffer"]
            assert (exact["winner"], exact["certified"]) == (buffered["winner"], buffered["certified"])
            changed = [p["launch_step"] for p in exact["paths"]] != [p["launch_step"] for p in buffered["paths"]]
            modeled_changed = mod["exact"]["starts"] != mod["buffer"]["starts"]
            model_time = {arm: sum(c * d for c, d in zip(costs, z["occ"])) for arm, z in mod.items()}
            row = dict(qid=qid, question_sha256=item["unique_id"], cap=cap, reference_sha256=ref_hash,
                       capped=sum(w == cap for w in lengths), abstaining_paths=votes.count(None),
                       gold=item["gold"], full_winner=full_winner,
                       full_correct=int(grade_integer(full_winner, item["gold"])=="correct"),
                       winner=exact["winner"], correct=int(grade_integer(exact["winner"], item["gold"])=="correct"),
                       certified=int(exact["certified"]), changed=int(changed),
                       fewer_steps=int(exact["steps"] < buffered["steps"]),
                       saving=1 - exact["wall_seconds"] / buffered["wall_seconds"],
                       modeled_saving=1 - model_time["exact"]/model_time["buffer"],
                       modeled_changed=int(modeled_changed), dispatch_prediction_correct=int(changed==modeled_changed))
            for arm, endpoint in endpoints.items():
                for key in ("position","steps","generation_tokens","canceled_tokens","launched","completed",
                            "canceled","wall_seconds","peak_allocated_bytes"):
                    row[f"{arm}_{key}"] = endpoint[key]
            by_cap[cap] = row
            summary_rows.append(row)
        short, long = by_cap[384], by_cap[1024]
        pairs.append(dict(qid=qid, question_sha256=item["unique_id"], capped_short=short["capped"],
                          capped_long=long["capped"], saving_short=short["saving"], saving_long=long["saving"],
                          delta=long["saving"] - short["saving"], changed_short=short["changed"],
                          changed_long=long["changed"], changed_delta=long["changed"] - short["changed"],
                          modeled_delta=long["modeled_saving"] - short["modeled_saving"],
                          change_of_winner=int(long["winner"] != short["winner"]),
                          same_work_and_ballots=int(all(p["length"]<384 or
                                  (p["length"]==384 and p["vote"] is None) for p in full["paths"]))))
    assert len(found) == 160 and found == {p.name for p in raw.glob("q*-*.json")}
    assert len(configs) == 1
    protocol_hash, config_hash = configs.pop()
    cfg_path = raw / f"config-{config_hash}.json"
    cfg, fingerprint = read(cfg_path)
    assert sha(json.dumps(cfg, sort_keys=True).encode()) == config_hash
    assert sha(json.dumps(cfg["protocol"], sort_keys=True).encode()) == cfg["protocol_hash"] == protocol_hash
    protocol=cfg["protocol"]
    assert (protocol["n"],protocol["slots"],protocol["quota"],protocol["caps"],protocol["allowance"]) == (16,4,14,[384,1024],1)
    assert cfg["vllm"] == "0.8.5" and "A800" in cfg["gpu"] and protocol["dtype"] == "bfloat16"
    assert protocol["selection_sha256"] == sha((HERE/"selection.json").read_bytes())
    frozen, _=read(HERE/"freeze.json")
    assert frozen==protocol["code_sha256"]
    assert all(sha((ROOT / name).read_bytes())==digest for name,digest in frozen.items())
    live_hashes[cfg_path.name] = fingerprint
    for path in sorted(raw.glob("driver*.json")):
        driver, digest=read(path)
        assert driver["freeze"]==frozen
        live_hashes[path.name]=digest
    return summary_rows, pairs, records, live_hashes, clock_sha


def analyze(raw, selection_path, out):
    selection, selection_hash = read(selection_path)
    rows, pairs, records, live_hashes, clock_sha = inspect(raw, selection)
    n = len(pairs)
    draws = np.random.default_rng(3841024).integers(n, size=(20000, n))
    metrics = {"interaction": summarize([r["delta"] for r in pairs], draws),
               "modeled_interaction": summarize([r["modeled_delta"] for r in pairs],draws),
               "difference_dispatch_fraction": summarize([r["changed_delta"] for r in pairs], draws)}
    for cap in (384,1024):
        group = [r for r in rows if r["cap"]==cap]
        saving = summarize([r["saving"] for r in group], draws)
        count_fields = ("changed","fewer_steps","correct","full_correct","certified","capped",
                        "abstaining_paths","dispatch_prediction_correct","modeled_changed")
        metrics[str(cap)] = dict(saving=saving,
            modeled_saving=summarize([r["modeled_saving"] for r in group], draws),
            fraction_paths_reaching_cap=sum(r["capped"] for r in group)/(16*n),
            **{name:sum(r[name] for r in group) for name in count_fields},
            arms={arm:{key:sum(r[f"{arm}_{key}"] for r in group)
                       for key in ("steps","wall_seconds","generation_tokens","canceled_tokens","launched")}
                  for arm in ("exact","buffer")})
    lo, hi = metrics["interaction"]["ci"]
    long = metrics["1024"]
    metrics["interaction_verdict"] = "positive" if lo>0 else "nonpositive" if hi<=0 else "inconclusive"
    metrics["long_cap_utility_gate"] = bool(long["saving"]["mean"]>=.02 and long["saving"]["ci"][0]>0 and long["changed"]>0)
    panel_s = sum(r["wall_seconds"] for r in records)
    warmup_s = sum(r["warmup_seconds"] for r in records)
    details = dict(question_count=n, timed_panels=len(records), results=metrics,
                   same_work_and_ballots_questions=sum(p["same_work_and_ballots"] for p in pairs),
                   sessions={session:sorted({r["qid"] for r in records if r["session_id"]==session})
                             for session in sorted({r["session_id"] for r in records})},
                   questions_spanning_sessions=[p["qid"] for p in pairs
                                                 if len({r["session_id"] for r in records if r["qid"]==p["qid"]})>1],
                   seed=3841024, paired_question_draws=20000, clock_sha256=clock_sha,
                   selection_sha256=selection_hash, protocol_sha256=selection["protocol_sha256"],
                   timed_tokens=sum(r["generation_tokens"] for r in records),
                   timed_panel_seconds=panel_s, warmup_seconds=warmup_s, timed_gpu_hours=panel_s/3600,
                   timed_compute_yuan_at_6_99_per_hour=6.99*panel_s/3600,
                   observation="Actual provider billing also includes instance uptime outside the timed panels.",
                   failed_files={p.name:sha(p.read_bytes()) for p in sorted(raw.glob("failed-*.json"))},
                   raw_sha256=live_hashes)
    out.mkdir(parents=True,exist_ok=False)
    for name, data in (("cap_pairs",rows),("questions",pairs)):
        with (out/f"{name}.csv").open("x",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (out/"summary.json").write_text(json.dumps(details,indent=2)+"\n",encoding="utf8")
    print(json.dumps(metrics,indent=2))


if __name__ == "__main__":
    if not __debug__:
        raise ValueError("Run with Python assertions enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",required=True,type=Path)
    parser.add_argument("--selection",type=Path,default=HERE/"selection.json")
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    analyze(args.raw,args.selection,args.out)
