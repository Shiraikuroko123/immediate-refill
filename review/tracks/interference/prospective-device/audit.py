"""Independently audit the event trace and decompose end-to-end timing by phase."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import median


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_row(row):
    lengths = [256, 128, 128] if row["panel"] == "witness" else [128, 128]
    slots = len(lengths) - int(row["arm"] == "staged")
    assert row["token_counts"] == lengths
    events = defaultdict(list)
    for step, kind, lane in row["transitions"]:
        events[step].append((kind, lane))
    active = set()
    counts = Counter()
    phase = Counter()
    width = Counter()
    next_lane = 0
    for step, (ns, advanced, first) in enumerate(row["steps"]):
        assert ns > 0
        transitions = events.pop(step, [])
        adds = [lane for kind, lane in transitions if kind == "admit"]
        finishes = [lane for kind, lane in transitions if kind == "finish"]
        assert transitions == [("admit", i) for i in adds] + [("finish", i) for i in finishes]
        while next_lane < len(lengths) and len(active) < slots:
            assert adds.pop(0) == next_lane
            active.add(next_lane)
            next_lane += 1
        assert not adds
        assert len(set(advanced)) == len(advanced) and set(advanced) <= active
        assert sorted(first) == sorted(i for i in advanced if not counts[i])
        counts.update(advanced)
        assert set(finishes) == {i for i in advanced if counts[i] == lengths[i]}
        active.difference_update(finishes)
        kind = "first" if first else ("cached" if advanced else "empty")
        phase[kind] += ns
        width[f"{kind}_width{len(advanced)}"] += 1
    assert not events and not active and next_lane == len(lengths)
    assert [counts[i] for i in range(len(lengths))] == lengths
    assert len(row["token_hashes"]) == len(lengths)
    assert all(len(bytes.fromhex(h)) == 32 for h in row["token_hashes"])
    phase["host_outside_engine_step"] = row["wall_ns"] - sum(phase.values())
    assert phase["host_outside_engine_step"] >= 0
    return phase, width


def audit(directory):
    cal_path = directory/"calibrate.jsonl"
    val_path = directory/"validate.jsonl"
    pred_path = directory/"prediction.json"
    pred = json.loads(pred_path.read_bytes())
    cal = load(cal_path)
    val = load(val_path)
    assert pred["calibration_sha256"] == digest(cal_path)
    assert val[0]["event"] == "sealed_prediction" and val[0]["sha256"] == digest(pred_path)
    assert not any(r["event"] == "failed" for r in cal+val)
    assert len({next(r for r in events if r["event"]=="metadata")["pid"] for events in (cal,val)}) == 2
    clocks = defaultdict(dict)
    for row in cal:
        if row["event"] in ("warmup","trial"):
            assert row["steps"] == 96 and row["context"] == 512
            block = clocks[row["block"]]
            assert row["width"] not in block
            block[row["width"]] = row["ms_per_step"]
    assert set(clocks) == set(range(-1,8)) and all(set(block) == {1,2,3} for block in clocks.values())
    assert [median(clocks[i][w] for i in range(8)) for w in [1,2,3]] == pred["clock_ms"]
    actual = {}
    schedules = {}
    arms = {("witness","immediate"),("witness","staged"),("control","immediate"),("control","staged")}
    for row in val:
        if row["event"] not in ("warmup","trial"):
            continue
        idx = (row["block"],row["panel"],row["arm"])
        assert idx not in actual
        phase, width = inspect_row(row)
        actual[idx] = (row,phase,width)
        schedules[row["block"],row["pos"]] = (row["panel"],row["arm"])
    assert set(actual) == {(b,p,a) for b in range(-1,12) for p,a in arms}
    assert all(Counter(schedules[b,i] for b in range(12)) == Counter({arm:3 for arm in arms}) for i in range(4))
    output = {"raw_sha256":{p.name:digest(p) for p in (cal_path,val_path,pred_path)},"panels":{}}
    metadata = next(r for r in cal if r["event"] == "metadata")
    observed = [datetime.fromisoformat(r["utc"]) for r in cal+val]
    output["budget"] = dict(
        device=metadata["gpu"], vllm=metadata["vllm"],
        calibration_windows=sum(len(c) for c in clocks.values()),
        validation_panels_including_warmup=len(actual),
        validation_generated_tokens_including_warmup=sum(sum(r[0]["token_counts"]) for r in actual.values()),
        validation_generated_tokens_measured=sum(sum(r[0]["token_counts"]) for k,r in actual.items() if k[0]>=0),
        validation_summed_panel_wall_s=sum(r[0]["wall_ns"] for r in actual.values())/1e9,
        first_to_last_log_utc=[min(observed).isoformat(),max(observed).isoformat()],
        first_to_last_log_s=(max(observed)-min(observed)).total_seconds())
    for panel in ("witness","control"):
        effects = []
        contrasts = defaultdict(list)
        widths = defaultdict(Counter)
        mismatched_token_lanes = 0
        for b in range(12):
            first, second = [actual[b,panel,a] for a in ("immediate","staged")]
            w0,w1 = first[0]["wall_ns"],second[0]["wall_ns"]
            effects.append((w0-w1)/w0*100)
            mismatched_token_lanes += sum(a!=b for a,b in zip(first[0]["token_hashes"], second[0]["token_hashes"]))
            difference = {
                k:(first[1][k]-second[1][k])/w0*100
                for k in set(first[1]) | set(second[1])
            }
            assert abs(sum(difference.values())-effects[-1]) < 1e-10
            for k in ("first","cached","empty","host_outside_engine_step"):
                contrasts[k].append(difference.get(k,0))
            for arm, data in (("immediate",first),("staged",second)):
                widths[arm].update(data[2])
        ordered = sorted(effects)
        output["panels"][panel] = dict(wall_median_pct=median(effects),
            median_interval_pct=[ordered[2],ordered[9]],
            mean_wall_pct=sum(effects)/12,
            arm_median_wall_ms={arm:median(actual[b,panel,arm][0]["wall_ns"]/1e6 for b in range(12))
                                for arm in ("immediate","staged")},
            paired_token_hash_disagreements=mismatched_token_lanes,
            additive_mean_contributions_pp={k:sum(v)/12 for k,v in contrasts.items()},
            timed_block_step_counts={arm:dict(hist) for arm,hist in widths.items()})
        calibrated = output["panels"][panel]
        predicted = pred["panels"][panel]
        lo, hi = calibrated["median_interval_pct"]
        # Re-evaluate the prespecified rule without importing the primary analyzer.
        if abs(predicted["effect_pct"]) < 1:
            inside = lo >= -2 and hi <= 2
        elif predicted["effect_pct"] > 0:
            inside = lo > 0
        else:
            inside = hi < 0
        calibrated["passed"] = inside and abs(median(effects)-predicted["effect_pct"]) <= 2
    output["joint_transfer_passed"] = all(p["passed"] for p in output["panels"].values())
    if (directory/"summary.json").is_file():
        primary = json.loads((directory/"summary.json").read_bytes())
        assert primary["inputs"] == output["raw_sha256"]
        assert primary["joint_transfer_passed"] == output["joint_transfer_passed"]
        for name, panel in output["panels"].items():
            p = primary["panels"][name]
            assert abs(p["median_pct"]-panel["wall_median_pct"]) < 1e-10
            assert all(abs(a-b) < 1e-10 for a,b in zip(p["interval_pct"],panel["median_interval_pct"]))
            assert p["passed"] == panel["passed"]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.directory),indent=2))
