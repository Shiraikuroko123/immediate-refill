"""Reprice the archived four-slot certificates over observed clock ranges."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path
import runpy
from statistics import mean, median
import sys


HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY / "bound"))
from panel_bound import lower_demands, price


def clocks(path):
    samples = {}
    for line in path.read_bytes().splitlines():
        row = json.loads(line, parse_float=Fraction)
        if row["event"] == "trial" and row["context"] == 3968:
            key = row["block"], row["width"]
            assert key not in samples
            samples[key] = row["elapsed_s"] / row["steps"]
    assert set(samples) == set(product(range(8), range(1, 5)))
    measured = [tuple(samples[b, w] for w in range(1, 5)) for b in range(8)]
    columns = list(zip(*measured))
    middle = tuple(map(median, columns))
    low = tuple(map(min, columns))
    high = tuple(map(max, columns))
    corners = list(product(*zip(low, high)))
    return middle, low, high, measured, corners


def load_panels():
    archive = runpy.run_path(str(STUDY / "holdout/run.py"))
    original, old_hashes = archive["load_all"]()
    data = [(p.cohort, str(p.question), p.panel, p.lengths, p.votes) for p in original]
    frozen = json.loads((STUDY / "holdout/freeze.json").read_bytes())
    groups, new_hashes = archive["inputs"](frozen)
    for key, records in sorted(groups.items()):
        records = sorted(records)
        assert [r[0] for r in records] == list(range(16))
        data.append(("G", key, 0, tuple(r[1] for r in records), tuple(r[2] for r in records)))
    assert len(data) == 324
    return data, {"original": old_hashes, "GSM8K": new_hashes}


def previous_medians():
    targets = {}
    for name, group in (("bound/measured/panels.csv", None), ("holdout/ceilings/questions.csv", "G")):
        with (STUDY/name).open(encoding="utf8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["context"] == "3968":
                    key = group or row["cohort"], row["question"], int(row.get("panel", 0))
                    assert key not in targets
                    targets[key] = (float(row["greedy_s"]), float(row.get("work_lower_s") or row["lower_s"]))
    assert len(targets) == 324
    return targets


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def run(out):
    source = STUDY / "hardware/a800-live02/events.jsonl"
    middle, low, high, measured, corners = clocks(source)
    curves = [middle, *measured, *corners]
    panels, fingerprints = load_panels()
    previous = previous_medians()
    totals = defaultdict(lambda: [[Fraction(0), Fraction(0)] for _ in curves])
    count = defaultdict(int)
    for cohort, q, panel, lengths, votes in panels:
        _, occupancy, vectors, _ = lower_demands(lengths, votes, 4, 14, 1)
        count[cohort, q] += 1
        for j, curve in enumerate(curves):
            cost = sum(occupancy[w] * curve[w-1] for w in range(1, 5))
            lower = price(vectors, curve)
            assert 0 < lower <= cost
            if j == 0:
                assert (float(cost), float(lower)) == previous[cohort, q, panel]
            totals[cohort, q][j][0] += cost
            totals[cohort, q][j][1] += lower
    rows, all_curves = [], []
    for (cohort, q), times in sorted(totals.items()):
        ceilings = [1 - lower / cost for cost, lower in times]
        corners_only = ceilings[9:]
        varying = 1 - times[9][1] / times[-1][0]
        assert max(ceilings[:9]) <= max(corners_only) <= varying
        rows.append(dict(cohort=cohort, question=q, panels=count[cohort, q], median=float(ceilings[0]),
                         worst_observed_block=float(max(ceilings[1:9])), fixed_box=float(max(corners_only)),
                         maximizing_corner=corners_only.index(max(corners_only)), varying_box=float(varying)))
        for j, ((cost, lower), u) in enumerate(zip(times, ceilings)):
            all_curves.append(dict(cohort=cohort, question=q, scenario=j, greedy_s=float(cost),
                                   certificate_lower_s=float(lower), ceiling=float(u)))
    names = ("median", "worst_observed_block", "fixed_box", "varying_box")
    aggregate = {}
    for group in sorted({r["cohort"] for r in rows}):
        selected = [r for r in rows if r["cohort"] == group]
        aggregate[group] = {"questions": len(selected), "panels": sum(r["panels"] for r in selected),
                            **{k: mean(r[k] for r in selected) for k in names}}
    prior = json.loads((STUDY/"bound/measured/summary.json").read_bytes())["summary"]
    prior_means = {r["cohort"]: r["mean_ceiling"] for r in prior if r["context"] == 3968}
    prior_means["G"] = json.loads((STUDY/"holdout/ceilings/summary.json").read_bytes())["mean_ceiling"]["3968"]
    assert all(abs(v["median"] - prior_means[k]) < 1e-15 for k, v in aggregate.items())
    output = dict(post_hoc=True, panels=324, quantities="fraction of indexed greedy time",
                  summary=aggregate,
                  envelope_seconds={tag: list(map(float, value)) for tag,value in (("min",low),("median",middle),("max",high))},
                  scenarios_seconds=[list(map(float,c)) for c in curves],
                  sources_sha256=fingerprints, clock_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  checked_panel_medians=324)
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out/"questions.csv", rows)
    write_csv(out/"scenarios.csv", all_curves)
    (out/"summary.json").write_text(json.dumps(output, indent=2)+"\n", encoding="utf8")
    print(json.dumps(aggregate,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
