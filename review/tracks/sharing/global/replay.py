"""Replay the unit-rate indexed policy and price its occupancy at measured widths."""
import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/"tracks/quota/transfer_diag"))
from run import load, simulate, tally
sys.path.insert(0, str(ROOT/"tracks/lookahead/theory"))
from frontier import minimum_prefix


def occupancy(lengths, votes, slots, quota, excess):
    n = len(votes)
    if len(lengths) != n or min(lengths) <= 0 or not (0 < quota <= n and 0 < slots <= n and excess >= 0):
        raise ValueError("Invalid panel or constraints")
    observed, finishes = {}, {}
    durations = Counter()
    now = launched = tokens = 0
    while True:
        winner, gap = tally(observed.values())
        certified = gap > n-len(observed)
        if certified or (launched == quota and not finishes):
            return dict(ticks=now, widths=durations, tokens=tokens, winner=winner,
                        certified=certified, launched=launched)
        bound = min(quota, minimum_prefix(n, observed) + excess, len(observed) + slots)
        for idx in range(launched, bound):
            finishes[idx] = now + lengths[idx]
        launched = bound
        next_time = min(finishes.values())
        duration = next_time - now
        durations[len(finishes)] += duration
        tokens += len(finishes)*duration
        now = next_time
        for idx in [i for i, end in finishes.items() if end == now]:
            del finishes[idx]
            observed[idx] = votes[idx]


def read_curves(path, slots):
    blocks = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line, parse_float=Fraction)
        if row["event"] == "trial" and 1 <= row["width"] <= slots:
            group = blocks[row["context"], row["width"]]
            assert row["block"] not in group and row["steps"] == 96
            group[row["block"]] = row["elapsed_s"]/row["steps"]
    curves = {}
    for context in sorted({context for context, width in blocks}):
        measures = [blocks[context, width] for width in range(1, slots+1)]
        if any(sorted(m) != list(range(8)) for m in measures):
            raise ValueError("Need eight complete blocks at every width in the requested context")
        curves[context] = {"median": [statistics.median(m.values()) for m in measures]}
        curves[context].update({f"block{block}": [m[block] for m in measures] for block in range(8)})
    return curves


def main(out):
    slots, quota, excess = 2, 14, 1
    clock = ROOT/"tracks/sharing/hardware/a800-live01/events.jsonl"
    curves = read_curves(clock, slots)
    panels, hashes = load()
    inputs = {**hashes, "clock": hashlib.sha256(clock.read_bytes()).hexdigest(),
              "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for dependency in ("tracks/quota/transfer_diag/run.py", "tracks/quota/summarize.py", "tracks/lookahead/theory/frontier.py"):
        inputs[dependency] = hashlib.sha256((ROOT/dependency).read_bytes()).hexdigest()
    rows, by_question, matched = [], defaultdict(lambda: [Fraction(0), Fraction(0)]), 0
    for panel in panels:
        four = occupancy(panel.lengths, panel.votes, 4, quota, excess)
        reference = simulate(panel.lengths, panel.votes, "exact")
        assert (four["ticks"], four["tokens"], four["winner"], four["certified"]) == (
            reference["steps"], reference["tokens"], reference["winner"], reference["certified"])
        matched += 1
        sim = occupancy(panel.lengths, panel.votes, slots, quota, excess)
        d1, d2 = sim["widths"][1], sim["widths"][2]
        assert d1+d2 == sim["ticks"] and d1+2*d2 == sim["tokens"]
        for context, scenarios in curves.items():
            for case, curve in scenarios.items():
                tg = sum(sim["widths"][i]*curve[i-1] for i in range(1, slots+1))
                floor = sim["ticks"]*min(curve)
                sums = by_question[panel.cohort, context, case, panel.question]
                sums[0] += tg
                sums[1] += floor
                if case == "median":
                    rows.append(dict(cohort=panel.cohort, question=panel.question, panel=panel.panel, context=context,
                                     virtual_width1=d1, virtual_width2=d2, generated_tokens=sim["tokens"],
                                     launched=sim["launched"], greedy_s=float(tg), floor_s=float(floor),
                                     max_saving=1-float(floor/tg)))
    groups = defaultdict(list)
    for (cohort, context, case, question), (tg, floor) in by_question.items():
        groups[cohort, context, case].append(1-floor/tg)
    summary = []
    for cohort in ("A","O"):
        for context, scenarios in curves.items():
            group = groups[cohort,context,"median"]
            block_means = [statistics.mean(groups[cohort, context, f"block{b}"]) for b in range(8)]
            c1,c2 = scenarios["median"]
            summary.append(dict(cohort=cohort, context=context, questions=len(group), c1_ms=float(1000*c1),
                                c2_ms=float(1000*c2), question_mean_max_saving=float(statistics.mean(group)),
                                maximum_question_bound=float(max(group)),
                                block_mean_saving_range=[float(min(block_means)), float(max(block_means))]))
    out.mkdir(parents=True,exist_ok=False)
    with (out/"panels.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    result = dict(inputs=inputs, reference_matches_at_four_slots=matched, slots=slots, quota=quota, excess=excess,
                  summary=summary)
    (out/"summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    main(parser.parse_args().out)
