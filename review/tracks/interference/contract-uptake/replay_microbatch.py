"""Retrospective migration of completed microbatch charging to 16-vote GSM8K panels."""
import csv
import hashlib
import json
import runpy
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY / "information-baseline"))
from gates import gap, simulate


def first_lock(votes):
    return next((i for i in range(1, len(votes))
                 if gap(votes[:i]) > len(votes) - i), len(votes))


def winner(votes):
    counts = Counter(v for v in votes if v is not None)
    return counts.most_common(1)[0][0] if counts else None


def completed_batches(work, votes, width):
    n = len(votes)
    occ = [0] * 4
    for first in range(0, n, width):
        cohort = sorted(work[first:first+width])
        previous = 0
        for i, length in enumerate(cohort):
            occ[len(cohort)-i-1] += length-previous
            previous = length
        end = first + len(cohort)
        if gap(votes[:end]) > n-end or end == n:
            return end, occ
    raise AssertionError("empty panel")


def main(out):
    runner = runpy.run_path(str(STUDY / "holdout/run.py"))
    frozen = json.loads((STUDY / "holdout/freeze.json").read_bytes())
    groups, hashes = runner["inputs"](frozen)
    clock = STUDY / "hardware/a800-live02/events.jsonl"
    prices = list(map(float, runner["read_curves"](clock, 4)[3968]["median"]))
    rows = []
    for question, paths in sorted(groups.items()):
        ordered = sorted(paths)
        assert [i for i, _, _ in ordered] == list(range(16))
        work, votes = zip(*[(w, v) for _, w, v in ordered])
        lock = first_lock(votes)
        for width in (1, 2, 4):
            starts, occ = completed_batches(work, votes, width)
            assert starts == min(16, width * ((lock+width-1)//width))
            assert winner(votes[:starts]) == winner(votes)
            rows.append((question, f"batch{width}", lock, starts, starts-lock,
                         sum(c*d for c, d in zip(prices, occ))))
        for name, b, information in (("indexed0", 0, "indexed"),
                                     ("indexed1", 1, "indexed"),
                                     ("count_and_prefix1", 1, "prefix_bag"),
                                     ("indexed3", 3, "indexed"),
                                     ("absolute_only", 16, "indexed")):
            result = simulate(work, votes, b, 16, 4, information)
            starts = len(result["starts"])
            assert 0 <= starts-lock <= b
            decision_tick = sum(result["occ"])
            returned = [votes[i] for i, t in enumerate(result["starts"])
                        if t+work[i] <= decision_tick]
            assert len(returned) == 16 or gap(returned) > 16-len(returned)
            assert winner(returned) == winner(votes)
            rows.append((question, name, lock, starts, starts-lock,
                         sum(c*d for c, d in zip(prices, result["occ"]))))
    assert len(rows) == 200 * 8
    summary = {}
    for name in dict.fromkeys(r[1] for r in rows):
        arm = [r for r in rows if r[1] == name]
        summary[name] = dict(count=len(arm), mean_starts=mean(r[3] for r in arm),
                             mean_excess=mean(r[4] for r in arm),
                             excess_hist=dict(sorted(Counter(r[4] for r in arm).items())),
                             excess_gt_one=sum(r[4] > 1 for r in arm))
    pairs = []
    for width, allowance in [(1, 0), (2, 1), (4, 3)]:
        wave = {r[0]:r for r in rows if r[1] == f"batch{width}"}
        indexed = {r[0]:r for r in rows if r[1] == f"indexed{allowance}"}
        pairs.append(dict(microbatch=width, allowance=allowance,
                          mean_paired_time_saving=mean(1-indexed[k][5]/r[5] for k, r in wave.items()),
                          mean_paired_starts_saved=mean(r[3]-indexed[k][3] for k,r in wave.items())))
    indexed1 = {r[0]:r for r in rows if r[1] == "indexed1"}
    strong = {}
    for name in ("count_and_prefix1", "absolute_only"):
        base = {r[0]:r for r in rows if r[1] == name}
        strong[name] = dict(
            mean_paired_time_saving=mean(1-indexed1[k][5]/r[5] for k,r in base.items()),
            mean_paired_starts_saved=mean(r[3]-indexed1[k][3] for k,r in base.items()))
    report = dict(retrospective=True, slots=4, panel=16, absolute_quota=16, clock_context=3968,
                  unit="question", time="replay under previous DeepSeek A800 clock, not wall time",
                  clock_sha256=hashlib.sha256(clock.read_bytes()).hexdigest(),
                  source_sha256=hashes, arms=summary, same_allowance_pairs=pairs,
                  indexed1_vs_other_rolling_baselines=strong,
                  all_decisions_equal_full_panel_with_first_occurrence_ties=True)
    encoded_summary = json.dumps(report, indent=2)+"\n"
    out.mkdir(parents=True, exist_ok=False)
    with (out / "panels.csv").open("x", encoding="utf8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["question", "policy", "first_strict_serial_lock", "starts",
                         "excess_starts", "modeled_seconds"])
        writer.writerows(rows)
    (out / "summary.json").write_text(encoded_summary, encoding="utf8")
    print(json.dumps(dict(arms=summary, same_allowance_pairs=pairs, strong=strong), indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
