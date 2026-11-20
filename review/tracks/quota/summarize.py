"""Audit quota reservations and paired latency from archived engine events, without a GPU."""
import argparse
import csv
from collections import Counter
from hashlib import sha256
from itertools import groupby
import json
import math
from pathlib import Path
import random
import re
from statistics import mean


ROOT = Path(__file__).resolve().parent
N = 16
SLOTS = 4
BOOTSTRAP_SEED = 20261120
BOOTSTRAP_DRAWS = 20000
ALL_JOBS = ("fixed-exact", "fixed-buffer", "fixed-b0", "fixed-eager", "natural-exact", "natural-buffer")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    data = path.read_bytes()
    return json.loads(data), sha256(data).hexdigest()


def tally(votes):
    counts = sorted(Counter(x for x in votes if x is not None).items(), key=lambda item: (-item[1], item[0]))
    if not counts:
        return None, 0
    return counts[0][0], counts[0][1] - (counts[1][1] if len(counts) > 1 else 0)


def grade_integer(winner, gold):
    if gold is None:
        return "unscored"
    if winner is None:
        return "abstention"
    if not re.fullmatch(r"[+-]?[0-9]+", winner.strip()):
        return "unparsed"
    return "correct" if int(winner.strip()) == int(gold) else "incorrect"


def earliest(observed):
    for k in range(1, N):
        known = [v for i, v in observed.items() if i < k]
        if tally(known)[1] + k - len(known) > N - k:
            return k
    return N


def target(policy, quota, observed):
    physical_limit = min(quota, len(observed) + SLOTS)
    if policy in ("full", "eager"):
        return physical_limit
    relevant = observed
    if policy == "buffer":
        first_hole = next((i for i in range(N) if i not in observed), N)
        relevant = {i: observed[i] for i in range(first_hole)}
    return min(physical_limit, earliest(relevant) + (policy != "b0"))


def path_checks(row, ref):
    paths = row["paths"]
    require([p["index"] for p in paths] == list(range(row["launched"])), "Noncontiguous launched indices")
    require(row["completed"] == sum(p["finished"] for p in paths), "Completed count")
    require(row["canceled"] == sum(not p["finished"] for p in paths), "Canceled count")
    require(row["canceled_tokens"] == sum(p["length"] for p in paths if not p["finished"]), "Canceled work")
    require(row["generation_tokens"] == sum(p["length"] for p in paths), "Generation work")
    require(row["wall_seconds"] > 0 and math.isfinite(row["wall_seconds"]), "Bad timing")
    token_mismatches = 0
    vote_mismatches = 0
    for p in paths:
        idx = p["index"]
        original = ref["paths"][idx]
        length = p["length"]
        tokens = p["token_ids"]
        require(length == len(tokens), "Path length")
        require(p["token_sha256"] == sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(), "Token IDs")
        require(p["seed"] == original["seed"], "Seed mismatch")
        require(p["finished"] or p["vote"] is None, "Canceled path voted")
        token_mismatches += int(tokens != original["token_ids"][:length] or p["finished"] and length != original["length"])
        vote_mismatches += int(p["finished"] and p["vote"] != original["vote"])
        if row["condition"] == "fixed":
            expected = length == original["length"] if p["finished"] else length < original["length"]
            require(expected and (not p["finished"] or p["vote"] == original["vote"]), "Uncoupled fixed workload")
    return {"path_prefix_mismatches": token_mismatches, "completed_vote_mismatches": vote_mismatches}


def audit_events(row):
    """Check admissions, stopping and accounting using an independent prefix scan."""
    policy, quota = row["policy"], row["quota"]
    require(policy in ("full", "exact", "buffer", "b0", "eager"), "Unknown admission gate")
    require(quota == (16 if policy in ("full", "eager") else 14), "Unexpected quota")
    observed = {}
    active = set()
    launched = 0
    stopped = False
    exposures = 0
    last_step = -1
    for step, same_step in groupby(row["events"], key=lambda event: event["step"]):
        require(not stopped and step > last_step and (last_step != -1 or step == 0), "Nonmonotone event step")
        last_step = step
        events = list(same_step)
        position = 0
        while position < len(events) and events[position]["kind"] == "complete":
            event = events[position]
            idx = event["index"]
            require(idx in active, "Completion without active request")
            path = row["paths"][idx]
            require(path["finished"] and event["tokens"] == path["length"] and path["finish_step"] == step, "Completion differs from path")
            active.remove(idx)
            observed[idx] = path["vote"]
            position += 1
        winner, gap = tally(observed.values())
        reason = None
        if len(observed) == N:
            reason = "panel_complete"
        elif policy != "full" and gap > N - len(observed):
            reason = "strict_lock"
        elif launched == quota and not active:
            reason = "quota_exhausted"
        if reason:
            terminal = events[position:]
            stops = [e for e in terminal if e["kind"] == "stop"]
            cancels = [e for e in terminal if e["kind"] == "cancel"]
            require(len(stops) == 1 and terminal[-1] == stops[0] and stops[0]["reason"] == reason
                    and len(stops) + len(cancels) == len(terminal), "Incorrect terminal event")
            require((stops[0]["winner"], stops[0]["gap"], row["final_gap"]) == (winner, gap, gap), "Terminal tally")
            require(row["vote_counts"] == dict(Counter(v for v in observed.values() if v is not None)), "Vote counts")
            require(row["quota_reached"] == (quota == launched), "Quota-reached flag")
            require(row["stop_reason"] == reason and bool(row["certified"]) == (reason != "quota_exhausted"), "Invalid certificate")
            require(row["winner"] == winner and step == row["steps"], "Terminal winner or time")
            for event in cancels:
                idx = event["index"]
                require(idx in active and not row["paths"][idx]["finished"], "Canceled inactive or completed path")
                require(event["tokens"] == row["paths"][idx]["length"], "Canceled work differs from path")
                active.remove(idx)
            require(not active, "Missing cancellations")
            stopped = True
            continue
        limit = target(policy, quota, observed)
        extra = max(launched, target("exact", quota, observed)) - max(launched, target("buffer", quota, observed))
        exposures += int(extra > 0)
        for idx in range(launched, limit):
            require(position + 1 < len(events), "Missing reservation or launch")
            reserve, launch = events[position:position + 2]
            require((reserve["kind"], reserve["index"], reserve["before"], reserve["after"], reserve["quota"])
                    == ("reserve", idx, idx, idx + 1, quota), "Invalid reservation")
            require((launch["kind"], launch["index"]) == ("launch", idx), "No launch after debit")
            require(idx < quota and len(active) < SLOTS and row["paths"][idx]["launch_step"] == step,
                    "Admission exceeds quota, occupies a full slot, or disagrees with path timing")
            active.add(idx)
            launched += 1
            position += 2
        require(position == len(events) and active, "Unexpected event or stalled engine")
    require(stopped and row["launched"] == row["quota_used"] == launched and len(observed) == row["completed"], "Incomplete event log")
    return {"admission_histories_with_index_advantage": exposures}


def percentile(values, probability):
    fractional = (len(values) - 1) * probability
    index = int(fractional)
    return values[index] + (fractional - index) * (values[math.ceil(fractional)] - values[index])


def summarize_pairs(rows):
    by = {(r["qid"], r["condition"], r["policy"]): r for r in rows}
    qids = sorted({r["qid"] for r in rows})
    rng = random.Random(BOOTSTRAP_SEED)
    samples = [rng.choices(range(len(qids)), k=len(qids)) for _ in range(BOOTSTRAP_DRAWS)]
    comparisons, per_question = [], []
    for condition, base in (("fixed", "buffer"), ("fixed", "b0"), ("fixed", "eager"), ("natural", "buffer")):
        if not all((q, condition, gate) in by for q in qids for gate in ("exact", base)):
            continue
        paired = []
        for q in qids:
            a = by[q, condition, "exact"]
            b = by[q, condition, base]
            require(a["config_hash"] == b["config_hash"], "Unmatched devices or configurations")
            question = dict(qid=q, unique_id=a["unique_id"], condition=condition, chosen="exact", base=base,
                            winners_agree=a["winner"] == b["winner"], both_certified=a["certified"] and b["certified"],
                            exact_absolute_excess_over_14=a["absolute_excess_over_14"],
                            base_absolute_excess_over_14=b["absolute_excess_over_14"])
            for metric in ("wall_seconds", "generation_tokens"):
                question["exact_" + metric] = a[metric]
                question["base_" + metric] = b[metric]
                question[metric + "_saved"] = 1 - a[metric] / b[metric]
            paired.append(question)
        per_question.extend(paired)
        comparison = dict(condition=condition, chosen="exact", base=base, questions=len(qids),
                          chosen_quota=a["quota"], base_quota=b["quota"],
                          winner_agreements=sum(q["winners_agree"] for q in paired))
        for metric in ("wall_seconds", "generation_tokens"):
            values = [q[metric + "_saved"] for q in paired]
            distribution = sorted(sum(values[i] for i in sample) / len(sample) for sample in samples)
            comparison[metric] = dict(macro_saved=mean(values), low=percentile(distribution, .025),
                                      high=percentile(distribution, .975), strictly_better=sum(v > 0 for v in values),
                                      tied=sum(v == 0 for v in values),
                                      pooled_saved=1 - sum(q["exact_" + metric] for q in paired)
                                                    / sum(q["base_" + metric] for q in paired))
        comparisons.append(comparison)
    return comparisons, per_question


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def check_warmup(row, prompts):
    tokens = row["warmup_prompt_ids"]
    require(isinstance(tokens, list) and tokens and all(type(t) is int for t in tokens), "Invalid warmup prompt")
    require(len(row["warmup_ids"]) == 1, "Warmup must emit one token")
    known = prompts.setdefault(row["config_hash"], tokens)
    require(tokens == known, "Warmup prompt tokens differ within the configuration")


def summarize(raw, out, qids, jobs, questions, selection_digest=None):
    require(qids and len(set(qids)) == len(qids) and all(i >= 0 for i in qids), "Need distinct, nonnegative question indices")
    require(jobs and set(jobs) <= set(ALL_JOBS) and len(set(jobs)) == len(jobs), "Invalid jobs")
    panels = []
    manifests = {}
    warmup_prompts = {}
    for q in qids:
        ref, ref_hash = load(raw / f"q{q:02d}-natural-full.json")
        require((ref["qid"], ref["condition"], ref["policy"], ref["completed"], ref["canceled"]) ==
                (q, "natural", "full", N, 0), "Incomplete reference")
        path_checks(ref, ref)
        check_warmup(ref, warmup_prompts)
        audit_events(ref)
        require(ref["reference_sha256"] is None and ref["position"] == 0, "Reference identity")
        ref_votes = {p["index"]: p["vote"] for p in ref["paths"]}
        ref_winner = tally(ref_votes.values())[0]
        k = earliest(ref_votes)
        require(ref["winner"] == ref_winner and ref["reference_serial_prefix"] == k, "Full reference mismatch")
        if questions:
            require(ref["unique_id"] == questions[q]["unique_id"], "Reference does not match selection")
        gold = questions[q].get("answer") if questions else None
        for job in jobs:
            condition, gate = job.split("-")
            filename = f"q{q:02d}-{job}.json"
            try:
                r, file_hash = load(raw / filename)
                require((r["qid"], r["unique_id"], r["condition"], r["policy"], r["reference_sha256"]) ==
                        (q, ref["unique_id"], condition, gate, ref_hash), "Unpaired record")
                cfg = r["config_hash"]
                if cfg not in manifests:
                    manifest, _ = load(raw / f"config-{cfg}.json")
                    require(sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() == cfg, "Config hash")
                    require(sha256(json.dumps(manifest["protocol"], sort_keys=True).encode()).hexdigest() ==
                            manifest["protocol_hash"], "Protocol hash")
                    require(manifest["protocol"]["engine_warmup"] == "Ready.", "Different warmup prompt text")
                    if selection_digest:
                        require(manifest["protocol"]["selection_sha256"] == selection_digest, "Selection hash")
                    manifests[cfg] = manifest
                require(ref["protocol_hash"] == r["protocol_hash"] == manifests[cfg]["protocol_hash"], "Different panel protocols")
                check_warmup(r, warmup_prompts)
                counts = path_checks(r, ref)
                history = audit_events(r)
                require(r["reference_winner"] == ref_winner and r["reference_serial_prefix"] == k, "Incorrect reference stats")
                exhausted = r["stop_reason"] == "quota_exhausted"
                mismatch = int(r["winner"] != ref_winner)
                grade = grade_integer(r["winner"], gold)
                if condition == "fixed":
                    require(not r["certified"] or not mismatch, "False certified winner")
                    if gate != "eager":
                        require(r["launched"] <= k + (gate != "b0"), "Overrun bound")
                require(r["launch_delta_vs_reference"] == r["launched"] - k, "Incorrect launch excess")
                fields = ("qid", "unique_id", "condition", "policy", "quota", "quota_used", "certified",
                          "stop_reason", "winner", "reference_winner", "wall_seconds", "generation_tokens",
                          "canceled_tokens", "completed", "canceled", "launched", "steps", "config_hash", "position",
                          "quota_reached")
                panels.append({**{key: r[key] for key in fields}, "source_sha256": file_hash,
                               "reference_sha256": ref_hash, "reference_serial_prefix": k,
                               "positive_reference_excess": max(0, r["launched"] - k),
                               "absolute_excess_over_14": max(0, r["launched"] - 14),
                               "exhausted": int(exhausted), "winner_mismatch": mismatch,
                               "exhausted_mismatch": int(exhausted and mismatch),
                               "exhausted_abstention": int(exhausted and r["winner"] is None),
                               "exhausted_nonabstaining_mismatch": int(exhausted and mismatch and r["winner"] is not None),
                               "gold": gold, "gold_status": grade, "reference_gold_status": grade_integer(ref_winner, gold),
                               "exhausted_gold_incorrect_integer": int(exhausted and grade == "incorrect"),
                               "exhausted_gold_unparsed": int(exhausted and grade == "unparsed"),
                               **counts, **history})
            except (KeyError, ValueError, IndexError) as exc:
                raise ValueError(f"{filename}: {exc}") from exc
    totals = []
    for job in jobs:
        condition, gate = job.split("-")
        group = [r for r in panels if r["condition"] == condition and r["policy"] == gate]
        fields = ("wall_seconds", "generation_tokens", "canceled_tokens", "launched", "canceled", "completed",
                  "certified", "exhausted", "winner_mismatch", "exhausted_mismatch", "exhausted_nonabstaining_mismatch",
                  "exhausted_abstention", "path_prefix_mismatches", "completed_vote_mismatches",
                  "admission_histories_with_index_advantage", "quota_reached",
                  "exhausted_gold_incorrect_integer", "exhausted_gold_unparsed")
        totals.append(dict(condition=condition, policy=gate, questions=len(group),
                           gold_status=dict(Counter(r["gold_status"] for r in group)),
                           reference_gold_status=dict(Counter(r["reference_gold_status"] for r in group)),
                           max_positive_reference_excess=max(r["positive_reference_excess"] for r in group),
                           questions_above_14=sum(r["absolute_excess_over_14"] > 0 for r in group),
                           sum_absolute_excess_over_14=sum(r["absolute_excess_over_14"] for r in group),
                           max_absolute_excess_over_14=max(r["absolute_excess_over_14"] for r in group),
                           **{field: sum(r[field] for r in group) for field in fields}))
    contrasts, pairs = summarize_pairs(panels)
    summary = dict(qids=qids, jobs=jobs, bootstrap_resamples=BOOTSTRAP_DRAWS, bootstrap_seed=BOOTSTRAP_SEED,
                   primary_contrast="fixed, exact versus buffer, question mean of 1 - exact / buffer",
                   correctness_comparator="full-panel plurality on the archived sample, with lexical tie-break",
                   gold_scoring="signed decimal integer comparison after stripping whitespace; noninteger strings unparsed",
                   configs={k: v["gpu"] for k, v in manifests.items()}, warmup_prompt_ids_by_config=warmup_prompts,
                   contrasts=contrasts, totals=totals)
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "panels.csv", panels)
    write_csv(out / "questions.csv", pairs)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--selection", type=Path, default=ROOT / "selection.json", help="Frozen question manifest")
    parser.add_argument("--qids", help="Explicit subset for pilot or partial-run inspection")
    parser.add_argument("--jobs", default=",".join(ALL_JOBS), help="Comma-separated condition-gate jobs")
    args = parser.parse_args()
    manifest, selection_digest = load(args.selection)
    selection = manifest["questions"]
    if args.qids:
        ids = list(map(int, args.qids.split(",")))
    elif selection:
        ids = list(range(len(selection)))
    else:
        parser.error("Provide --selection for a complete study or --qids for an explicit subset")
    print(json.dumps(summarize(args.raw, args.out, ids, args.jobs.split(","), selection, selection_digest), indent=2))
