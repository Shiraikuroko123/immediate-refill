"""Recompute the three GSM8K contrasts and their common-denominator decomposition."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
ORIGINAL = STUDY / "holdout/results/questions.csv"
POSTHOC = STUDY / "information-baseline/results/questions.csv"


def intervals(gains, seed, resamples):
    means = np.random.default_rng(seed).choice(gains, (resamples, len(gains))).mean(axis=1)
    return (100 * np.quantile(means, [0.025, 0.975])).tolist()


def analyze():
    with ORIGINAL.open(encoding="utf8", newline="") as f:
        primary = [r for r in csv.DictReader(f)
                   if r["arm"] == "indexed_vs_buffer" and r["context"] == "3968"]
    with POSTHOC.open(encoding="utf8", newline="") as f:
        additional = [r for r in csv.DictReader(f) if r["cohort"] == "archive"]
    ids = [r["question"] for r in primary]
    assert len(ids) == len(set(ids)) == 200
    indexed = np.array([float(r["time"]) for r in primary])
    prefix = np.array([float(r["base_time"]) for r in primary])
    refs = {}
    for name in ("buffer", "bag", "prefix_bag"):
        arm = [r for r in additional if r["comparator"] == name]
        assert [r["question"] for r in arm] == ids
        ref = np.array([float(r["comparator_time"]) for r in arm])
        np.testing.assert_allclose([float(r["indexed_time"]) for r in arm], indexed, rtol=1e-13, atol=0)
        refs[name] = ref
    np.testing.assert_allclose(refs["buffer"], prefix, rtol=1e-13, atol=0)
    refs["buffer"] = prefix
    contrasted = {}
    for name, reference in refs.items():
        gains = 1 - indexed / reference
        seed, resamples = (19431, 10000) if name == "buffer" else (651, 20000)
        contrasted[name] = {"question_mean_saving_pct": 100 * float(gains.mean()),
                            "reported_seed": seed, "reported_resamples": resamples,
                            "reported_95pct_percentile_interval_pct": intervals(gains, seed, resamples)}
    normalized_early = (prefix - refs["prefix_bag"]) / prefix
    normalized_late = (refs["prefix_bag"] - indexed) / prefix
    np.testing.assert_allclose(normalized_early + normalized_late, 1 - indexed / prefix,
                               rtol=0, atol=1e-15)
    return {
        "context_tokens": 3968, "questions": len(ids),
        "input_sha256": {p.relative_to(STUDY).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (ORIGINAL, POSTHOC)},
        "contrasts": contrasted,
        "unordered_ci_pct_if_using_primary_protocol": intervals(1 - indexed / refs["bag"], 19431, 10000),
        "mean_contributions_relative_to_question_prefix_time_pp": {
            "prefix_to_prefix_plus_multiset": 100 * float(normalized_early.mean()),
            "prefix_plus_multiset_to_indexed": 100 * float(normalized_late.mean()),
            "total": 100 * float((1 - indexed / prefix).mean()),
        },
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
