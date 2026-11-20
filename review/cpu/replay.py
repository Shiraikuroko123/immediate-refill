"""Rebuild the manuscript's current and inherited numerical comparisons on a CPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent


def run(root, tmp):
    study = root / "tracks/interference"
    reads = set()
    # Every entry runs in a fresh interpreter to keep the archived modules independent.
    def task(name, script, arguments, expected_files, stdout_reference=None):
        trace = tmp / (name + ".inputs.json")
        path = study / script
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                   OMP_NUM_THREADS="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
        proc = subprocess.run([sys.executable, "-B", str(HERE / "worker.py"), str(root),
                               str(trace), str(path), *map(str, arguments)], cwd=root, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode:
            raise RuntimeError(f"{name} failed:\n" + (proc.stderr + proc.stdout)[-3000:].decode("utf8", "replace"))
        files = set(json.loads(trace.read_bytes()))
        files.add(path.relative_to(root).as_posix())
        for new, saved in expected_files:
            reference = study / saved
            files.add(reference.relative_to(root).as_posix())
            if new.read_bytes().replace(b"\r\n", b"\n") != reference.read_bytes().replace(b"\r\n", b"\n"):
                raise ValueError(f"{name} gave a different {new.name}")
        if stdout_reference:
            reference = study / stdout_reference
            files.add(reference.relative_to(root).as_posix())
            if json.loads(proc.stdout) != json.loads(reference.read_bytes()):
                raise ValueError(f"{name} gave a different result")
        return files

    def outputs(out, ref, names):
        return [(out / n, ref + "/" + n) for n in names]

    info = tmp / "information"
    opp = tmp / "opportunity"
    contract = tmp / "contract"
    caps = tmp / "caps"
    bounds = tmp / "bounds"
    sess = study / "prospective-device/session01"
    cases = [
        ("sealed-panel-estimates", "prospective-device/experiment.py", ["analyze", "--out", sess],
         [], "prospective-device/session01/summary.json"),
        ("independent-device-audit", "prospective-device/audit.py", [sess], [],
         "prospective-device/session01/audit.json"),
        ("natural-opportunity", "opportunity/replay.py", ["--out", opp],
         outputs(opp, "opportunity/results", ("summary.json", "questions.csv"))),
        ("full-panel-contract", "contract-uptake/replay_microbatch.py", [contract],
         outputs(contract, "contract-uptake/complete", ("summary.json", "panels.csv"))),
        ("theory-examples", "theorem-audit/check.py", [], [], "theorem-audit/examples.json"),
        ("verified-candidates", "verified-candidate/check.py", ["--out", tmp / "candidates.json"],
         [(tmp / "candidates.json", "verified-candidate/results.json")]),
        ("strong-baseline", "information-baseline/replay.py", [info],
         outputs(info, "information-baseline/results", ("summary.json", "questions.csv"))),
        ("cap-wall-times", "cap-diagnostic/analyze_live.py",
         ["--raw", study / "cap-diagnostic/raw", "--out", caps],
         outputs(caps, "cap-diagnostic/device-results", ("summary.json", "questions.csv", "cap_pairs.csv"))),
        ("clock-cover", "clock-uncertainty/curvature.py", ["--out", tmp / "cover.json"],
         [(tmp / "cover.json", "clock-uncertainty/results/concave-cover.json")]),
        ("sharpness", "clock-uncertainty/sharpness.py", ["--out", tmp / "sharpness.json"],
         [(tmp / "sharpness.json", "clock-uncertainty/results/sharpness.json")]),
        ("occupancy", "width-generalization/check.py", ["--out", tmp / "width.json"],
         [(tmp / "width.json", "width-generalization/results.json")]),
        ("one-extra-bounds", "clock-uncertainty/run.py", ["--out", bounds],
         outputs(bounds, "clock-uncertainty/results", ("summary.json", "questions.csv", "scenarios.csv"))),
        ("scope-and-boundary-examples", "derivation-corrections/scope_check.py", [],
         [], "derivation-corrections/scope-results.json"),
        ("baseline-attribution-and-intervals", "derivation-corrections/baseline_check.py", [],
         [], "derivation-corrections/baseline_results.json"),
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for case, found in zip(cases, pool.map(lambda c: task(*c), cases)):
            reads.update(found)
            print("Rebuilt " + case[0], file=sys.stderr, flush=True)

    def data(path):
        return json.loads(path.read_bytes())
    comparison = data(info / "summary.json")["summary"]
    cap = data(caps / "summary.json")
    clock = data(tmp / "cover.json")["zero_allowance_universal_guarantees"]
    timing = data(sess / "summary.json")
    result = {
        "checks": [c[0] for c in cases],
        "cached_network_policy": "Python DNS lookup and socket connect blocked in every analysis process",
        "current_claims": {
            "prospective_validation": {k: {"median_pct": v["median_pct"],
                                           "predicted_pct": v["prediction"]["effect_pct"]}
                                       for k, v in timing["panels"].items()},
            "oracle_mean_saving_pct": data(opp / "summary.json")["oracle_mean_gain_pct"],
            "complete_panel_contract": data(contract / "summary.json")["indexed1_vs_other_rolling_baselines"],
            "verified_candidate_configurations": data(tmp / "candidates.json")["exhaustive"]["panel_configurations"]
        },
        "inherited_claims_rebuilt": {
            "archive_mean_saving_pct_by_comparator": {r["comparator"]:100 * r["mean_saving"]
                                                       for r in comparison if r["cohort"] == "archive"},
            "cap_study_questions": cap["question_count"], "cap_study_panels": cap["timed_panels"],
            "cap_mean_savings_pct": {k:100 * cap["results"][k]["saving"]["mean"] for k in ("384", "1024")},
            "cap_interaction_pp": 100 * cap["results"]["interaction"]["mean"],
            "max_greedy_time_savable_pct_by_context": {
                str(r["context"]):100 * r["saving_over_greedy_at_most"] for r in clock},
            "matching_hull_constructions": data(tmp / "sharpness.json")["chords"],
            "occupancy_configurations": data(tmp / "width.json")["totals"]["configurations"],
            "uncertainty_panels": data(bounds / "summary.json")["panels"],
        },
    }
    return result, sorted(n for n in reads if "__pycache__" not in Path(n).parts and not n.endswith(".pyc"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="extracted directory containing tracks/")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--list-inputs", type=Path, help="write observed relative input paths for packaging")
    args = parser.parse_args()
    if not __debug__:
        raise SystemExit("Run with normal Python assertions enabled.")
    with tempfile.TemporaryDirectory() as directory:
        report, required = run(args.root.resolve(strict=True), Path(directory))
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report:
        args.report.write_text(text, encoding="utf8")
    if args.list_inputs:
        args.list_inputs.write_text(json.dumps(required, indent=2) + "\n", encoding="utf8")
    print(text, end="")
