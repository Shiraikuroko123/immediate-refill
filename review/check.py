"""Verify the extracted artifact and recompute the principal device results on CPU."""
from hashlib import sha256
import json
from pathlib import Path
from subprocess import check_output
import sys
import tempfile


def read(path):
    return json.loads(path.read_bytes())


def main():
    root = Path(__file__).resolve().parent
    manifest = read(root / "MANIFEST.sha256.json")
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Fingerprint mismatch: {name}")

    study = root / "tracks/interference"
    device = study / "prospective-device"
    session = device / "session01"

    def run(script, *args):
        return check_output([sys.executable, "-B", str(script), *map(str, args)], cwd=root)

    for script, args, expected in (
        (device / "experiment.py", ["analyze", "--out", session], session / "summary.json"),
        (device / "audit.py", [session], session / "audit.json"),
        (study / "theorem-audit/check.py", [], study / "theorem-audit/examples.json"),
        (study / "derivation-corrections/scope_check.py", [], study / "derivation-corrections/scope-results.json"),
    ):
        if json.loads(run(script, *args)) != read(expected):
            raise ValueError(f"Recomputation disagrees with {expected.name}")

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        oracle = directory / "verified-candidates.json"
        run(study / "verified-candidate/check.py", "--out", oracle)
        if read(oracle) != read(study / "verified-candidate/results.json"):
            raise ValueError("Finite scheduling oracle disagrees")
        replay = directory / "opportunity"
        run(study / "opportunity/replay.py", "--out", replay)
        for name in ("summary.json", "questions.csv"):
            reference = study / "opportunity/results" / name
            if (replay / name).read_text("utf8") != reference.read_text("utf8"):
                raise ValueError(f"Three-path replay disagrees with {reference.name}")

    original = read(session / "prediction.json")["panels"]
    observed = read(session / "audit.json")["panels"]
    report = {
        "files_matching_manifest": len(manifest),
        "three_path_predicted_saving_pct": original["witness"]["effect_pct"],
        "three_path_wall_median_saving_pct": observed["witness"]["wall_median_pct"],
        "two_path_control_wall_median_saving_pct": observed["control"]["wall_median_pct"],
        "full_200_question_opportunity_table_matches": True,
        "separation_examples_and_finite_oracle_match": True,
        "fixed_panel_scope_and_timing_examples_match": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run without -O so the frozen scientific checks stay enabled.")
    main()
