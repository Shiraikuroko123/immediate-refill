"""Post hoc ablation of out-of-order ballot use and individual identity assignment."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

import numpy as np

from gates import simulate

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent


def main(out):
    runner = runpy.run_path(str(STUDY/"holdout/run.py"))
    clock = STUDY/"hardware/a800-live02/events.jsonl"
    costs = runner["read_curves"](clock, 4)[3968]["median"]
    groups, hashes = runner["inputs"](json.loads((STUDY/"holdout/freeze.json").read_bytes()))
    panels = []
    for key, paths in sorted(groups.items()):
        assert [r[0] for r in sorted(paths)] == list(range(16))
        lengths, votes = zip(*[(w, v) for _, w, v in sorted(paths)])
        panels.append(("archive",key,lengths,votes))
    for path in sorted((STUDY/"cap-diagnostic/raw").glob("q*-natural-full.json")):
        content = path.read_bytes()
        hashes[path.name] = sha256(content).hexdigest()
        ref = json.loads(content)
        for cap in (384,1024):
            lengths = [min(r["length"], cap) for r in ref["paths"]]
            votes = [r["vote"] if r["length"] < cap else None for r in ref["paths"]]
            panels.append((f"device-paths-{cap}",ref["unique_id"],lengths,votes))
    assert len(panels) == 264
    rows = []
    for cohort, key, work, votes in panels:
        outcomes = {name:simulate(work, votes, 1, 14, 4, name) for name in ("bag","prefix_bag","indexed")}
        ref = runner["schedule"](work, votes, "exact")
        assert outcomes["indexed"]["starts"] == ref["starts"] and outcomes["indexed"]["occ"] == ref["occ"]
        outcomes["buffer"] = runner["schedule"](work, votes, "buffer")
        times = {name:sum(c*k for c,k in zip(costs,v["occ"])) for name,v in outcomes.items()}
        for name in ("buffer", "bag", "prefix_bag"):
            rows.append(dict(cohort=cohort, question=key, comparator=name,
                             indexed_time=float(times["indexed"]), comparator_time=float(times[name]),
                             saving=float(1-times["indexed"]/times[name]),
                             dispatch_changes=outcomes["indexed"]["starts"] != outcomes[name]["starts"]))
    summary = []
    for cohort in dict.fromkeys(r["cohort"] for r in rows):
        for name in ("buffer", "bag", "prefix_bag"):
            group = [r for r in rows if r["cohort"]==cohort and r["comparator"]==name]
            gains = np.array([r["saving"] for r in group])
            means = np.random.default_rng(651).choice(gains, (20000,len(gains))).mean(1)
            summary.append(dict(cohort=cohort,comparator=name, questions=len(group),
                                mean_saving=float(gains.mean()), median_saving=float(np.median(gains)),
                                paired_question_ci=np.quantile(means,[.025,.975]).tolist(),
                                changed=sum(r["dispatch_changes"] for r in group),
                                wins=int((gains>1e-12).sum()),losses=int((gains< -1e-12).sum())))
    out.mkdir(parents=True,exist_ok=False)
    with (out/"questions.csv").open("x",newline="",encoding="utf8") as stream:
        writer = csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    result = dict(retrospective=True,clock_context=3968,boot_seed=651,resamples=20000,
                  clock_sha256=sha256(clock.read_bytes()).hexdigest(), source_sha256=hashes, summary=summary)
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
