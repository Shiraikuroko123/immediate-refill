"""Count terminal two-launch states in the paper's original indexed schedules."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"tracks/quota/transfer_diag"))
sys.path.insert(0, str(ROOT/"tracks/impact/design"))
from run import N, Panel, load, simulate
from summarize import tally, target
from amc import fetch


def load_all():
    panels, hashes = load()
    folder = ROOT/"tracks/impact/design"
    manifest = json.loads((folder/"source.json").read_bytes())
    for file in manifest["files"]:
        name = Path(file["path"]).name
        if name.endswith("_b2.jsonl") and not (folder/"raw"/name).is_file():
            raise FileNotFoundError(folder/"raw"/name)
    amc, sources = fetch(2, folder/"raw")
    for source in sources:
        hashes["AMC23/"+source["file"]] = source["sha256"]
    for question, paths in sorted(amc.items()):
        panels.append(Panel("M",question,0,tuple(p[1] for p in paths),tuple(p[2] for p in paths)))
    assert len(panels)==124
    return panels, hashes


def draw_state(panel, quota, observations, launched, active, limit, time):
    matched = (launched == quota - 2 and not active and limit == quota
               and tally(observations.values())[1] <= N-len(observations))
    if not matched:
        return None
    assert list(sorted(observations))==list(range(quota-2))
    votes = list(observations.values())
    margin = tally(votes)[1]
    unresolved_after = N - len(votes) - 1
    # Equal weights and unrestricted unknown labels. A vote for the leader
    # maximizes the attained margin after exactly one additional completion.
    possible = margin + 1 > unresolved_after
    actual = [tally(votes+[panel.votes[i]])[1]>unresolved_after for i in (quota-2, quota-1)]
    assert not any(actual) or possible
    return dict(time=time, margin=margin, possible_either=int(possible),
                actual_first=int(actual[0]), actual_second=int(actual[1]),
                length_first=panel.lengths[quota-2], length_second=panel.lengths[quota-1])


def replay(panel, quota):
    now = 0
    observed, active, starts = {}, {}, []
    hit = None
    while True:
        margin = tally(observed.values())[1]
        if margin>N-len(observed) or (len(starts)==quota and not active):
            break
        limit = target("exact",quota,observed)
        cell = draw_state(panel,quota,observed,len(starts),active,limit,now)
        if cell:
            assert hit is None
            hit = cell
        for i in range(len(starts),limit):
            starts.append(now)
            active[i] = now+panel.lengths[i]
        assert 0 < len(active) <= 4
        now = min(active.values())
        for i in [i for i,t in active.items() if t==now]:
            observed[i] = panel.votes[i]
            del active[i]
    if quota==14:
        comparator = simulate(panel.lengths,panel.votes,"exact")
        assert (now,starts)==(comparator["steps"],comparator["starts"])
    return hit, now


def hardware_state(panel, raw):
    """Replay actual recorded reservations after jointly processing finishes."""
    starts, finishes = {}, {}
    for p in raw["paths"]:
        starts.setdefault(p["launch_step"], []).append(p["index"])
        if p["finished"]:
            finishes.setdefault(p["finish_step"], []).append(p["index"])
    launched = 0
    observed, active = {}, set()
    hit = None
    for t in sorted(starts.keys() | finishes.keys()):
        for i in finishes.get(t, []):
            active.remove(i)
            observed[i] = panel.votes[i]
        if tally(observed.values())[1] > N-len(observed):
            assert not starts.get(t)
            break
        limit = target("exact",14,observed)
        new = starts.get(t, [])
        assert new == list(range(launched, limit))
        cell = draw_state(panel,14,observed,launched,active,limit,t)
        if cell:
            assert hit is None
            hit = cell
        active.update(new)
        launched += len(new)
    return hit


def write_csv(path, rows):
    with path.open("x",encoding="utf8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(out):
    if not __debug__:
        raise ValueError("Verification requires Python without -O")
    panels, hashes = load_all()
    rows = []
    amc_reference = ROOT/"tracks/impact/design/complete/questions.csv"
    with amc_reference.open(encoding="utf8",newline="") as f:
        archived_amc = {r["question"]:int(r["exact"]) for r in csv.DictReader(f)}
    for panel in panels:
        checks = []
        for quota in (14,16):
            hit, time = replay(panel,quota)
            if panel.cohort=="M" and quota==14:
                assert time==archived_amc[panel.question]
            checks.append(("unit-rate replay",quota,hit))
        if panel.cohort=="O":
            file = ROOT/"tracks/quota/raw"/f"q{panel.question:02d}-fixed-exact.json"
            raw = file.read_bytes()
            hashes["OlympiadBench/"+file.name]=hashlib.sha256(raw).hexdigest()
            checks.append(("A800 fixed-work log",14,hardware_state(panel,json.loads(raw))))
        for regime,quota,hit in checks:
            row = dict(cohort=panel.cohort,question=panel.question,panel=panel.panel,
                       measurement=regime,quota=quota,hit=int(hit is not None))
            row.update(hit or dict(time=None,margin=None,possible_either=None,actual_first=None,
                                   actual_second=None,length_first=None,length_second=None))
            rows.append(row)
    grouped = []
    for key in sorted(set((r["measurement"],r["cohort"],r["quota"]) for r in rows)):
        pool=[r for r in rows if (r["measurement"],r["cohort"],r["quota"])==key]
        matched=[r for r in pool if r["hit"]]
        subsets = dict(reached=matched, possible=[r for r in matched if r["possible_either"]],
                       actual_any=[r for r in matched if r["actual_first"] or r["actual_second"]],
                       actual_first_only=[r for r in matched if r["actual_first"] and not r["actual_second"]],
                       actual_second_only=[r for r in matched if r["actual_second"] and not r["actual_first"]],
                       actual_both=[r for r in matched if r["actual_first"] and r["actual_second"]],
                       actual_neither=[r for r in matched if not r["actual_first"] and not r["actual_second"]])
        counts = {f"{name}_{unit}":len(s) if unit=="panels" else len({r["question"] for r in s})
                  for name,s in subsets.items() for unit in ("panels","questions")}
        grouped.append(dict(measurement=key[0],cohort=key[1],quota=key[2],panels=len(pool),
                            questions=len({r["question"] for r in pool}),**counts))
    out.mkdir(exist_ok=False,parents=True)
    write_csv(out/"panels.csv",rows)
    write_csv(out/"summary.csv",grouped)
    (out/"manifest.json").write_text(json.dumps(dict(input_sha256=hashes,
          script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          amc_reference_sha256=hashlib.sha256(amc_reference.read_bytes()).hexdigest()),
          indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args().out)
