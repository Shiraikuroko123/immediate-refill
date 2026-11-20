"""Cross every archived ordered length vector with every archived ordered vote vector."""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
QUOTA_ROOT = HERE.parent
sys.path.insert(0, str(QUOTA_ROOT))
from summarize import earliest, tally, target

N, P, Q, B = 16, 4, 14, 1


@dataclass(frozen=True)
class Panel:
    cohort: str
    question: int
    panel: int
    lengths: tuple
    votes: tuple

    @property
    def key(self):
        return self.cohort, self.question, self.panel


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load():
    hashes = {}
    original_aime = QUOTA_ROOT.parent / "lookahead/heldout/traces.csv"
    hashes["AIME25/traces.csv"] = file_hash(original_aime)
    rows = defaultdict(list)
    for row in csv_rows(original_aime):
        rows[int(row["qid"]), int(row["panel"])].append(
            (int(row["order"]), int(row["length"]), row["final"] or None))
    panels = []
    for (qid, panel), records in sorted(rows.items()):
        records.sort()
        assert [r[0] for r in records] == list(range(N))
        panels.append(Panel("A", qid, panel, tuple(r[1] for r in records), tuple(r[2] for r in records)))
    assert len(panels) == 60 and len({p.question for p in panels}) == 30
    for qid in range(24):
        path = QUOTA_ROOT / "raw" / f"q{qid:02d}-natural-full.json"
        ref = json.loads(path.read_bytes())
        assert (ref["qid"], ref["condition"], ref["policy"], ref["completed"]) == (qid, "natural", "full", N)
        assert [p["index"] for p in ref["paths"]] == list(range(N))
        hashes["OlympiadBench/" + path.name] = file_hash(path)
        panels.append(Panel("O", qid, 0, tuple(p["length"] for p in ref["paths"]),
                            tuple(p["vote"] for p in ref["paths"])))
    assert all(min(p.lengths) > 0 for p in panels)
    return panels, hashes


@lru_cache(maxsize=None)
def history(votes, mask):
    observed = {i: vote for i, vote in enumerate(votes) if mask & (1 << i)}
    winner, gap = tally(observed.values())
    return winner, gap > N-len(observed), target("exact", Q, observed), target("buffer", Q, observed)


def simulate(lengths, votes, gate):
    ends, launched, mask, now = {}, 0, 0, 0
    opportunities, decisions, starts = 0, 0, []
    while True:
        winner, certified, exact, buffered = history(votes, mask)
        if certified or (launched == Q and not ends):
            break
        assert buffered <= exact
        decisions += 1
        opportunities += int(exact > max(launched, buffered))
        limit = exact if gate == "exact" else buffered
        for index in range(launched, limit):
            ends[index] = now + lengths[index]
            starts.append(now)
        launched = max(launched, limit)
        assert 0 < len(ends) <= P
        now = min(ends.values())
        for index in [i for i, t in ends.items() if t == now]:
            mask |= 1 << index
            del ends[index]
    tokens = sum(min(lengths[i], now-start) for i, start in enumerate(starts))
    return dict(steps=now, tokens=tokens, launched=launched, decisions=decisions, opportunities=opportunities,
                certified=certified, winner=winner, starts=starts)


def pair(lengths, votes):
    e = simulate(lengths.lengths, votes.votes, "exact")
    b = simulate(lengths.lengths, votes.votes, "buffer")
    assert (e["winner"], e["certified"]) == (b["winner"], b["certified"])
    diverged = e["starts"] != b["starts"]
    assert diverged == bool(e["opportunities"]) == bool(b["opportunities"])
    assert e["steps"] <= b["steps"]
    assert diverged or e["tokens"] == b["tokens"]
    return dict(length_cohort=lengths.cohort, length_question=lengths.question, length_panel=lengths.panel,
                vote_cohort=votes.cohort, vote_question=votes.question, vote_panel=votes.panel,
                original=int(lengths.key == votes.key), divergent=int(diverged),
                faster=int(e["steps"] < b["steps"]), certified=int(e["certified"]),
                exact_steps=e["steps"], buffer_steps=b["steps"], exact_tokens=e["tokens"], buffer_tokens=b["tokens"],
                exact_decisions=e["decisions"], exact_opportunities=e["opportunities"],
                buffer_decisions=b["decisions"], buffer_opportunities=b["opportunities"])


def grouped_stats(rows):
    by_question = defaultdict(list)
    for r in rows:
        by_question[r["length_question"],r["vote_question"]].append(r)
    question_savings = [1-sum(r["exact_steps"] for r in g)/sum(r["buffer_steps"] for r in g)
                        for g in by_question.values()]
    return dict(panel_pairs=len(rows), question_pairs=len(by_question),
                divergent=sum(r["divergent"] for r in rows),
                divergent_fraction=mean(r["divergent"] for r in rows),
                faster=sum(r["faster"] for r in rows),
                exact_opportunities=sum(r["exact_opportunities"] for r in rows),
                exact_decisions=sum(r["exact_decisions"] for r in rows),
                buffer_opportunities=sum(r["buffer_opportunities"] for r in rows),
                buffer_decisions=sum(r["buffer_decisions"] for r in rows),
                macro_time_saved=mean(question_savings),
                mean_panel_time_saved=mean(1-r["exact_steps"]/r["buffer_steps"] for r in rows),
                pooled_time_saved=1-sum(r["exact_steps"] for r in rows)/sum(r["buffer_steps"] for r in rows))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(out):
    if not __debug__:
        raise ValueError("Run without -O so source and policy checks remain active")
    panels, hashes = load()
    combinations = [pair(l, v) for l in panels for v in panels]
    groups, summaries, units = {}, [], []
    for length_cohort in "AO":
        for vote_cohort in "AO":
            cell = [r for r in combinations if
                    (r["length_cohort"],r["vote_cohort"]) == (length_cohort,vote_cohort)]
            groups[(length_cohort,vote_cohort,"all")] = cell
            if length_cohort == vote_cohort:
                for key, predicate in (
                        ("original",lambda r:r["original"]),
                        ("off_diagonal",lambda r:not r["original"]),
                        ("different_question",lambda r:r["length_question"]!=r["vote_question"])):
                    groups[(length_cohort,vote_cohort,key)] = [r for r in cell if predicate(r)]
    for (length_cohort,vote_cohort,kind), rows in groups.items():
        base = dict(length_cohort=length_cohort, vote_cohort=vote_cohort, pairing=kind)
        summaries.append({**base, **grouped_stats(rows)})
        qgroups = defaultdict(list)
        for r in rows:
            qgroups[r["length_question"],r["vote_question"]].append(r)
        for (lq,vq),group in qgroups.items():
            units.append({**base, "length_question":lq, "vote_question":vq,
                          "panel_pairs":len(group), "divergent":sum(r["divergent"] for r in group),
                          "exact_steps":sum(r["exact_steps"] for r in group),
                          "buffer_steps":sum(r["buffer_steps"] for r in group)})
    # Independently check the original AIME time estimates at the panel level.
    previous = csv_rows(QUOTA_ROOT/"prior-replay/panels.csv")
    expected = {(int(r["qid"]),int(r["panel"]),r["gate"]):r for r in previous if r["dataset"]=="aime25"}
    for r in groups["A","A","original"]:
        for gate in ("exact","buffer"):
            p = expected[r["length_question"],r["length_panel"],gate]
            for metric in ("steps","tokens"):
                assert r[gate+"_"+metric] == int(p[metric])
    hashes["original_aime_check"] = file_hash(QUOTA_ROOT/"prior-replay/panels.csv")
    # Check how often the two actual fixed A800 engine schedules issued different starts.
    for r in groups["O","O","original"]:
        qid = r["length_question"]
        logs = []
        for gate in ("exact","buffer"):
            file = QUOTA_ROOT/"raw"/f"q{qid:02d}-fixed-{gate}.json"
            raw = json.loads(file.read_bytes())
            assert raw["reference_sha256"] == hashes[f"OlympiadBench/q{qid:02d}-natural-full.json"]
            hashes["OlympiadBench/"+file.name] = file_hash(file)
            logs.append([(e["step"],e["index"]) for e in raw["events"] if e["kind"]=="launch"])
        assert (logs[0] != logs[1]) == bool(r["divergent"])
    # Preserve which individual donor panels can create opportunities in each partner cohort.
    donors = []
    for p in panels:
        top = Counter(v for v in p.votes[:Q] if v is not None)
        positive = len([v for v in p.votes[:Q] if v is not None])
        for role in ("length","vote"):
            other = "vote" if role=="length" else "length"
            for partner in "AO":
                group = [r for r in combinations if
                         (r[role+"_cohort"],r[role+"_question"],r[role+"_panel"]) == p.key
                         and r[other+"_cohort"]==partner]
                donors.append(dict(cohort=p.cohort,question=p.question,panel=p.panel,role=role,partner=partner,
                    pairs=len(group),divergent=sum(r["divergent"] for r in group),faster=sum(r["faster"] for r in group),
                    n_distinct_lengths14=len(set(p.lengths[:Q])),
                    length_sum14=sum(p.lengths[:Q]),max_equal_length_multiplicity14=max(Counter(p.lengths[:Q]).values()),
                    observed_labels14=positive,distinct_labels14=len(top),leading_margin14=tally(p.votes[:Q])[1],
                    full_serial_stop=earliest(dict(enumerate(p.votes)))))
    assert len(combinations) == 84**2
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out/"pairs.csv",combinations)
    write_csv(out/"questions.csv",units)
    write_csv(out/"summary.csv",summaries)
    write_csv(out/"donors.csv",donors)
    manifest = dict(stage="post-hoc exhaustive reassignment of observed lengths and votes",
                    n=N,p=P,q=Q,b=B,panels=84,pairings=len(combinations),sources_sha256=hashes,
                    script_sha256=file_hash(Path(__file__)),
                    imported_policy_sha256=file_hash(QUOTA_ROOT/"summarize.py"))
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    main(parser.parse_args().out)
