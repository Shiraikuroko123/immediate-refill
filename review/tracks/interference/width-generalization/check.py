"""Independently optimize each threshold of the occupancy majorization on integer-time schedules."""
from argparse import ArgumentParser
from collections import Counter
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
from random import Random


def margin(labels):
    counts = sorted(Counter(x for x in labels if x is not None).values(), reverse=True) + [0, 0]
    return counts[0] - counts[1]


def oracle(work, votes, quota, slots):
    n = len(votes)

    @lru_cache(None)
    def compatible_prefix(done):
        # At prefix t all its unseen entries may support the currently leading label.
        reached = []
        missing = 0
        for j in range(n - 1):
            if done & (1 << j):
                reached.append(votes[j])
            else:
                missing += 1
            if margin(reached) + missing > n - j - 1:
                return j + 1
        return n

    serial = compatible_prefix((1 << n) - 1)
    terminal_work = sum(work[:min(serial, quota)])

    @lru_cache(None)
    def explore(remaining):
        seen = sum((1 << j) for j, w in enumerate(remaining) if w == 0)
        labels = [votes[j] for j in range(len(remaining)) if seen & (1 << j)]
        if len(labels) == quota or margin(labels) > n - len(labels):
            assert len(labels) == len(remaining) == min(serial, quota)
            return (0,) * (slots + 1), (0,) * (slots + 1), 1
        bound = min(compatible_prefix(seen), quota, len(labels) + slots)
        best = [-1] * (slots + 1)
        trajectories = 0
        greedy = None
        for count in range(len(remaining), bound + 1):
            residual = remaining + work[len(remaining):count]
            width = sum(w > 0 for w in residual)
            if width == 0:
                continue
            after = tuple(max(0, w - 1) for w in residual)
            subbest, subgreedy, paths = explore(after)
            trajectories += paths
            for j in range(0, slots + 1):
                best[j] = max(best[j], max(0, width - j) + subbest[j])
            if count == bound:
                greedy = tuple(max(0, width - j) + subgreedy[j] for j in range(slots + 1))
        assert greedy is not None
        return tuple(best), greedy, trajectories

    best, greedy, paths = explore(())
    assert best[0] == greedy[0] == terminal_work
    assert best == greedy, (work, votes, quota, slots, best, greedy)
    return paths, explore.cache_info().currsize


def main():
    rng = Random(11027)
    groups = {}
    seen = set()

    def check(work, votes, quota, slots, group):
        key = (work, votes, quota, slots)
        if key in seen:
            return
        seen.add(key)
        paths, states = oracle(work, votes, quota, slots)
        counts = groups.setdefault(group, dict(configurations=0, trajectories=0, states=0, nontrivial_thresholds=0))
        counts["configurations"] += 1
        counts["trajectories"] += paths
        counts["states"] += states
        counts["nontrivial_thresholds"] += slots - 1

    # Exhaustively vary the launched jobs of a six-vote, four-slot, quota-four panel.
    # The two suffix votes are never launched and do not affect the run.
    for jobs, votes in product(product((1, 2), repeat=4), product((None, "A", "B"), repeat=4)):
        check(jobs + (1, 1), votes + (None, None), 4, 4, "exhaustive_n6_p4_q4")
    # These additional larger panels can replenish full occupancy after a completion.
    for slots in (4, 5, 6, 7, 8):
        n = 2 * slots
        for _ in range(24):
            jobs = tuple(rng.choices((1, 1, 2, 3), k=n))
            votes = tuple(rng.choices(("A", "A", "B", None), k=n))
            # Allow one to four replenishments; states and all admissible histories are enumerated.
            quota = min(n, slots + rng.randrange(1, 5))
            check(jobs, votes, quota, slots, f"seeded_p{slots}")
    return dict(seed=11027, groups=groups, totals={k:sum(v[k] for v in groups.values())
                     for k in next(iter(groups.values()))})


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    report = json.dumps(main(), indent=2)
    if args.out:
        with args.out.open("x", encoding="utf8") as f:
            f.write(report + "\n")
    print(report)
