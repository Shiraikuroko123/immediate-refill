"""Reproduce the two separation examples using integer quarter-ticks."""
from itertools import combinations, product
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "information-baseline"))
from gates import allowed


def earliest_serial(votes):
    for n in range(1, len(votes) + 1):
        diff = abs(votes[:n].count("A") - votes[:n].count("B"))
        if diff > len(votes) - n:
            return n
    return len(votes)


def voting_example():
    votes = ("A", "A", "A", "B", "B", "B", "A", "B", "B")
    first_five_work = (40, 16, 12, 8, 4)
    target = 20
    histories = []
    for wanted in ({2, 4}, {3, 5}, {2, 3}):
        starts = [0] + [
            target - w + (-1 if i in wanted else 1)
            for i, w in enumerate(first_five_work[1:], 2)
        ]
        assert starts == sorted(starts)
        ends = [s + w for s, w in zip(starts, first_five_work)]
        for i, s in enumerate(starts):
            seen = {j: votes[j] for j in range(i) if ends[j] <= s}
            assert i - len(seen) < 5 and allowed(seen, i + 1, 0, 9, "indexed")
        seen = {j: votes[j] for j in range(5) if ends[j] <= target}
        assert {j + 1 for j in seen} == wanted
        feasible = allowed(seen, 6, 0, 9, "indexed")
        # Compare with independent enumeration of all binary vote completions.
        unknown = [i for i in range(len(votes)) if i not in seen]
        lengths = set()
        for labels in product(("A", "B"), repeat=len(unknown)):
            filled = dict(seen)
            filled.update(zip(unknown, labels))
            lengths.add(earliest_serial([filled[i] for i in range(len(votes))]))
        assert feasible == (min(lengths) >= 6)
        histories.append(dict(done=sorted(wanted), starts_in_quarters=starts,
                              active=5 - len(seen), next_allowed=feasible,
                              min_compatible_binary_serial=min(lengths)))
    observed = [set(x["done"]) for x in histories]
    truth = [h["next_allowed"] for h in histories]
    assert truth == [True, True, False]
    # Include every possible predecessor set among all 8 other real jobs.
    other = set(range(1, 10)) - {6}
    realizations = sum(
        [set(pred).issubset(s) for s in observed] == truth
        for n in range(len(other) + 1) for pred in combinations(sorted(other), n)
    )
    assert realizations == 0
    return dict(votes=votes, slots=5, quota=9, witness_time_in_quarters=target,
                work_of_first_five_in_quarters=first_five_work, histories=histories,
                matching_fixed_predecessor_sets=realizations)


def dag_example():
    # Only jobs 3 and 4 depend on job 1. Each job needs four quarter-ticks.
    schedules = {"immediate": [0, 0, 4, 4], "delayed": [0, 4, 4, 6]}
    clock = (0, 1, 2, 2.5)
    rows = {}
    for name, starts in schedules.items():
        assert starts == sorted(starts) and starts[2] >= starts[0] + 4
        width = [sum(s <= t < s + 4 for s in starts) for t in range(max(starts) + 4)]
        assert max(width) <= 3 and sum(width) == 16
        rows[name] = dict(starts_in_quarters=starts, makespan=len(width) / 4,
                          width_per_quarter=width,
                          excess_over_two=sum(max(0, h - 2) for h in width) / 4,
                          wall=sum(clock[h] for h in width) / 4)
    assert rows["immediate"]["wall"] == 4 and rows["delayed"]["wall"] == 3.75
    return dict(clock=clock, schedules=rows)


if __name__ == "__main__":
    print(json.dumps(dict(voting=voting_example(), ordinary_dag=dag_example()), indent=2))
