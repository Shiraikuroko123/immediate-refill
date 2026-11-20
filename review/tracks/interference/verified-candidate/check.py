"""Integer-time oracle for ordered collection of a target count of verified candidates."""
import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import product
import json
from math import lcm
from pathlib import Path


def serial_stop(outcomes, target):
    passes = 0
    for index, value in enumerate(outcomes, 1):
        passes += value
        if passes == target:
            return index
    return len(outcomes)


@lru_cache(None)
def compatible_stop(observations, target):
    """Enumerate all completions of the partially known binary verification results."""
    missing = [i for i, value in enumerate(observations) if value is None]
    possible = []
    for filling in product((0, 1), repeat=len(missing)):
        completed = list(observations)
        for index, flag in zip(missing, filling):
            completed[index] = flag
        possible.append(serial_stop(completed, target))
    return min(possible)


def panel_oracle(outcomes, work, target, slots):
    size = len(work)
    expected = serial_stop(outcomes, target)

    @lru_cache(None)
    def explore(remaining):
        launched = len(remaining)
        observed = [outcomes[i] if i < launched and remaining[i] == 0 else None for i in range(size)]
        passes = observed.count(1)
        failures = observed.count(0)
        done = passes + failures
        exact = compatible_stop(tuple(observed), target)
        limit = min(exact, done + slots)
        assert limit == min(size, done + slots, target + failures)
        assert done <= launched <= limit
        if passes == target or done == size:
            assert done == expected == launched
            return (0,) * (slots-1), (0,) * (slots-1), 1

        max_features = [0] * (slots-1)
        greedy_features = None
        histories = 0
        for new_count in range(launched, limit + 1):
            running = remaining + work[launched:new_count]
            width = sum(n > 0 for n in running)
            if width == 0:
                continue   # Empty-system waiting cannot improve a schedule.
            child = tuple(max(0, n-1) for n in running)
            greedy_child, best_child, count = explore(child)
            histories += count
            marginal = [max(0, width-j) for j in range(1, slots)]
            max_features = [max(old, bonus+best) for old, bonus, best in zip(max_features, marginal, best_child)]
            if new_count == limit:
                greedy_features = tuple(bonus+later for bonus,later in zip(marginal,greedy_child))
        assert greedy_features is not None
        return greedy_features, tuple(max_features), histories

    greedy, best, histories = explore(())
    assert greedy == best
    return explore.cache_info().currsize, len(greedy), histories


def all_fail_profile(work, target, slots):
    remaining = []
    occupancy = Counter()
    while sum(v == 0 for v in remaining) < len(work):
        observed = [0 if i < len(remaining) and remaining[i] == 0 else 1 for i in range(len(work))]
        earliest_stop = serial_stop(observed, target)
        limit = min(earliest_stop, observed.count(0)+slots)
        remaining.extend(work[len(remaining):limit])
        width = sum(v > 0 for v in remaining)
        assert width > 0
        occupancy[width] += 1
        remaining = [max(0, v-1) for v in remaining]
    return occupancy


def witnesses():
    checked = 0
    for slots in range(2, 9):
        for b in range(2, slots+1):
            for j in range(1, b):
                for a in range(j):
                    shorts = lcm(b-a, j-a)
                    long = shorts // (j-a)
                    high = shorts // (b-a)
                    work = (long,)*a + (1,)*shorts
                    immediate = all_fail_profile(work, b, slots)
                    staged = all_fail_profile(work, b, j)
                    target_profile = Counter({b:high})
                    if a>0:
                        target_profile[a] += long-high
                    assert immediate == target_profile and staged == Counter({j:long})
                    # Use a positive clock with a single defect at j and concave majorant c_hat[k] = k.
                    price = [Fraction(k) if k != j else Fraction(k,2) for k in range(slots+1)]
                    immediate_time = sum(count*price[k] for k,count in immediate.items())
                    staged_time = sum(count*price[k] for k,count in staged.items())
                    assert immediate_time == 2*staged_time
                    checked += 1
    return checked


def main():
    configs = states = thresholds = histories = 0
    for size in range(1, 5):
        for outcomes in product((0,1), repeat=size):
            for work in product((1,2), repeat=size):
                for target in range(1,size+1):
                    for slots in range(1,size+1):
                        counts = panel_oracle(outcomes,work,target,slots)
                        states += counts[0]
                        thresholds += counts[1]
                        histories += counts[2]
                        configs += 1
    example_immediate = all_fail_profile((2,1,1), 3, 3)
    example_staged = all_fail_profile((2,1,1), 3, 2)
    costs = (0,10,11,13)
    return dict(exhaustive=dict(panel_configurations=configs,per_configuration_reachable_states_summed=states,
                                nontrivial_threshold_checks=thresholds,integer_histories_summed=histories,
                                distinct_partial_assignment_target_pairs=compatible_stop.cache_info().currsize,
                                size="1..4",work="1 or 2",outcomes="all binary vectors",targets="1..N",slots="1..N"),
                exact_integer_witnesses=witnesses(),witness_slots="2..8",
                example=dict(work=[2,1,1],target=3,outcomes=[0,0,0],costs=list(costs),
                             immediate=sum(costs[k]*n for k,n in example_immediate.items()),
                             staged=sum(costs[k]*n for k,n in example_staged.items())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    report = json.dumps(main(),indent=2) + "\n"
    if args.out:
        with args.out.open("xb") as stream:
            stream.write(report.encode("utf8"))
    else:
        print(report,end="")
