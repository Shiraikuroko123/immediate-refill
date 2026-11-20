"""Check the finite witnesses in the proof corrections using exact virtual ticks."""
import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import product
import json
from pathlib import Path


def locked(labels, n):
    counts = sorted(Counter(x for x in labels if x != "_").values(), reverse=True) + [0, 0]
    return counts[0] - counts[1] > n - len(labels)


def serial(votes):
    return next((i for i in range(1, len(votes)) if locked(votes[:i], len(votes))), len(votes))


@lru_cache(None)
def earliest(partial):
    unknown = [i for i, v in enumerate(partial) if v is None]
    outcomes = []
    for values in product("AB_", repeat=len(unknown)):
        filled = list(partial)
        for i, v in zip(unknown, values):
            filled[i] = v
        outcomes.append(serial(filled))
    return min(outcomes)


def partial_votes(n, finished, labels):
    return tuple(labels[i] if i in finished else None for i in range(n))


def schedules(n, work, votes):
    # Compare every integer-time launch choice with greedy on the same panel.
    # State entries record the remaining work of the consecutive started paths.
    quota = slots = 3
    clock = (0, 10, 11, 13)

    @lru_cache(None)
    def visit(remaining):
        done = {i for i, w in enumerate(remaining) if w == 0}
        if len(done) == quota or locked([votes[i] for i in done], n):
            return 0, 0, 0
        limit = min(earliest(partial_votes(n, done, votes)), len(done) + slots, quota)
        options = []
        for starts in range(len(remaining), limit + 1):
            running = remaining + work[len(remaining):starts]
            width = sum(w > 0 for w in running)
            if width == 0:
                continue
            next_state = tuple(max(0, w - 1) for w in running)
            greedy, best, max_width = visit(next_state)
            options.append((clock[width] + greedy, clock[width] + best, max(width, max_width)))
        return options[-1][0], min(x[1] for x in options), max(x[2] for x in options)

    return visit(())


def concrete_trace(n, q, slots, allowance, work, votes, starts):
    # Use chronological virtual events, processing all finishes before same-time starts.
    assert len(starts) <= q and list(starts) == sorted(starts)
    finishes = [t + w for t, w in zip(starts, work)]
    times = sorted(set(starts) | set(finishes))
    launched = 0
    done = set()
    durations = Counter()
    previous = 0
    for t in times:
        durations[launched - len(done)] += t - previous
        done.update(i for i, end in enumerate(finishes) if end == t)
        new = starts.count(t)
        if locked([votes[i] for i in done], n) or len(done) == q:
            assert launched == len(starts) and not new
            return t, dict(sorted((k, d) for k, d in durations.items() if d))
        bound = min(q, len(done) + slots, earliest(partial_votes(n, done, votes)) + allowance)
        assert launched + new <= bound
        launched += new
        previous = t
    raise ValueError("The proposed execution does not reach a decision")


def check():
    fixed = {}
    for n in (3, 4):
        trials, width = 0, 0
        worst = Fraction(0)
        for votes in product("AB_", repeat=n):
            for work in product((1, 2), repeat=n):
                greedy, best, used = schedules(n, work, votes)
                worst = max(worst, Fraction(greedy, best))
                width = max(width, used)
                trials += 1
        fixed[str(n)] = dict(panels=trials, max_width=width, worst_ratio=str(worst))
    assert fixed["3"] == dict(panels=216, max_width=2, worst_ratio="1")
    assert fixed["4"] == dict(panels=1296, max_width=3, worst_ratio="23/22")

    args = (8, 4, 3, 0, (1,) * 8, ("A",) * 8)
    nominal = (0, 10, 11, 13)
    majorant = (0, 10, Fraction(23, 2), 13)
    threshold = Fraction(45, 4)
    def policy(clock):
        return (0, 0, 1, 1) if clock[2] < threshold else (0, 0, 1, 2)
    def price(starts, clock):
        _, d = concrete_trace(*args, starts)
        return sum(clock[k] * ticks for k, ticks in d.items())
    repricing = [price(policy(nominal), nominal), price(policy(nominal), majorant),
                 price(policy(majorant), majorant)]
    assert repricing == [22, 23, Fraction(63, 2)]

    args = (3, 3, 2, 1, (2, 2, 1), ("A",) * 3)
    greedy_return, _ = concrete_trace(*args, (0, 0))
    delayed_return, _ = concrete_trace(*args, (0, 2, 2))
    assert (greedy_return, delayed_return) == (2, 3)

    args = (2, 2, 2, 0, (10, 1), ("A", "A"))
    greedy_end, _ = concrete_trace(*args, (0, 0))
    waiting_end, _ = concrete_trace(*args, (0, 1))
    compressed_end = waiting_end - 1
    assert compressed_end < greedy_end == waiting_end

    compatible = [v for v in product("AB", repeat=3) if v[0] != v[1]]
    constrained_minimum = min(map(serial, compatible))
    assert all({v[i] for v in compatible} == {"A", "B"} for i in range(3))
    assert earliest((None,) * 3) == 2 and constrained_minimum == 3

    labels = tuple("AAABBABBB")
    earliest_stops = [earliest(partial_votes(9, indices, labels))
                      for indices in ({1, 3}, {2, 4}, {1, 2})]
    assert earliest_stops == [6, 6, 5]
    return dict(fixed_quota_three_and_three_slots=fixed,
                wall_timer_prices=[str(x) for x in repricing],
                return_before_comparator_starts_third=[greedy_return, delayed_return],
                compressed_end_and_greedy_decision=[compressed_end, greedy_end],
                earliest_serial_free_and_correlated_votes=[2, constrained_minimum],
                nine_vote_predecessor_example=dict(earliest=earliest_stops,
                    sixth_start_at_B0=[6 <= k for k in earliest_stops],
                    sixth_start_at_B1=[6 <= k + 1 for k in earliest_stops]))


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run without -O for the finite checks.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = json.dumps(check(), indent=2) + "\n"
    if args.out:
        with args.out.open("x", encoding="utf-8", newline="\n") as f:
            f.write(text)
    else:
        print(text, end="")
