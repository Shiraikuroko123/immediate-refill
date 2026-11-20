"""Admission with different amounts of returned vote-to-index information.

Path indices in observations are zero-based. The candidate's ordinal j is one-based.
"""
from collections import Counter
from itertools import combinations


def gap(votes):
    counts = Counter(v for v in votes if v is not None)
    first, second = (sorted(counts.values(), reverse=True) + [0, 0])[:2]
    return first - second


def prefix_size(observed):
    h = 0
    while h in observed:
        h += 1
    return h


def largest_gap(observed, candidate, allowance, information):
    """Largest compatible margin at the one prefix that decides the next admission."""
    t = max(0, candidate - allowance - 1)
    if information == "indexed":
        known = [vote for index, vote in observed.items() if index < t]
        return gap(known) + t - len(known)
    if information not in ("bag", "prefix_bag"):
        raise ValueError(information)
    h = min(prefix_size(observed), t) if information == "prefix_bag" else 0
    prefix = [observed[i] for i in range(h)]
    bag = [vote for i, vote in observed.items() if i >= h]
    outside = sum(i >= t for i in observed)
    # The completed index mask is observable. Exactly this many ballots must
    # occupy positions after t. Duplicated ballots do not change the maximum.
    kept_count = len(bag) - outside
    missing = t - len(prefix) - kept_count
    return max(gap(prefix + list(kept)) + missing for kept in combinations(bag, kept_count))


def allowed(observed, candidate, allowance, total, information):
    t = max(0, candidate - allowance - 1)
    return largest_gap(observed, candidate, allowance, information) <= total - t


class WindowAdmission:
    """Exact budget test for the next launch. The engine enforces slots and quota."""

    def __init__(self, total, allowance):
        self.total = total
        self.allowance = allowance
        self.launched = 0
        self.prefix_counts = Counter()
        self.prefix_returns = 0
        self.recent = {}

    @property
    def cutoff(self):
        return max(0, self.launched - self.allowance)

    def _add_to_prefix(self, vote):
        self.prefix_returns += 1
        if vote is not None:
            self.prefix_counts[vote] += 1

    def on_launch(self):
        old_cutoff = self.cutoff
        self.launched += 1
        if self.cutoff > old_cutoff and old_cutoff in self.recent:
            self._add_to_prefix(self.recent.pop(old_cutoff))

    def on_return(self, index, vote):
        # The engine supplies once-only completion events, including abstentions.
        assert 0 <= index < self.launched
        if index < self.cutoff:
            self._add_to_prefix(vote)
        else:
            self.recent[index] = vote
        assert len(self.recent) <= self.allowance

    def can_launch(self):
        top, second = (sorted(self.prefix_counts.values(), reverse=True) + [0, 0])[:2]
        possible_margin = top - second + self.cutoff - self.prefix_returns
        return possible_margin <= self.total - self.cutoff


def simulate(lengths, votes, allowance, quota, slots, information):
    total = len(votes)
    time = 0
    active, observed, starts = {}, {}, []
    occupancy = [0] * slots
    window = WindowAdmission(total, allowance)
    while len(observed) < quota and gap(observed.values()) <= total - len(observed):
        for j in range(len(starts) + 1, min(quota, len(observed) + slots) + 1):
            exact = window.can_launch()
            assert exact == allowed(observed, j, allowance, total, "indexed")
            if information == "indexed":
                accept = exact
            else:
                accept = allowed(observed, j, allowance, total, information)
            if not accept:
                break
            assert exact
            window.on_launch()
            starts.append(time)
            active[j-1] = time + lengths[j-1]
        assert active
        finish = min(active.values())
        occupancy[len(active) - 1] += finish - time
        time = finish
        for index in [i for i, end in active.items() if end == time]:
            observed[index] = votes[index]
            window.on_return(index, votes[index])
            del active[index]
    return dict(starts=starts, occ=occupancy)
