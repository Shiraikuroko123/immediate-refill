"""Budgeted ordered admission under strict plurality stopping.

Public functions use zero-based indices and None for a completed abstention.
Absent keys correspond to unfinished (or unstarted) paths.
"""
from collections.abc import Hashable, Mapping


def minimum_prefix(n: int, observed: Mapping[int, Hashable | None]) -> int:
    """Least serial stopping length compatible with a partial vote vector."""
    if n < 1 or any(i < 0 or i >= n for i in observed):
        raise ValueError("Need positive n and zero-based indices in [0, n).")
    counts = {}
    first = second = known = 0
    leader = None
    for index in range(n - 1):
        if index in observed:
            known += 1
            label = observed[index]
            if label is not None:
                count = counts.get(label, 0) + 1
                counts[label] = count
                if label == leader:
                    first = count
                elif count > first:
                    second, first, leader = first, count, label
                elif count > second:
                    second = count
        length = index + 1
        if first - second + length - known > n - length:
            return length
    return n


def launch_limit(n: int, slots: int, budget: int, observed: Mapping[int, Hashable | None]) -> int:
    """Maximal launched count after nonterminal completions, before dispatch."""
    if slots < 1 or budget < 0:
        raise ValueError("Slots must be positive and budget nonnegative.")
    return min(n, minimum_prefix(n, observed) + budget, len(observed) + slots)
