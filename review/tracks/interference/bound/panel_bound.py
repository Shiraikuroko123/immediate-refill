"""Whole-panel lower bounds under an occupancy-dependent shared virtual clock."""
from collections import Counter
from functools import lru_cache
from itertools import combinations
from fractions import Fraction as F
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tracks/lookahead/theory"))
from frontier import minimum_prefix


def locked(votes, total):
    totals = Counter(v for v in votes if v is not None)
    counts = sorted(totals.values(), reverse=True) + [0, 0]
    return counts[0] - counts[1] > total - len(votes)


def earliest(lengths, votes, slots, quota, excess):
    """Run unit-rate greedy. To lower-bound finishes beyond its decision, continue with only the cap and slots."""
    obs, active, ends, widths = {}, {}, [0] * quota, Counter()
    clock = used = 0
    first = None
    while used < quota or active:
        if first is None and locked(list(obs.values()), len(votes)):
            first = clock
        limit = min(quota, slots + len(obs))
        if first is None:
            limit = min(limit, minimum_prefix(len(votes), obs) + excess)
        for i in range(used, limit):
            active[i] = ends[i] = clock + lengths[i]
        used = limit
        next_finish = min(active.values())
        if first is None:
            widths[len(active)] += next_finish - clock
        clock = next_finish
        for i in [i for i, t in active.items() if t == clock]:
            obs[i] = votes[i]
            del active[i]
    return first if first is not None else clock, ends, widths


def lower_demands(lengths, votes, slots, quota, excess, variants=None):
    """Enumerate necessary terminal masks and remove pointwise dominated service requirements."""
    ticks, ends, widths = earliest(lengths, votes, slots, quota, excess)
    full = (1 << quota) - 1
    required = []
    predecessors = []
    for j in range(quota):
        predecessors.append(sum(1 << i for i in range(j) if lengths[i] <= lengths[j]))
    valid = 0
    for mask in range(1, full + 1):
        indices = [i for i in range(mask.bit_length()) if mask & (1 << i)]
        if any(mask & predecessors[i] != predecessors[i] for i in indices):
            continue
        if mask != full and not locked([votes[i] for i in indices], len(votes)):
            continue
        # Admission immediately before the last completed index was launched could use
        # only a subset of this terminal evidence. At least one slot then belonged to it.
        last = indices[-1]
        prev_obs = {i: votes[i] for i in indices[:-1]}
        if last + 1 > min(minimum_prefix(len(votes), prev_obs) + excess, slots + len(prev_obs)):
            continue
        peak = 0
        mandatory = []
        for i in range(last, -1, -1):
            if mask & (1 << i):
                peak = max(lengths[i], peak)
            mandatory.append(min(lengths[i], peak))
        mandatory.sort(reverse=True)
        vector = [max(ticks, max(ends[i] for i in indices))]
        for m in range(2, slots):
            vector.append(sum(mandatory[:m]))
        if slots > 1:
            vector.append(sum(mandatory))
        if variants is not None:
            done = sorted((lengths[i] for i in indices),reverse=True)
            zeros = (0,) * (slots - 2)
            variants["completed_work"].append((ticks,) + zeros + (sum(done),))
            variants["ordered_work"].append((ticks,) + zeros + (sum(mandatory),))
            variants["release_and_ordered_work"].append((vector[0],) + zeros + (sum(mandatory),))
            variants["ordered_top_m"].append((ticks,) + tuple(vector[1:]))
        valid += 1
        vec = tuple(vector)
        if any(all(a <= b for a, b in zip(old, vec)) for old in required):
            continue
        required = [old for old in required if not all(a <= b for a, b in zip(vec, old))]
        required.append(vec)
    if not required:
        raise AssertionError("Lost every candidate terminal state")
    return ticks, widths, sorted(required), valid


def solve_exact(matrix, rhs):
    n = len(rhs)
    a = [[F(x) for x in row] + [F(b)] for row, b in zip(matrix, rhs)]
    for j in range(n):
        pivot = next((i for i in range(j,n) if a[i][j]), None)
        if pivot is None:
            return None
        a[j], a[pivot] = a[pivot], a[j]
        divisor = a[j][j]
        a[j] = [x / divisor for x in a[j]]
        for i in range(n):
            if i != j and a[i][j]:
                scalar = a[i][j]
                a[i] = [p - scalar*q for p,q in zip(a[i],a[j])]
    return tuple(row[-1] for row in a)


@lru_cache(maxsize=None)
def dual_vertices(costs):
    """All feasible dual extreme points, with exact rational arithmetic (small P)."""
    p = len(costs)
    rows = [tuple(min(m, k) for m in range(1, p + 1)) for k in range(1, p + 1)]
    rows.extend(tuple(-int(i == j) for j in range(p)) for i in range(p))
    rhs = costs + (0,) * p
    vertices = set()
    for ix in combinations(range(2*p), p):
        point = solve_exact([rows[i] for i in ix], [rhs[i] for i in ix])
        if point is not None and all(sum(a*b for a,b in zip(row,point)) <= v for row,v in zip(rows,rhs)):
            vertices.add(point)
    return sorted(vertices)


def price(vectors, costs):
    dual = dual_vertices(tuple(costs))
    return min(max(sum(a*b for a,b in zip(d,y)) for y in dual) for d in vectors)
