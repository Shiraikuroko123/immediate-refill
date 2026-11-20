"""Check the upper-hull lower-bound construction against the indexed launch test."""
import argparse
from collections import Counter
from fractions import Fraction
from itertools import combinations
import json
from math import lcm
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tracks/lookahead/theory"))
from frontier import minimum_prefix


def simulate(physical_slots, left, middle, right, suffix_limit):
    shorts = lcm(right - left, middle - left)
    long_length = shorts // (middle-left)
    prefix = physical_slots * ((shorts + left + physical_slots) // physical_slots + 1)
    total = 2 * (prefix + right - 1)
    quota = prefix + left + shorts
    epsilon = Fraction(1, 1000)
    work = [epsilon]*prefix + [Fraction(long_length)]*left + [Fraction(1)]*shorts
    votes = ["A"]*prefix + ["B"]*(left+shorts)
    seen, running, durations = {}, {}, Counter()
    started, tick = 0, Fraction(0)
    while True:
        counts = sorted(Counter(seen.values()).values(),reverse=True)+[0,0]
        if counts[0]-counts[1] > total-len(seen) or started==quota and not running:
            break
        limit = physical_slots if len(seen)<prefix else suffix_limit
        end = min(minimum_prefix(total, seen), quota, len(seen)+limit)
        for index in range(started,end):
            running[index]=tick+work[index]
        started=end
        finish = min(running.values())
        durations[len(running)] += finish-tick
        for index in [i for i, time in running.items() if time==finish]:
            seen[index]=votes[index]
            del running[index]
        tick=finish
    assert started == quota == len(seen)
    prefix_duration = prefix * epsilon / physical_slots
    assert durations[physical_slots]>=prefix_duration
    durations[physical_slots] -= prefix_duration
    return +durations, shorts, long_length


def check():
    groups = {}
    for slots in range(2,9):
        counts=0
        for left, middle, right in combinations(range(slots+1),3):
            greedy, shorts, long_length = simulate(slots,left,middle,right,slots)
            limited, _, _ = simulate(slots,left,middle,right,middle)
            full_time = Fraction(shorts,right-left)
            expected = Counter({right:full_time})
            if left:
                expected[left]=long_length-full_time
            assert greedy == expected
            assert limited == {middle:long_length}
            counts+=1
        groups[slots]=counts
    return dict(chords=sum(groups.values()),chords_by_physical_slots=groups,
                prefix_work="1/1000",both_event_profiles_verified=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    text=json.dumps(check(),indent=2)+"\n"
    if args.out:
        with args.out.open("x",encoding="utf8") as stream:
            stream.write(text)
    print(text,end="")
