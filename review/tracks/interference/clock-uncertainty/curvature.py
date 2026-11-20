"""Evaluate the necessary and sufficient clock inequalities on measured A800 blocks."""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from statistics import median


def calculate(path):
    data = path.read_bytes()
    cells = defaultdict(dict)
    for raw in data.splitlines():
        row = json.loads(raw, parse_float=Fraction)
        if row["event"] == "trial":
            key = row["block"], row["width"]
            cell = cells[row["context"]]
            assert key not in cell
            cell[key] = row["elapsed_s"] / row["steps"]

    rows, guarantees = [], []
    for context, cells_for_context in sorted(cells.items()):
        assert len(cells_for_context) == 8 * 4
        blocks = [(0, *(cells_for_context[block, width] for width in range(1, 5))) for block in range(8)]
        medians = tuple(median(values) for values in zip(*blocks))
        # The upper boundary of the convex hull of the (width, price) points.
        cover = []
        for k, value in enumerate(medians):
            candidates = [value]
            for a in range(k):
                for b in range(k + 1, len(medians)):
                    candidates.append(((b-k)*medians[a]+(k-a)*medians[b])/(b-a))
            cover.append(max(candidates))
        assert cover[0] == 0 and all(cover[k]>=medians[k] for k in range(1,5))
        assert all(2*cover[k]>=cover[k-1]+cover[k+1] for k in range(1,4))
        ratio = max(cover[k]/medians[k] for k in range(1,5))
        guarantees.append(dict(context=context, median_ms=[float(1000*x) for x in medians],
                               majorant_ms=[float(1000*x) for x in cover],
                               greedy_over_opt_at_most=float(ratio),
                               saving_over_greedy_at_most=float(1-1/ratio)))
        for j in range(1, 4):
            def excess(c):
                return c[j-1] + c[j+1] - 2*c[j]
            witness_greedy = medians[j-1] + medians[j+1]
            rows.append({
                "context": context, "index": j,
                "violating_blocks": sum(excess(clock) > 0 for clock in blocks),
                "median_clock_excess_ms": float(1000 * excess(medians)),
                "witness_saving_over_greedy_on_median_clock": float(excess(medians) / witness_greedy),
            })
    return {"source_sha256": hashlib.sha256(data).hexdigest(), "blocks_per_context": 8, "rows": rows,
            "zero_allowance_universal_guarantees": guarantees}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    clock_file = Path(__file__).resolve().parent.parent / "hardware/a800-live02/events.jsonl"
    report = json.dumps(calculate(clock_file), indent=2) + "\n"
    if args.out:
        with args.out.open("x", encoding="utf8") as stream:
            stream.write(report)
    print(report, end="")
