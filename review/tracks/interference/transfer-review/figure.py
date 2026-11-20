"""Plot the sealed clock prediction and all of its prospective validation blocks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=ROOT / "manuscript/transfer-review/transfer-blocks.pdf")
args = parser.parse_args()
session = ROOT / "tracks/interference/prospective-device/session01"
measured = json.loads((session / "summary.json").read_bytes())["panels"]
sealed = json.loads((session / "prediction.json").read_bytes())["panels"]
fig, axes = plt.subplots(1, 2, figsize=(7, 2.45), layout="constrained")
for ax, key, title in zip(axes, ("witness", "control"), ("A. Three requests (256, 128, 128)", "B. Two requests (128, 128)")):
    data = measured[key]
    x = [p["block"] + 1 for p in data["pairs"]]
    y = [p["effect_pct"] for p in data["pairs"]]
    ax.axhspan(*data["interval_pct"], color="#4b79a1", alpha=.14)
    ax.axhline(sealed[key]["effect_pct"], color="#ae521f", ls="--", lw=1.5, label="Sealed prediction")
    ax.scatter(x, y, color="#204f77", s=17, label="Measured block", zorder=3)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xticks([1, 3, 6, 9, 12])
    ax.set_xlabel("Validation block", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["right", "top"]].set_visible(False)
axes[0].set_ylabel("Time saved by staging (%)", fontsize=9)
fig.savefig(args.out, metadata={"CreationDate": None, "ModDate": None})
