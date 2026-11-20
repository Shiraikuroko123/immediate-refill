"""Draw the two comparisons with their separate denominators."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
records = []
with (STUDY/"holdout/results/questions.csv").open(encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        if row["context"]=="3968" and row["arm"]=="indexed_vs_buffer":
            records.append(100*(1-float(row["time"])/float(row["base_time"])))
gains = np.array(sorted(records))
assert len(gains)==200
raw = json.loads((STUDY/"bound/measured/summary.json").read_bytes())["summary"]
caps = {(r["cohort"],r["context"]):100*r["mean_ceiling"] for r in raw}
new = json.loads((STUDY/"holdout/ceilings/summary.json").read_bytes())["mean_ceiling"]
caps.update({("G",int(c)):100*v for c,v in new.items()})
plt.rcParams.update({"font.size":8,"axes.titlesize":9,"axes.titleweight":"semibold","pdf.fonttype":42})
fig, (left,right)=plt.subplots(1,2,figsize=(6.7,2.6),layout="constrained")
left.axhline(0,color="#7f858b",lw=.7)
left.axhline(gains.mean(),color="#b34d27",lw=1.1,ls="--",label=f"mean {gains.mean():.2f}%")
left.plot(np.arange(1,len(gains)+1),gains,color="#205f87",lw=1.5)
left.set(xlabel="GSM8K questions ordered by saving",ylabel="Time saved vs. buffered (%)",
         title="A. Using out-of-order votes")
left.legend(frameon=False,fontsize=7,loc="upper left")
labels=["AIME","AMC","Olympiad","GSM8K"]
for index,(n,color) in enumerate([(512,"#acaeac"),(2048,"#6f9ab3"),(3968,"#1b5478")]):
    right.scatter(np.arange(4)+(index-1)*.16,[caps[c,n] for c in "AMOG"], s=21,
                  c=color,marker=["o","^","s"][index],label=f"{n:,}")
right.set_xticks(range(4),labels=labels)
right.set(xlabel="Fixed-trajectory cohort", ylabel="Saving ceiling vs. indexed (%)",
          title="B. Room for further rescheduling",ylim=(0,1))
right.legend(title="Clock context (tokens)",ncol=3,loc="upper left",frameon=False,
             fontsize=6,title_fontsize=6,handletextpad=.2,columnspacing=.55)
for ax in (left,right):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.13,lw=.6)
fig.savefig(HERE/"evidence.pdf",bbox_inches="tight")
fig.savefig(HERE/"evidence.png",bbox_inches="tight",dpi=180)
