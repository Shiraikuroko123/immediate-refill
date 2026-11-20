"""Plot the full 32-question A800 study with each question retaining equal weight."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "device-results"
BLUE, RUST, CHARCOAL = "#226699", "#ad4d17", "#283d48"


def inputs(source):
    summary = json.loads((source / "summary.json").read_bytes())
    with (source / "questions.csv").open(encoding="utf8", newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: int(r["qid"]))
    assert summary["question_count"] == len(rows) == 32 and summary["timed_panels"] == 160
    assert [int(row["qid"]) for row in rows] == list(range(32))
    values = np.asarray([[float(r[col]) for r in rows] for col in ("saving_short", "saving_long", "delta")]) * 100
    np.testing.assert_allclose(values[2], values[1] - values[0], atol=1e-12)
    metrics = summary["results"]
    stats = [metrics["384"]["saving"], metrics["1024"]["saving"], metrics["interaction"]]
    intervals = np.asarray([[stat["mean"], *stat["ci"]] for stat in stats]) * 100
    np.testing.assert_allclose(intervals[:, 0], values.mean(axis=1), atol=1e-12)
    return values, intervals


def draw(source, out):
    values, intervals = inputs(source)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig = plt.figure(figsize=(8.5,7.0),facecolor="white")
    grid = fig.add_gridspec(2,2, left=.095,right=.965,bottom=.17,top=.8,
                            height_ratios=(3.8,1),hspace=.42,wspace=.31)
    pair = fig.add_subplot(grid[0,0])
    delta = fig.add_subplot(grid[0,1],sharey=pair)
    avg = fig.add_subplot(grid[1,0])
    change = fig.add_subplot(grid[1,1])
    fig.text(.095,.945,"Does a longer output limit change the value of indexed admission?",
             fontsize=14,weight="bold",ha="left",color=CHARCOAL)
    fig.text(.095,.905,"A800  /  Qwen2.5-1.5B-Instruct  /  32 GSM8K questions",fontsize=10,color="#526470")
    pair.set_title("Time saved at each output limit (%)",loc="left",fontsize=10,weight="bold")
    delta.set_title("Long minus short limit (percentage points)",loc="left",fontsize=10,weight="bold")
    q = np.arange(32)
    for index in range(32):
        pair.plot(values[:2,index],[index-.12,index+.12],c="#8f9ba3",lw=.8,zorder=2)
    pair.scatter(values[0],q-.12,facecolors="white",edgecolors=BLUE,s=18,lw=1.2,zorder=4)
    pair.scatter(values[1],q+.12,color=RUST,marker="^",s=17,lw=.7,zorder=3)
    delta.hlines(q,0,values[2],color="#a5b5bc",lw=.9)
    delta.scatter(values[2],q,c=CHARCOAL,s=13,marker="D",zorder=3)
    pair.set_ylim(32,-1)
    pair.set_yticks([0,4,8,12,16,20,24,28,31])
    pair.set_ylabel("Question index")
    delta.tick_params(axis="y",labelleft=False)
    legend = [Line2D([0],[0], marker="o",markerfacecolor="white",markeredgecolor=BLUE,lw=0,label="384 tokens"),
              Line2D([0],[0], marker="^",color=RUST,lw=0,label="1,024 tokens"),
              Line2D([0],[0], marker="D",color=CHARCOAL,lw=0,label="Within-question change")]
    fig.legend(handles=legend,ncol=3,loc="center left",bbox_to_anchor=(.09,.859),
               fontsize=8.5,frameon=False,handletextpad=.35,columnspacing=1.8)
    for a in (pair,delta,avg,change):
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_color("#a9b4bb")
        a.tick_params(axis="y",length=0,pad=3)
        a.tick_params(axis="x",length=3,width=.6,color="#9aa7ae")
        a.xaxis.set_major_locator(MaxNLocator(nbins=5))
        a.axvline(0,color="#74848c",lw=.85,ls="--",zorder=1)
    pair.grid(axis="y",color="#e4e9ec",lw=.55)
    delta.grid(axis="y",color="#e4e9ec",lw=.55)
    for ax,i,y,c,marker in ((avg,0,.73,BLUE,"o"),(avg,1,.25,RUST,"^"),(change,2,.5,CHARCOAL,"D")):
        m,lo,hi = intervals[i]
        ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt=marker,c=c,mfc="white" if i==0 else c,
                    ms=5,capsize=3,lw=1.3,zorder=3)
    avg.set_ylim(0,1); change.set_ylim(0,1)
    avg.set_yticks([.73,.25],labels=["384","1,024"])
    change.set_yticks([.5],labels=["Δ"])
    for ax, subset in ((avg,intervals[:2]),(change,intervals[2:])):
        lo,hi = min(0,subset[:,1].min()),max(0,subset[:,2].max())
        pad = max(.05,(hi-lo)*.12)
        ax.set_xlim(lo-pad,hi+pad)
        ax.set_title("Question means and 95% intervals",loc="left",fontsize=9,weight="bold")
    fig.text(.095,.086,"Each question uses the buffered wall time at its own output limit as denominator. Positive values favor indexed admission.",
             fontsize=8,color=CHARCOAL)
    fig.text(.095,.060,"Intervals resample 32 questions in pairs (20,000 draws). Mean panels use expanded horizontal scales.",
             fontsize=8,color=CHARCOAL)
    out.mkdir(exist_ok=True,parents=True)
    fig.savefig(out/"cap-comparison.pdf", metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(out/"cap-comparison.png", dpi=185)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--out",type=Path,default=HERE)
    args = parser.parse_args()
    draw(args.source,args.out)
