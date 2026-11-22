# When Is Immediate Refill Optimal for Parallel Voting?

[Project page](https://shiraikuroko123.github.io/immediate-refill/) · [English paper](docs/downloads/immediate-refill-paper.pdf) · [中文论文](docs/downloads/immediate-refill-zh.pdf) · [中文研究说明](docs/downloads/中文研究说明.md)

Under an ordered voting protocol with zero extra launches, immediate refill is universally optimal exactly when decoding-step costs are discretely concave over reachable widths. The paper gives a tight worst-case latency ratio over variable panel sizes and quotas. On one A800, the calibrated clock predicted a 3.13% saving from delaying the third of three fixed-length requests. The median measured saving across 12 paired blocks was 2.46%.

| Resource | Contents |
| --- | --- |
| [English LaTeX source](docs/downloads/immediate-refill-source.zip) | Source, bibliography, figures and plotting inputs for the current manuscript |
| [Evidence ZIP](docs/downloads/immediate-refill-supplement.zip) | Raw measurement records, fixed workloads, scheduling code, theorem checks and scripts for CPU replay |
| [File checksums](docs/downloads/SHA256SUMS.txt) | SHA-256 digests of the current paper, source, evidence and Chinese edition |

Extract the evidence ZIP into an empty directory. In that directory, using a fresh Python 3.10 environment, run

```sh
python -m pip install -r requirements.txt
python -B reproduce.py --out results.json
```

The report reconstructs the recorded device results, the archived panel comparisons and the sensitivity analyses. The included README gives the figure commands and describes each input and output.

The [server archive](server-data/README.md) preserves the A800 and H20 research files inventoried on 21 November 2026. To check all 732 files from the root of a clone, run `python -B server-data/verify.py`. The earlier [reviewer snapshot](review/README.md) and [original ZIP](dist/scheduling-review.zip) remain available for historical comparisons.
