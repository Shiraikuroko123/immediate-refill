# When Is Immediate Refill Optimal for Parallel Voting?

[Interactive project page](https://shiraikuroko123.github.io/immediate-refill/) · [中文项目页](https://shiraikuroko123.github.io/immediate-refill/zh.html) · [English paper](docs/downloads/immediate-refill-paper.pdf) · [中文论文](docs/downloads/immediate-refill-zh.pdf) · [中文研究说明](docs/downloads/中文研究说明.md)

Under an ordered voting protocol with zero extra launches, immediate refill is universally optimal exactly when decoding-step costs are discretely concave over reachable widths. The paper gives a tight worst-case latency ratio over variable panel sizes and quotas. On one A800, the calibrated clock predicted a 3.13% saving from delaying the third of three fixed-length requests. The median measured saving across 12 paired blocks was 2.46%.

Follow a [five-path voting example](https://shiraikuroko123.github.io/immediate-refill/#method) to see which launches are safe. The [experiment walkthrough](https://shiraikuroko123.github.io/immediate-refill/#experiments) explains how the A800 prediction was tested and why the information available to a scheduler changes the comparison. In a separate fixed-clock replay of 200 GSM8K problems, mean savings fell from 8.71% against a contiguous-prefix baseline to 0.30% against a baseline that also counts out-of-prefix votes. [Inspect all three comparisons for any problem](https://shiraikuroko123.github.io/immediate-refill/#explore).

| Resource | Contents |
| --- | --- |
| [English LaTeX source](docs/downloads/immediate-refill-source.zip) | Source, bibliography, figures and plotting inputs for the current manuscript |
| [Evidence ZIP](docs/downloads/immediate-refill-supplement.zip) | Raw measurement records, fixed workloads, scheduling code, theorem checks and scripts for CPU replay |
| [Reference results](docs/downloads/replay-results.json) | Compare your CPU replay output with the published device, archive and theory results |
| [File checksums](docs/downloads/SHA256SUMS.txt) | SHA-256 digests of the current paper, source, evidence and Chinese edition |

Extract the evidence ZIP into an empty directory. In that directory, using a fresh Python 3.10 environment, run

```sh
python -m pip install -r requirements.txt
python -B reproduce.py --out results.json
```

The report reconstructs the recorded device results, the archived panel comparisons and the sensitivity analyses. The included README gives the figure commands and describes each input and output.

The [server archive](server-data/README.md) preserves the A800 and H20 research files inventoried on 21 November 2026. To check all 732 files from the root of a clone, run `python -B server-data/verify.py`. The earlier [reviewer snapshot](review/README.md) and [original ZIP](dist/scheduling-review.zip) remain available for historical comparisons.
