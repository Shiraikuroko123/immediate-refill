# Recompute the paper's numerical results on a CPU

Use the extracted directory containing `tracks/` as the working directory. The command reads the archived model outputs and returns a JSON report. New analysis files are written to a temporary directory. No GPU account or inference engine is needed.

```shell
python -m pip install -r cpu/requirements.txt
python -B cpu/replay.py --root . --report new-results.json
```

Use Python 3.10 and keep assertions enabled. The program compares regenerated tables with their archived counterparts after normalizing line endings. Fourteen checks cover the previous twelve groups and the two derivation-review corrections. `required-files.json` lists the scientific files accessed by the numerical replay. The installed Requests library is imported by the archived data reader. The eight selected trajectory files are already available locally. The audit worker denies Python network connections and DNS lookup during each analysis.

The calculations include the two prospective A800 contrasts, the opportunity to choose between two admission actions on 200 natural path triples, the eight-policy budget replay, the vote-gate examples and the verified-candidate oracle. For the earlier studies, they also reproduce the strong information baseline, all paired timings for 32 questions on the A800, the concave majorants of three width clocks, the 210 matching worst-case constructions, finite occupancy checks and the fixed-clock bounds on 324 panels. See `numerical-check.json` for exact estimates and `replay.py` for the scripts, inputs and reference tables used by each comparison.

`archive-check.json` records a separate check on the full 753-file private source archive. That run used the original verification program and local LaTeX installation to reproduce all 18 PDF pages with identical extracted text. The 14 numerical calculations above use only the Python dependencies in `requirements.txt`. The archived measurements establish what happened on the original A800. Collecting new timings uses the separate GPU instructions.
