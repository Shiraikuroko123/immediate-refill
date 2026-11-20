# Reproducing *When Is Immediate Refill Optimal for Parallel Voting?*

Unzip the supplement and run the following command from its top-level directory using Python 3.10 or 3.11. This first check uses the Python standard library. The historical source instance can be offline. A reviewer needs only the ZIP.

```sh
python -B check.py
```

The script checks the SHA-256 of every supplied file. From the original sealed prediction and per-step device events it reconstructs the predicted 3.13% saving and the observed 2.46% median saving on the three-request panel. It also recomputes the equal-length control, the two precedence examples, the finite verification-candidate checks, and the complete 200-question, two-action opportunity table. The oracle that knows all three final lengths saves at most 0.364% on average in that table. All computation uses temporary output files.

For the extended CPU analysis, use the packages listed in [the CPU instructions](cpu/README.md), then run

```sh
python -B cpu/replay.py --root . --report cpu-results.json
```

This path recomputes fourteen groups of results, including the three contrasts at 8.71%, 0.67% and 0.30%, the 32-question two-cap comparison, and the worst-case certificates derived from the archived A800 clocks. Two checks reproduce the fixed-panel and timing counterexamples and the information ablation with its bootstrap intervals. Each analysis runs with network access via Python sockets disabled. The [reference outputs](cpu/numerical-check.json) identify the quantities compared. The included [paper](manuscript/transfer-review/main.pdf) has its LaTeX sources alongside it. To recompile, run `pdflatex main`, `bibtex main`, `pdflatex main`, and `pdflatex main` in that directory.

To collect new measurements for the prospective three-request experiment on a Linux A800-SXM4-80GB machine, download the pinned Qwen model and run the following commands from the extracted supplement. Choose a destination with enough space for the model and the Docker image. Use an idle GPU and a new output directory for each attempt.

```sh
python models/restore.py qwen ../qwen-fixed
bash runtime/run.sh "$PWD" "$(realpath ../qwen-fixed)" "$PWD/../new-a800-attempt"
```

The first command downloads and checks the model and tokenizer files against the reference fingerprints. The second uses the pinned vLLM image, records the new device and software environment, calibrates the clock, seals the new prediction, starts a new engine process for validation, and saves the new raw events. The original device used Python 3.10.12. The public image uses Python 3.12. Hardware clocks, drivers and competing workloads can change wall times. Use the frozen decision rule and report all twelve paired blocks. A fresh measurement can disagree with the original. Both model releases and runtime details are documented. [Model revisions and checks](models/README.md) · [Linux and GPU requirements](runtime/README.md)

The earlier three-context calibration used DeepSeek-R1-Distill-Qwen-1.5B. Its public weights can be restored with `python models/restore.py deepseek ../deepseek-fixed`. In the prepared vLLM 0.8.5 Linux environment use the following original calibration settings. The separate 32-question Qwen output-cap study has a frozen driver and input selection.

```sh
VLLM_USE_V1=0 HF_HUB_OFFLINE=1 python -B tracks/sharing/hardware/calibrate_vllm.py --model ../deepseek-fixed --out ../new-clock --contexts 512 2048 3968 --widths 1 2 3 4 --steps 96 --prelude 8 --blocks 8 --max-model-len 8192 --temperature 0.6 --top-p 0.95
VLLM_USE_V1=0 HF_HUB_OFFLINE=1 python -B tracks/interference/cap-diagnostic/live.py --model ../qwen-fixed --out ../new-cap-data
```

The model download, package installation and container pull initially need a network connection. Archived numerical checks can run offline. External datasets and models retain their respective terms. [Sources and attribution](SOURCES.md)
