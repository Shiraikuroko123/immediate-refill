# When Is Immediate Refill Optimal for Parallel Voting?

This repository preserves the manuscript, the code and inputs used to reproduce its numerical results, and the research files inventoried on the A800 and H20 hosts on 21 November 2026 (Beijing time).

**Project page:** [open the live research showcase](https://shiraikuroko123.github.io/immediate-refill/) · [browse the static site source](docs/)

| To do this | Start here |
| --- | --- |
| Read the paper or edit its LaTeX | [PDF](review/manuscript/transfer-review/main.pdf) · [TeX and bibliography](review/manuscript/transfer-review/) |
| Recompute the paper's results from recorded experiments | [Reviewer instructions](review/README.md) · [Extended CPU analysis](review/cpu/README.md) |
| Obtain the exact portable supplement | [Download scheduling-review.zip](dist/scheduling-review.zip) |
| Restore the original server project directories | [Server backup](server-data/README.md) · [A800 inventory](server-data/a800/inventory.json) · [H20 inventory](server-data/h20/inventory.json) |
| Collect fresh GPU measurements | [Restore the pinned model files](review/models/README.md) · [GPU runtime and commands](review/runtime/README.md) |

From the root of a clone, run these checks with Python 3.10. The first verifies the 732 archived server files. The second checks the portable artifact and recomputes the primary device results from raw events. Both work offline.

```sh
python -B server-data/verify.py
python -B review/check.py
```

For all 14 groups of numerical results, install the pinned CPU dependencies and run the replay from `review/`.

```sh
cd review
python -m pip install -r cpu/requirements.txt
python -B cpu/replay.py --root . --report ../cpu-results.json
```

The manuscript PDF has SHA-256 `0d70afff5ebfc7eab790c42f9ca572a6c0f1439db787c391574a5f9322a0eeaa`. The supplement ZIP has SHA-256 `fc3ccc26d2f205febc4d87bf4609b97013e36ebd8fc207e7bbfa1b7d8f1f0853`. Data sources, model revisions and applicable notices are documented in [SOURCES.md](review/SOURCES.md).
