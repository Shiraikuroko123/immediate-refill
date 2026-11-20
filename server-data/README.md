# GPU experiment snapshots

Each machine directory contains a copy of its original `llm-fixed-panel-20261120` experiment directory, including run logs, historical scripts, and raw measurements. `inventory.json` preserves the inventory collected on that machine with the scan time in UTC, relative paths, sizes, SHA-256 hashes, and all skipped directories.

| Machine | Files | Bytes | Inventory |
| --- | ---: | ---: | --- |
| A800 | 536 | 44,814,470 | [a800/inventory.json](a800/inventory.json) |
| H20 | 196 | 8,667,647 | [h20/inventory.json](h20/inventory.json) |

From the repository root, run `python -B server-data/verify.py`. It uses only the Python standard library and checks both inventories, all 732 file contents, and the directory listings. To restore either machine's experiment files, copy that machine's `llm-fixed-panel-20261120` directory to the destination. Keep the two copies separate because the machines sometimes used identical paths for different measurements.

The inventories cover the experiment directories at the recorded scan times. Public model weights, software environments, and caches were excluded during the original collection. See the reviewer package for model revisions, weight restoration, and commands for reproducing the paper.
