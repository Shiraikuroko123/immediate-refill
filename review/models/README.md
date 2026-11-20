# Restoring the published model files

The prospective A800 trial used Qwen2.5-1.5B-Instruct. The earlier width calibration used DeepSeek-R1-Distill-Qwen-1.5B. The restore script uses Python's standard library and the SHA-256 values recorded in the research archive. Run these commands from the directory containing this file. Choose checkpoint directories on a disk with at least 3.2 GB for Qwen and 3.6 GB for DeepSeek.

```sh
python restore.py qwen ./checkpoints/qwen
python restore.py deepseek ./checkpoints/deepseek
python restore.py qwen ./checkpoints/qwen --verify-only
python restore.py deepseek ./checkpoints/deepseek --verify-only
```

The first two commands download any missing files and check every file's size and hash. The final two commands run offline. Hugging Face is the first source and ModelScope is the fallback. `--source modelscope` selects ModelScope directly. Each source is pinned to an immutable revision. Each accepted file must match the corresponding original digest. One auxiliary Qwen file, `configuration.json`, belongs to ModelScope and consists of the two bytes `{}`. The script reconstructs those bytes and verifies the digest. Use the Qwen directory as `DEVICE_MODEL` for the prospective trial. The earlier calibration driver accepts the DeepSeek directory via `--model`.

| Checkpoint | Hugging Face revision | Weight bytes |
|---|---|---:|
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 3,087,467,144 |
| [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/tree/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562) | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` | 3,554,214,621 |

The portability check restored all 11 configuration and tokenizer files in fresh directories. Offline verification matched all 11 original hashes. For the two weight files it checked the original SHA-256 against the Hugging Face Git LFS object ID and the ModelScope metadata, and requested an initial byte range from the pinned releases. This check did not transfer both entire weights or execute an inference run. A reviewer executing the commands above verifies the full weights locally. [File manifest](manifest.json) · [Small-file verification](small-files-verified.json) · [Public registry and byte-range observations](source-check.json)
