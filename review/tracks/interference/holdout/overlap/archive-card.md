---
license: cc-by-4.0
---

# rollouts-qwen4b — evaluation rollouts

Model: Qwen/Qwen3-4B-Base. Tokenizer used for answer positions: Qwen/Qwen3-4B-Base.
Protocol: 32 rollouts per problem (two seeded batches of 16), temperature 0.6, top-p 0.95,
budget 31,744 generated tokens, seed 20260819. Prompts and grader: the paper's repository
(sophicle/reason). Rollout jsonl files are kept as written (one graded rollout per line, with the
completion), under the run directories as on disk (`run/`, `run_regraded/`, `*_execgraded/`);
cell summaries are at `<prompt>/<benchmark>/summary/<arm>.json`.
