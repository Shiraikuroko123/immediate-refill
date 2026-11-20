"""Disposable full-parameter training benchmark. No checkpoint is written."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(1729)
    batch, context = 1, 512
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.bfloat16).to("cuda").train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
    tokens = torch.randint(model.config.vocab_size, (batch,context), device="cuda")
    with args.output.open("x", encoding="utf-8", buffering=1) as out:
        for i in range(4):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = model(input_ids=tokens,labels=tokens,use_cache=False).loss
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            row = dict(iteration=i, warmup=i==0, seconds=seconds, input_tokens=context*batch,
                       input_tok_per_sec=batch*context/seconds,
                       device=torch.cuda.get_device_name(),torch=torch.__version__,dtype="BF16",
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),loss=float(loss.detach()))
            out.write(json.dumps(row)+"\n")


if __name__=="__main__":
    main()
