"""Run the GSM8K pilot using the original audited quota driver."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"tracks/quota"))
import live

def integer_box(text):
    value=live.run_gpu.boxed(text)
    if value is None: return None
    if not re.fullmatch(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)", value): return None
    return str(int(value.replace(",","")))

def order(qid, selected):
    seq=("fixed-exact","fixed-buffer","natural-exact","natural-buffer")
    permutations=((0,1,2,3),(2,3,1,0),(1,0,3,2),(3,2,0,1))
    return [seq[i] for i in permutations[qid%4] if seq[i] in selected]

orig_drive=live.drive
def drive(engine,tokenizer,sample,delta,cuda,q,qid,prompt_ids,condition,policy,reference,ref_sha,config_hash,position,out):
    namespace="gsm-reference-20261120" if condition=="fixed" or policy=="full" else "gsm-natural-20261120"
    def seed(uid,index):
        return int.from_bytes(hashlib.sha256(f"{namespace}/{uid}/{index}".encode()).digest()[:4],"big")%(2**31-1)
    live.seed=seed
    row=orig_drive(engine,tokenizer,sample,delta,cuda,q,qid,prompt_ids,condition,policy,reference,ref_sha,config_hash,position,out)
    row["seed_namespace"]=namespace
    return row

if __name__=="__main__":
    live.COMPLETION_CAP=384
    live.TEMPERATURE=.7
    live.TOP_P=.95
    live.PROMPT="\nSolve the problem. Show your reasoning briefly and put your final integer answer in \\boxed{}."
    live.boxed=integer_box
    live.jobs_for=order
    live.drive=drive
    args=live.cli().parse_args()
    args.model_identity="Qwen/Qwen2.5-1.5B-Instruct"
    args.jobs="fixed-exact,fixed-buffer,natural-exact,natural-buffer"
    args.max_model_len=2048
    frozen=json.loads((HERE/"freeze.json").read_text())
    for name,expected in frozen.items():
        if live.digest((ROOT/name).read_bytes())!=expected:
            raise ValueError(f"Frozen file changed: {name}")
    args.out.mkdir(exist_ok=True,parents=True)
    meta=dict(freeze=frozen, natural_seed="gsm-natural-20261120", reference_seed="gsm-reference-20261120",
              note="Each panel's seed_namespace gives the actual seed. The legacy driver's generic namespace in its config is superseded here.")
    path=args.out/"driver.json"
    if path.exists() and json.loads(path.read_bytes())!=meta: raise ValueError("Driver metadata changed")
    if not path.exists(): live.store(path,meta)
    live.run(args)
