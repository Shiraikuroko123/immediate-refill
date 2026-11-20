import hashlib
import json
from pathlib import Path
import re
import requests
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
REV="740312add88f781978c0658806c59bc2815b9866"
URL=f"https://huggingface.co/datasets/openai/gsm8k/resolve/{REV}/main/test-00000-of-00001.parquet"

def digest(data):
    return hashlib.sha256(data).hexdigest()

if __name__=="__main__":
    src=HERE/"test.parquet"
    if not src.is_file():
        r=requests.get(URL,timeout=(10,45)); r.raise_for_status()
        src.write_bytes(r.content)
    items=pq.read_table(src).to_pylist()
    assert len(items)==1319
    ranking=sorted((digest(row["question"].encode()),idx) for idx,row in enumerate(items))
    assert len({p[0] for p in ranking})==len(items)
    selected=[]
    for key,idx in ranking[:16]:
        item=items[idx]
        gold=item["answer"].rsplit("####",1)[1].strip().replace(",","")
        assert re.fullmatch(r"[+-]?[0-9]+",gold)
        selected.append(dict(unique_id=key, source_row=idx, problem=item["question"], gold=str(int(gold))))
    obj=dict(revision=REV,url=URL,source_sha256=digest(src.read_bytes()),count=len(items),questions=selected)
    dest=HERE/"selection.json"
    data=(json.dumps(obj,ensure_ascii=False,indent=2)+"\n").encode()
    if dest.exists() and dest.read_bytes()!=data: raise ValueError("Frozen selection differs")
    dest.write_bytes(data)
    print(len(selected),digest(data))
