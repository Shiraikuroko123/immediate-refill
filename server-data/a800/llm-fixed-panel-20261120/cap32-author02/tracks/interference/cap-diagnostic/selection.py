"""Select the cap-intervention questions without reading generated answers."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent


def digest(data):
    return sha256(data).hexdigest()


def select():
    selection = json.loads((STUDY / "gsm_live/selection.json").read_bytes())
    source = STUDY / "gsm_live/test.parquet"
    assert digest(source.read_bytes()) == selection["source_sha256"]
    published = (STUDY / "holdout/overlap/publisher-gsm8k.jsonl").read_bytes().splitlines()
    by_text = {r["problem"]: r["problem_key"] for r in map(json.loads, published)}
    seen_keys = {json.loads(line)["problem_key"] for shard in (STUDY / "holdout/raw").glob("*.jsonl")
                 for line in shard.read_bytes().splitlines()}
    seen_ids = {row["unique_id"] for row in selection["questions"]}
    official = pq.read_table(source).to_pylist()
    assert len(official) == len(by_text) == 1319 and len(seen_keys) == 200 and len(seen_ids) == 16
    candidates = []
    for index, row in enumerate(official):
        text = row["question"]
        uid = digest(text.encode("utf8"))
        if by_text[text] in seen_keys or uid in seen_ids:
            continue
        gold = str(int(row["answer"].rsplit("####", 1)[1].replace(",", "").strip()))
        candidates.append((digest(("cap-transfer-v1/" + text).encode("utf8")),
                           dict(unique_id=uid, source_row=index, problem=text, gold=gold)))
    assert len(candidates) == 1105
    return dict(revision=selection["revision"], source_sha256=selection["source_sha256"], eligible=len(candidates),
                protocol_sha256=digest((HERE / "PROTOCOL.md").read_bytes()),
                questions=[q for _,q in sorted(candidates)[:32]])


if __name__ == "__main__":
    data = (json.dumps(select(), ensure_ascii=False, indent=2) + "\n").encode("utf8")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "selection.json"
    if path.exists():
        assert path.read_bytes() == data
    else:
        with path.open("xb") as f:
            f.write(data)
    print(f"32 selected of 1105, SHA-256 {digest(data)}")
