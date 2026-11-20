"""Fetch the fixed Qwen checkpoint from the official ModelScope repository."""
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import sys


def main():
    root = pathlib.Path(__file__).resolve().parent
    specs = json.loads((root / "model-files.json").read_text(encoding="utf-8"))
    target = pathlib.Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)

    def fetch(spec):
        dest = target / spec["name"]
        url = ("https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct/resolve/"
               + spec["revision"] + "/" + spec["name"])
        part = dest.with_suffix(dest.suffix + ".part")
        if not dest.exists():
            subprocess.run(["curl", "-fsSL", "--retry", "3", "--connect-timeout", "12",
                            "-C", "-", "-o", str(part), url], check=True)
            candidate = part
        else:
            candidate = dest
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(2 ** 20), b""):
                digest.update(block)
        assert digest.hexdigest() == spec["sha256"] and candidate.stat().st_size == spec["size"], spec["name"]
        if candidate == part:
            candidate.replace(dest)
        print("verified", dest.name, spec["size"], flush=True)

    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        list(pool.map(fetch, specs))


if __name__ == "__main__":
    main()
