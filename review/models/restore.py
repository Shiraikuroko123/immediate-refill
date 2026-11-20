"""Restore a frozen checkpoint on any host using Python 3.8+ and HTTPS."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen


MANIFEST = Path(__file__).with_name("manifest.json")


def checked(path, spec):
    if path.stat().st_size != spec["size"]:
        raise ValueError(f"Incorrect byte count for {spec['name']}")
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != spec["sha256"]:
        raise ValueError(f"Incorrect SHA-256 for {spec['name']}")


def urls(model, spec, source):
    repo = model["repository"]
    name = quote(spec["name"])
    if spec["huggingface_path"] and source in ("auto", "huggingface"):
        yield f"https://huggingface.co/{repo}/resolve/{model['huggingface_revision']}/{name}"
    if source in ("auto", "modelscope"):
        yield f"https://www.modelscope.cn/models/{repo}/resolve/{spec['modelscope_revision']}/{name}"


def restore(target, model, spec, source, verify):
    dest = target / spec["name"]
    if dest.exists() or verify:
        checked(dest, spec)
        return
    # ModelScope adds this two-byte object to its Qwen repository.
    # It can be reproduced from the manifest even when accessing only Hugging Face.
    if spec["name"] == "configuration.json" and spec["huggingface_path"] is None:
        value = b"{}"
        if hashlib.sha256(value).hexdigest() != spec["sha256"] or len(value) != spec["size"]:
            raise ValueError("ModelScope configuration record differs")
        with dest.open("xb") as output:
            output.write(value)
        return
    errors = []
    for url in urls(model, spec, source):
        part = None
        try:
            with tempfile.NamedTemporaryFile(dir=target, delete=False) as output:
                part = Path(output.name)
                with urlopen(Request(url, headers={"User-Agent": "checkpoint-restore/1"}), timeout=50) as reply:
                    shutil.copyfileobj(reply, output, length=1024 * 1024)
            checked(part, spec)
            part.replace(dest)
            return
        except (OSError, ValueError) as error:
            errors.append(f"{url}: {type(error).__name__}: {error}")
        finally:
            if part is not None and part.exists():
                part.unlink()
    raise RuntimeError(f"Unable to restore {dest}\n" + "\n".join(errors))


def main(argv):
    models = json.loads(MANIFEST.read_bytes())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=models)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source", choices=("auto", "huggingface", "modelscope"), default="auto")
    parser.add_argument("--verify-only", action="store_true", help="Only validate existing files, offline.")
    parser.add_argument("--metadata-only", action="store_true", help="Skip weight shards and verify small files.")
    args = parser.parse_args(argv)
    model = models[args.model]
    if not args.verify_only:
        args.directory.mkdir(parents=True, exist_ok=True)
    for item in model["files"]:
        if args.metadata_only and item["name"].endswith(".safetensors"):
            continue
        restore(args.directory, model, item, args.source, args.verify_only)
        print(f"{item['sha256']}  {item['name']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
