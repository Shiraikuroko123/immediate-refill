"""Verify both GPU snapshots against the inventories captured on the servers."""

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_HASH = {
    "a800": "2741fcf771a6ee132f4a6caeacffd667b9672b29369c2ebc375ff53bee8bab25",
    "h20": "4be4b48d87550e34e5d60e53365798001bbeedce75a76d46801466d5c54f47e9",
}


def verify(machine, expected_inventory_hash):
    directory = ROOT / machine
    inventory = (directory / "inventory.json").read_bytes()
    if hashlib.sha256(inventory).hexdigest() != expected_inventory_hash:
        raise ValueError(f"{machine}/inventory.json differs from the server inventory")

    records = json.loads(inventory)["files"]
    recorded_paths = {record["relative"] for record in records}
    present_paths = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    } - {"inventory.json"}
    if present_paths != recorded_paths:
        missing = sorted(recorded_paths - present_paths)
        extra = sorted(present_paths - recorded_paths)
        raise ValueError(f"{machine}: missing files {missing}, unexpected files {extra}")

    for record in records:
        path = directory / record["relative"]
        content = path.read_bytes()
        if len(content) != record["size"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"{machine}/{record['relative']} differs from the server inventory")

    size = sum(record["size"] for record in records)
    print(f"{machine}: {len(records):,} files, {size:,} bytes verified")
    return len(records), size


if __name__ == "__main__":
    try:
        counts = [verify(machine, digest) for machine, digest in MANIFEST_HASH.items()]
    except (OSError, ValueError) as error:
        sys.exit(str(error))
    print(f"Total: {sum(n for n, _ in counts):,} files, {sum(size for _, size in counts):,} bytes verified")
