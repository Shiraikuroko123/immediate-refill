"""Execute an archived Python analysis and record which input files it opens."""
import atexit
import json
import os
from pathlib import Path
import runpy
import sys


root = Path(sys.argv[1]).resolve()
trace = Path(sys.argv[2])
script = Path(sys.argv[3]).resolve()
seen = set()
repo_cwd = os.path.normcase(str(root)) + os.sep


def observe(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("CPU replay attempted to contact a network service.")
    if event == "open" and isinstance(args[0], str):
        name = os.path.abspath(args[0])
        if os.path.normcase(name).startswith(repo_cwd) and os.path.isfile(name):
            seen.add(Path(name).relative_to(root).as_posix())


atexit.register(lambda: trace.write_text(json.dumps(sorted(seen)), encoding="utf8"))
sys.addaudithook(observe)
sys.path[0] = str(script.parent)
sys.argv = [str(script), *sys.argv[4:]]
runpy.run_path(str(script), run_name="__main__")
