"""Start an isolated local vLLM service, measure four policies, and stop it."""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18274)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--controlled-profile", type=Path, help="Replay fixed path lengths and archived votes")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    url = f"http://127.0.0.1:{args.port}/v1"
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    import importlib.metadata
    label = ", ".join(f"{pkg} {importlib.metadata.version(pkg)}"
                      for pkg in ("torch", "vllm", "transformers"))
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(args.model.resolve()),
           "--served-model-name", "qwen-study", "--host", "127.0.0.1", "--port", str(args.port),
           "--dtype", "bfloat16", "--max-model-len", "1024", "--max-num-seqs", "8",
           "--gpu-memory-utilization", "0.35", "--enforce-eager", "--disable-log-requests"]
    env = dict(os.environ, VLLM_USE_V1="0", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    start = time.time()
    (args.out / "server_config.json").write_text(json.dumps(dict(start=start, argv=cmd, versions=label, cache="v0 default off"),indent=2))
    with (args.out / "server.log").open("w") as log:
        server = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(125):
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited with {server.returncode}")
                try:
                    with urllib.request.urlopen(url + "/models", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    time.sleep(2)
            else:
                raise RuntimeError("Endpoint did not become ready in time")
            with (args.out / "gpu-ready.csv").open("w") as meta:
                subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,memory.used,driver_version","--format=csv"],
                               stdout=meta, check=True)
            with (args.out / "client.log").open("w") as client_log:
                common = ["--base-url",url,"--model","qwen-study","--out",str(args.out/"client"),
                          "--repeats",str(args.repeats)]
                if args.controlled_profile:
                    client = [str(root/"controlled_http.py"), "--profile",str(args.controlled_profile)] + common
                else:
                    client = [str(root/"run.py"), "--fixed", "--server-config",
                              label + ", V0/eager/BF16, no prefix caching"] + common
                subprocess.run([sys.executable, *client],
                               env=env, stdout=client_log, stderr=subprocess.STDOUT, check=True, timeout=1500)
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            (args.out/"elapsed.json").write_text(json.dumps({"elapsed_sec":time.time()-start,"server_rc":server.returncode}))


if __name__ == "__main__":
    main()
