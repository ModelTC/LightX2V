"""Keep the task PID alive until resource capture has a post-workload sample."""

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--initialize-cuda", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    sys.argv = command
    if args.initialize_cuda:
        import torch

        if torch.cuda.device_count() != 1 or not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError("Early CUDA initialization requires one explicitly visible GPU")
        # Use this worker's normal primary context before CPU model preparation.
        # RNG setup, model loading, warmup and request timing remain downstream.
        torch.cuda.init()
        early_context = torch.empty(1, dtype=torch.uint8, device="cuda")
    try:
        runpy.run_path(command[0], run_name="__main__")
    except SystemExit as error:
        if error.code not in (None, 0):
            raise
    # The measured workload and its files are complete. Do not exit (and lose
    # /proc lineage) until the controller observes the far side of this boundary.
    fields = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
    marker = {"pid": os.getpid(), "start_ticks": int(fields[19]), "end": time.monotonic()}
    temporary = args.completion.with_suffix(".tmp")
    temporary.write_text(json.dumps(marker) + "\n")
    temporary.replace(args.completion)
    deadline = time.monotonic() + 20
    while not args.release.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("Controller did not acknowledge the post-workload resource sample")
        time.sleep(0.05)
    if args.initialize_cuda:
        del early_context


if __name__ == "__main__":
    main()
