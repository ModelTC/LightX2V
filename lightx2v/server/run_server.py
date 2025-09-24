#!/usr/bin/env python
"""Example script to run the LightX2V server."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lightx2v.server.main import run_server


def main():
    parser = argparse.ArgumentParser(description="Run LightX2V inference server")

    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--model_cls", type=str, required=True, help="Model class name")
    parser.add_argument("--config_json", type=str, help="Path to model config JSON file")
    parser.add_argument("--task", type=str, default="i2v", help="Task type (i2v, etc.)")

    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of processes per node (GPUs to use)")

    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")

    args = parser.parse_args()

    run_server(args)


if __name__ == "__main__":
    main()
