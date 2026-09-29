"""Isolated process: its GPU visibility is fixed before importing torch/simulators."""

import json
import sys
from pathlib import Path


def main():
    from simulator.sim.bench.evaluator import run_task
    from simulator.sim.bench.policies.realtimewam import build_policy

    payload = json.loads(Path(sys.argv[1]).read_text())
    policy = build_policy(payload["config"])
    try:
        for task in payload["tasks"]:
            run_task(payload["config"], task, policy)
    finally:
        policy.close()


if __name__ == "__main__":
    main()
