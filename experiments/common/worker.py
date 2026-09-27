"""Isolated process: its GPU visibility is fixed before importing torch/simulators."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    from experiments.common.evaluator import run_task
    from experiments.policies.realtimewam import build_policy

    payload = json.loads(Path(sys.argv[1]).read_text())
    policy = build_policy(payload["config"])
    try:
        for task in payload["tasks"]:
            run_task(payload["config"], task, policy)
    finally:
        policy.close()


if __name__ == "__main__":
    main()
