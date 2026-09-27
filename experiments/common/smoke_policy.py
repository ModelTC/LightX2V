"""Synthetic observation -> action smoke. No simulator or robot execution."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import numpy as np

    from experiments.common.config import load_config
    from experiments.common.interfaces import Observation
    from experiments.policies.realtimewam import build_policy

    cfg = load_config(sys.argv[1], sys.argv[2:])
    robotwin = cfg["benchmark"] == "robotwin"
    cameras = ("head_camera", "left_camera", "right_camera") if robotwin else ("agentview", "wrist")
    policy = build_policy(cfg)
    try:
        observation = Observation({name: np.full((256, 256, 3), 127, dtype=np.uint8) for name in cameras}, np.zeros(14 if robotwin else 8, dtype=np.float32), "pick up the object")
        policy.reset_episode({"synthetic": True})
        result = policy.predict_action_chunk(observation)
        actions = result.validate("robotwin_joint_position" if robotwin else "libero_delta_eef", 14 if robotwin else 7)
        print(
            json.dumps(
                {
                    "synthetic_smoke_only": True,
                    "shape": list(actions.shape),
                    "finite": bool(np.isfinite(actions).all()),
                    "min": actions.min(axis=0).tolist(),
                    "max": actions.max(axis=0).tolist(),
                    "timing": result.timing,
                }
            )
        )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
