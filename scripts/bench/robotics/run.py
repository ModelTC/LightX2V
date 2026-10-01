"""Repository-local robotics evaluation; does not require ROS/colcon."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "lightx2v_ros/src/simulator"), str(ROOT / "lightx2v_ros/src/common")]


def main():
    from simulator.sim.benchmark import load_config, run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=["libero", "libero_plus", "robotwin"])
    parser.add_argument("overrides", nargs="*", help="key=value overrides; see configs/bench/robotics/eval.yaml")
    args = parser.parse_args()
    run(load_config(args.benchmark, args.overrides))


if __name__ == "__main__":
    main()
