"""Run robotwin evaluation without ROS/colcon; accepts key=value overrides."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "lightx2v_ros/src/simulator"), str(ROOT / "lightx2v_ros/src/common")]


def main():
    from simulator.robotwin_node import benchmark as backend
    from simulator.sim.benchmark import load_config, run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. MULTIRUN.num_gpus=1 dry_run=true")
    args = parser.parse_args()
    run(load_config("robotwin", backend, args.overrides))


if __name__ == "__main__":
    main()
