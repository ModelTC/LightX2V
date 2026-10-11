"""Run robotwin evaluation without ROS/colcon; accepts key=value overrides."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "lightx2v_ros/src/simulator"), str(ROOT / "lightx2v_ros/src/common")]


if __name__ == "__main__":
    from simulator.robotwin_node.benchmark import RoboTwinBenchmark

    RoboTwinBenchmark.main()
