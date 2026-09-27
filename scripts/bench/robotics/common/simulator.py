"""Import repository-local simulator adapters without initializing ROS."""

import sys

from scripts.bench.robotics.common.config import ROOT


def bootstrap_simulator():
    for relative in ("lightx2v_ros/src/common", "lightx2v_ros/src/simulator"):
        path = str(ROOT / relative)
        if path not in sys.path:
            sys.path.insert(0, path)
