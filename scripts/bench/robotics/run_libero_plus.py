import sys
from pathlib import Path

# Reuse the ROS-free simulator package without requiring a colcon installation.
_root = Path(__file__).resolve().parents[3]
_script_dir = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) != _script_dir]
sys.path[:0] = [str(_root), str(_root / "lightx2v_ros/src/simulator"), str(_root / "lightx2v_ros/src/common")]
from simulator.sim.bench.manager import main  # noqa: E402

if __name__ == "__main__":
    main("libero_plus")
