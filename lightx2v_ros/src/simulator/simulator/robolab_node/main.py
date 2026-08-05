import os
import sys


def _launch_isaac_sim():
    # RoboLab requires cv2 to be imported before isaaclab/Kit.
    import cv2  # noqa: F401
    from isaaclab.app import AppLauncher

    launcher_args = {
        "enable_cameras": True,
        "device": os.environ.get("ROBOLAB_DEVICE", "cuda:0"),
        "headless": os.environ.get("HEADLESS", "1") == "1",
        "livestream": int(os.environ.get("LIVESTREAM", "0")),
    }
    kit_args = os.environ.get("ROBOLAB_KIT_ARGS", "").strip()
    if kit_args:
        launcher_args["kit_args"] = kit_args

    # Keep ROS arguments away from Kit, then restore them for rclpy.init().
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        launcher = AppLauncher(launcher_args)
    finally:
        sys.argv = original_argv
    return launcher


def main(args=None):
    launcher = _launch_isaac_sim()
    try:
        from common.contract import get_contract

        from simulator.robolab_node.env import build_robolab_env
        from simulator.sim.node import run_simulator_node

        run_simulator_node(get_contract("robolab"), build_robolab_env, node_name="robolab_node", args=args)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
