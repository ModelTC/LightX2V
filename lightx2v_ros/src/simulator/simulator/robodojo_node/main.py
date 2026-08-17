import os
import sys
from pathlib import Path


def _extract_ros_param(name, argv=None):
    marker = f"{name}:="
    for token in argv or sys.argv:
        if token.startswith(marker):
            return token[len(marker) :]
    return None


def _as_bool(value, default=False):
    if value is None or value == "":
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _default_robodojo_root():
    return Path(os.environ.get("ROBODOJO_ROOT", "/app/RoboDojo"))


def _resolve_root(path=None):
    raw = str(path).strip() if path is not None else ""
    return Path(os.path.expandvars(raw)).expanduser().resolve() if raw else _default_robodojo_root().resolve()


def _add_robodojo_paths(root):
    paths = [str(root), str(root / "XPolicyLab")]
    for path in paths:
        while path in sys.path:
            sys.path.remove(path)
    sys.path[:0] = paths


def _device_id_from_args(argv):
    raw = _extract_ros_param("device_id", argv) or os.environ.get("ROBODOJO_DEVICE_ID", "0")
    return int(raw)


def _launch_isaac_sim():
    original_argv = sys.argv[:]
    robodojo_root = _resolve_root(_extract_ros_param("robodojo_root", original_argv))
    _add_robodojo_paths(robodojo_root)
    device_id = _device_id_from_args(original_argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    from isaaclab.app import AppLauncher

    required_kit_args = "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
    extra_kit_args = os.environ.get("ROBODOJO_KIT_ARGS", "").strip()
    kit_args = f"{required_kit_args} {extra_kit_args}".strip()

    launcher_args = {
        "enable_cameras": True,
        "device": "cuda:0",
        "headless": _as_bool(_extract_ros_param("headless", original_argv), os.environ.get("HEADLESS", "1") == "1"),
        "livestream": int(os.environ.get("LIVESTREAM", "0")),
        "kit_args": kit_args,
    }

    # Keep ROS arguments away from Kit, then restore them for rclpy.init().
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
        from simulator.robodojo_node.env import build_robodojo_env
        from simulator.sim.node import run_simulator_node

        contract = get_contract("robodojo")

        def _factory(node):
            return build_robodojo_env(node, simulation_app=launcher.app)

        run_simulator_node(contract, _factory, node_name="robodojo_node", args=args)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
