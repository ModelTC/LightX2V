"""Relocate downloaded planner assets without modifying the benchmark source."""

import importlib
import os
import sys
from pathlib import Path

from simulator.sim.bench.results import atomic_json


def relocate_asset_paths(value, root, config_dir, key=None):
    root, config_dir = Path(root).resolve(), Path(config_dir).resolve()
    if isinstance(value, dict):
        return {k: relocate_asset_paths(v, root, config_dir, k) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate_asset_paths(v, root, config_dir, key) for v in value]
    if not isinstance(value, str) or not value:
        return value
    path_keys = {"urdf_path", "collision_spheres", "asset_root_path", "usd_path", "isaac_usd_path"}
    if "/assets/" in value:
        target = root / "assets" / value.split("/assets/", 1)[1]
    elif value.startswith("assets/"):
        target = root / value
    elif key in path_keys:
        target = Path(value)
        if not target.is_absolute():
            target = config_dir / target
    else:
        return value
    target = target.resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Planner asset must be inside the selected RoboTwin repository: {value}")
    if not target.exists():
        raise FileNotFoundError(f"Missing repository-local planner asset: {target}")
    return str(target)


def install_planner_adapter(root, output):
    """Patch only the benchmark's planner constructor in this isolated worker."""
    planner = importlib.import_module("envs.robot.planner")
    original = getattr(planner, "CuroboPlanner", None)
    if original is None:
        raise ImportError("RoboTwin failed to import its real CuroboPlanner")
    if getattr(original, "_lightx2v_relocated", False):
        return

    class RepositoryCuroboPlanner(original):
        _lightx2v_relocated = True

        def __init__(self, robot_origion_pose, active_joints_name, all_joints, yml_path=None):
            import yaml

            source = Path(yml_path).resolve()
            config = relocate_asset_paths(yaml.safe_load(source.read_text()), root, source.parent)
            # Preserve the embodiment name: upstream tests 'aloha-agilex' in this path.
            target = Path(output) / "runtime/robotwin_planner" / str(os.getpid()) / source.parent.name / source.name
            atomic_json(target, config)  # JSON is valid YAML for both planner loaders.
            super().__init__(robot_origion_pose, active_joints_name, all_joints, yml_path=str(target))

    planner.CuroboPlanner = RepositoryCuroboPlanner
    # envs.robot may already import the class by value while loading its package.
    robot = sys.modules.get("envs.robot.robot")
    if robot and getattr(robot, "CuroboPlanner", None) is original:
        robot.CuroboPlanner = RepositoryCuroboPlanner
