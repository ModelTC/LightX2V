from .base_env import BaseSimEnv, Observation


def __getattr__(name):
    # Environment-only consumers (offline evaluation) do not require ROS.
    if name in {"SimulatorNode", "rgb_to_image_msg", "run_simulator_node"}:
        from . import node

        return getattr(node, name)
    raise AttributeError(name)


__all__ = [
    "BaseSimEnv",
    "Observation",
    "SimulatorNode",
    "rgb_to_image_msg",
    "run_simulator_node",
]
