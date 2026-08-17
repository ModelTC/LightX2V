import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


class _Msg:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None, frame_id="")
        self.data = None


def _stub_modules():
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *args, **kwargs: None
    rclpy.spin = lambda *args, **kwargs: None
    rclpy.ok = lambda: False
    rclpy.shutdown = lambda: None

    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = type("Node", (), {})

    common_contract = types.ModuleType("common.contract")
    common_contract.EnvContract = type("EnvContract", (), {})

    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.Image = type("Image", (_Msg,), {})

    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Bool = type("Bool", (_Msg,), {})
    std_msgs_msg.Float32MultiArray = type("Float32MultiArray", (_Msg,), {})
    std_msgs_msg.Int32 = type("Int32", (_Msg,), {})
    std_msgs_msg.String = type("String", (_Msg,), {})

    base_env = types.ModuleType("simulator.sim.base_env")
    base_env.BaseSimEnv = type("BaseSimEnv", (), {})

    return {
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "common": _package("common"),
        "common.contract": common_contract,
        "sensor_msgs": _package("sensor_msgs"),
        "sensor_msgs.msg": sensor_msgs_msg,
        "std_msgs": _package("std_msgs"),
        "std_msgs.msg": std_msgs_msg,
        "simulator": _package("simulator"),
        "simulator.sim": _package("simulator.sim"),
        "simulator.sim.base_env": base_env,
    }


def _load_node_module():
    module_path = Path(__file__).resolve().parents[1] / "lightx2v_ros/src/simulator/simulator/sim/node.py"
    spec = importlib.util.spec_from_file_location("simulator.sim.node", module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, _stub_modules()):
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


class _Logger:
    def __init__(self):
        self.errors = []
        self.infos = []

    def error(self, message):
        self.errors.append(message)

    def info(self, message):
        self.infos.append(message)


class _Env:
    accepted_action_dims = (2,)
    task_name = "stack_bowls"
    task_config = "arx_x5/layout_0"
    seed = 0
    supports_task_switch = False

    def __init__(self, *, success=False, done=False):
        self.success = success
        self.done = done
        self.actions = []

    @property
    def task_description(self):
        return "stack the bowls"

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        return SimpleNamespace(images={}, state=np.zeros(2, dtype=np.float32)), self.success, self.done

    def list_tasks(self):
        return []

    def list_task_configs(self):
        return []


class SimulatorStatusUpdateTest(unittest.TestCase):
    def _make_node(self, module, *, success=False, done=False, max_episode_steps=0):
        node = object.__new__(module.SimulatorNode)
        node.contract = SimpleNamespace(
            name="robodojo",
            cameras=("head_camera", "left_camera", "right_camera"),
            policy_input_cameras=("head_camera", "left_camera", "right_camera"),
        )
        node.env = _Env(success=success, done=done)
        node.state = module.RUNNING
        node.step_index = 0
        node.episode_step = 0
        node.episode_index = 1
        node.success = False
        node.max_episode_steps = max_episode_steps
        node.loop = False
        node.history = []
        node._in_env_step = False
        node.logger = _Logger()
        node.observation_count = 0
        node.status_snapshots = []

        def publish_observation():
            node.observation_count += 1

        def publish_status():
            node.status_snapshots.append(node.build_status())

        node.publish_observation = publish_observation
        node.publish_status = publish_status
        node.get_logger = lambda: node.logger
        return node

    def test_running_action_publishes_status_with_latest_episode_step(self):
        module = _load_node_module()
        node = self._make_node(module)

        module.SimulatorNode.on_action(node, SimpleNamespace(data=[0.25, -0.5]))

        self.assertEqual(node.episode_step, 1)
        self.assertEqual(node.step_index, 1)
        self.assertEqual(node.state, module.RUNNING)
        self.assertEqual(node.observation_count, 1)
        self.assertEqual(len(node.status_snapshots), 1)
        self.assertEqual(node.status_snapshots[-1]["episode_step"], 1)
        self.assertEqual(node.status_snapshots[-1]["state"], module.RUNNING)

    def test_step_cap_finish_path_publishes_single_failure_status(self):
        module = _load_node_module()
        node = self._make_node(module, max_episode_steps=1)

        module.SimulatorNode.on_action(node, SimpleNamespace(data=[0.0, 0.0]))

        self.assertEqual(node.episode_step, 1)
        self.assertEqual(node.state, module.FAILURE)
        self.assertEqual(node.observation_count, 1)
        self.assertEqual(len(node.status_snapshots), 1)
        self.assertEqual(node.status_snapshots[-1]["state"], module.FAILURE)
        self.assertEqual(node.status_snapshots[-1]["episode_step"], 1)
        self.assertEqual(node.history[-1]["outcome"], "failure")
        self.assertEqual(node.history[-1]["steps"], 1)

    def test_success_finish_path_publishes_single_success_status(self):
        module = _load_node_module()
        node = self._make_node(module, success=True)

        module.SimulatorNode.on_action(node, SimpleNamespace(data=[1.0, 1.0]))

        self.assertEqual(node.episode_step, 1)
        self.assertEqual(node.state, module.SUCCESS)
        self.assertEqual(node.observation_count, 1)
        self.assertEqual(len(node.status_snapshots), 1)
        self.assertEqual(node.status_snapshots[-1]["state"], module.SUCCESS)
        self.assertEqual(node.status_snapshots[-1]["episode_step"], 1)
        self.assertEqual(node.history[-1]["outcome"], "success")


if __name__ == "__main__":
    unittest.main()
