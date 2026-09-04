"""Offline contract tests for the OpenPI ROS bridge.

Run after sourcing the ROS underlay and LightX2V overlay.  The tests exercise
message conversion and callback ordering without starting DDS participants or
loading model weights.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import types
import unittest
from collections import deque
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROS_SOURCE_ROOT = PROJECT_ROOT / "lightx2v_ros/src"
for package in ("common", "inference", "simulator"):
    sys.path.insert(0, str(ROS_SOURCE_ROOT / package))


def _stub_lightx2v_imports():
    """Keep helper tests independent of LightX2V CUDA initialization."""
    modules = {
        "lightx2v": PROJECT_ROOT / "lightx2v",
        "lightx2v.models": PROJECT_ROOT / "lightx2v/models",
        "lightx2v.models.networks": PROJECT_ROOT / "lightx2v/models/networks",
        "lightx2v.models.networks.openpi": PROJECT_ROOT / "lightx2v/models/networks/openpi",
        "lightx2v.models.runners": PROJECT_ROOT / "lightx2v/models/runners",
        "lightx2v.models.runners.openpi": PROJECT_ROOT / "lightx2v/models/runners/openpi",
        "lightx2v.utils": PROJECT_ROOT / "lightx2v/utils",
    }
    for name, path in modules.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    runner = types.ModuleType("lightx2v.models.runners.openpi.openpi_runner")
    runner.OpenPIPolicy = object
    sys.modules[runner.__name__] = runner

    config = types.ModuleType("lightx2v.utils.set_config")
    config.auto_calc_config = lambda values: values
    config.get_default_config = dict
    sys.modules[config.__name__] = config


try:
    from builtin_interfaces.msg import Time
    from common.contract import LIBERO_CONTRACT
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float64MultiArray, MultiArrayDimension, String

    _stub_lightx2v_imports()
    from inference.openpi_node.main import OpenPINode, image_msg_to_rgb, observation_identity, state_identity
    from simulator.libero_node.env import LiberoEnv, quat_to_axis_angle
    from simulator.sim.node import RUNNING, SimulatorNode, parse_action_identity, rgb_to_image_msg

    from lightx2v.models.networks.openpi.infer.post_infer import OpenPIPostInfer
    from lightx2v.models.networks.openpi.infer.pre_infer import _resize_with_pad
except ImportError as exc:  # pragma: no cover - depends on sourced ROS setup
    raise unittest.SkipTest(f"source the ROS underlay before running this test: {exc}") from exc


class _Logger:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, message):
        self.errors.append(str(message))

    def info(self, _message):
        return None

    def warning(self, message):
        self.warnings.append(str(message))


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _ChunkPolicy:
    output_action_dim = 7

    def __init__(self):
        self.calls = []

    def predict_action_chunk(self, *, images, state, task_description):
        call_index = len(self.calls)
        self.calls.append(
            {
                "images": {name: image.copy() for name, image in images.items()},
                "state": state.copy(),
                "task_description": task_description,
            }
        )
        return np.arange(70, dtype=np.float64).reshape(10, 7) + call_index * 1000.0


def _policy_harness():
    node = types.SimpleNamespace(
        contract=LIBERO_CONTRACT,
        numeric_message_type=Float64MultiArray,
        numeric_dtype=np.float64,
        images={camera: None for camera in LIBERO_CONTRACT.policy_input_cameras},
        state=None,
        task_description="",
        episode_index=None,
        plan_epoch=0,
        pending_observation=None,
        last_processed_observation=None,
        pending_actions=deque(),
        actions_per_plan=5,
        policy=_ChunkPolicy(),
        action_pub=_Publisher(),
        get_logger=lambda: _Logger(),
    )
    node._try_process_observation = types.MethodType(OpenPINode._try_process_observation, node)
    node._publish_action = types.MethodType(OpenPINode._publish_action, node)
    return node


def _context(episode, observation, prompt="pick up the bowl", plan_epoch=0):
    message = String()
    message.data = json.dumps(
        {
            "episode": episode,
            "observation": observation,
            "plan_epoch": plan_epoch,
            "task_description": prompt,
        }
    )
    return message


def _state(episode, observation, values=None):
    message = Float64MultiArray()
    message.layout.dim = [MultiArrayDimension(label=f"episode={episode};observation={observation}", size=8, stride=8)]
    message.data = (np.arange(8, dtype=np.float64) if values is None else np.asarray(values)).tolist()
    return message


def _image(camera, episode, observation, value=0):
    pixels = np.full((4, 5, 3), value, dtype=np.uint8)
    return rgb_to_image_msg(pixels, Time(), camera, episode, observation)


def _deliver_observation(node, episode, observation, plan_epoch=0):
    OpenPINode._on_context(node, _context(episode, observation, plan_epoch=plan_epoch))
    OpenPINode._image_callback(node, "wrist")(_image("wrist", episode, observation, 29))
    OpenPINode._on_state(node, _state(episode, observation))
    OpenPINode._image_callback(node, "agentview")(_image("agentview", episode, observation, 11))


class RosMessageContractTest(unittest.TestCase):
    def test_tagged_rgb_round_trip(self):
        image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)[:, ::-1]
        message = rgb_to_image_msg(image, Time(), "agentview", 13, 21)

        self.assertEqual(message.header.frame_id, "agentview|13|21")
        self.assertEqual(observation_identity(message.header.frame_id), (13, 21))
        np.testing.assert_array_equal(image_msg_to_rgb(message), image)

    def test_bgr_message_with_row_padding(self):
        rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        row_bytes = 3 * 3 + 4
        encoded = np.full((2, row_bytes), 255, dtype=np.uint8)
        encoded[:, :9] = rgb[:, :, ::-1].reshape(2, 9)
        message = Image(height=2, width=3, encoding="bgr8", step=row_bytes, data=encoded.tobytes())

        np.testing.assert_array_equal(image_msg_to_rgb(message), rgb)

    def test_identity_parser_rejects_untagged_and_malformed_frames(self):
        self.assertIsNone(observation_identity("agentview"))
        self.assertIsNone(observation_identity("agentview|episode|2"))
        self.assertIsNone(observation_identity("agentview|1"))
        self.assertEqual(state_identity("episode=5;observation=91"), (5, 91))
        self.assertIsNone(state_identity("episode=5"))
        self.assertIsNone(state_identity("episode=5;observation=bad"))


class OpenPINodeSynchronizationTest(unittest.TestCase):
    def test_waits_for_one_matching_set_and_ignores_duplicate_delivery(self):
        node = _policy_harness()
        OpenPINode._on_context(node, _context(3, 8))
        OpenPINode._on_state(node, _state(3, 7))
        OpenPINode._image_callback(node, "agentview")(_image("agentview", 3, 8))
        OpenPINode._image_callback(node, "wrist")(_image("wrist", 3, 8))
        self.assertEqual(len(node.action_pub.messages), 0)

        OpenPINode._on_state(node, _state(3, 8))
        self.assertEqual(len(node.policy.calls), 1)
        self.assertEqual(len(node.action_pub.messages), 1)

        OpenPINode._on_context(node, _context(3, 8))
        OpenPINode._on_state(node, _state(3, 8))
        self.assertEqual(len(node.policy.calls), 1)
        self.assertEqual(len(node.action_pub.messages), 1)

    def test_executes_five_actions_from_each_ten_action_chunk(self):
        node = _policy_harness()
        for observation in range(6):
            _deliver_observation(node, 0, observation)

        self.assertEqual(len(node.policy.calls), 2)
        published = np.asarray([message.data for message in node.action_pub.messages])
        first_chunk = np.arange(70, dtype=np.float64).reshape(10, 7)
        expected = np.concatenate([first_chunk[:5], first_chunk[:1] + 1000.0])
        np.testing.assert_array_equal(published, expected)

    def test_new_episode_discards_the_old_action_tail_without_resetting_policy(self):
        node = _policy_harness()
        _deliver_observation(node, 0, 0)
        self.assertEqual(len(node.pending_actions), 4)

        _deliver_observation(node, 1, 1)
        self.assertEqual(len(node.policy.calls), 2)
        self.assertEqual(len(node.pending_actions), 4)
        np.testing.assert_array_equal(node.action_pub.messages[-1].data, np.arange(7) + 1000.0)

    def test_new_plan_epoch_discards_actions_queued_before_resume(self):
        node = _policy_harness()
        _deliver_observation(node, 0, 0, plan_epoch=0)
        self.assertEqual(len(node.pending_actions), 4)

        _deliver_observation(node, 0, 1, plan_epoch=1)
        self.assertEqual(len(node.policy.calls), 2)
        self.assertEqual(len(node.pending_actions), 4)
        np.testing.assert_array_equal(node.action_pub.messages[-1].data, np.arange(7) + 1000.0)

    def test_float64_state_and_action_are_not_quantized(self):
        node = _policy_harness()
        values = np.array(
            [math.pi, np.nextafter(1.0, 2.0), 0.1234567890123, -0.987654321098, 0.1, -0.2, 0.3, -0.4],
            dtype=np.float64,
        )
        OpenPINode._on_state(node, _state(5, 9, values))
        self.assertEqual(node.state[2].dtype, np.float64)
        np.testing.assert_array_equal(node.state[2], values)

        OpenPINode._publish_action(node, values[:7], 5, 9)
        published = np.asarray(node.action_pub.messages[-1].data, dtype=np.float64)
        np.testing.assert_array_equal(published, values[:7])
        self.assertEqual(parse_action_identity(node.action_pub.messages[-1].layout.dim[0].label), (5, 9, 0))


class OfficialLiberoParityTest(unittest.TestCase):
    def test_state_matches_official_quaternion_conversion_in_float64(self):
        quaternion = np.array([0.11, -0.23, 0.37, 0.88], dtype=np.float64)
        denominator = np.sqrt(1.0 - quaternion[3] * quaternion[3])
        expected_axis_angle = (quaternion[:3] * 2.0 * math.acos(quaternion[3])) / denominator
        np.testing.assert_array_equal(quat_to_axis_angle(quaternion), expected_axis_angle)

        observation = {
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float64),
            "robot0_eef_quat": quaternion,
            "robot0_gripper_qpos": np.array([0.04, -0.04], dtype=np.float64),
        }
        state = LiberoEnv._state(object(), observation)
        expected_state = np.concatenate([observation["robot0_eef_pos"], expected_axis_angle, observation["robot0_gripper_qpos"]])
        self.assertEqual(state.dtype, np.float64)
        np.testing.assert_array_equal(state, expected_state)

    def test_simulator_consumes_float64_action_without_cast(self):
        action = np.array(
            [math.pi, np.nextafter(1.0, 2.0), 0.1234567890123, -0.987654321098, 0.1, -0.2, 0.3],
            dtype=np.float64,
        )

        class Env:
            accepted_action_dims = (7,)

            def step(self, received):
                self.received = received.copy()
                return object(), False, False

        env = Env()
        node = types.SimpleNamespace(
            state=RUNNING,
            numeric_dtype=np.float64,
            env=env,
            _in_env_step=False,
            step_index=4,
            episode_step=2,
            max_episode_steps=220,
            success=False,
            get_logger=lambda: _Logger(),
            publish_observation=lambda: None,
            publish_status=lambda: None,
            _finish_episode=lambda _outcome: None,
        )
        message = Float64MultiArray(data=action.tolist())
        SimulatorNode.on_action(node, message)

        self.assertEqual(env.received.dtype, np.float64)
        np.testing.assert_array_equal(env.received, action)

    def test_simulator_rejects_action_from_an_old_plan_epoch(self):
        class Env:
            accepted_action_dims = (7,)

            def step(self, _received):
                raise AssertionError("stale action must not reach the environment")

        logger = _Logger()
        node = types.SimpleNamespace(
            state=RUNNING,
            numeric_dtype=np.float64,
            env=Env(),
            episode_index=2,
            step_index=9,
            plan_epoch=1,
            get_logger=lambda: logger,
        )
        message = Float64MultiArray(data=[0.0] * 7)
        message.layout.dim = [MultiArrayDimension(label="episode=2;observation=9;plan_epoch=0", size=7, stride=7)]

        SimulatorNode.on_action(node, message)

        self.assertEqual(len(logger.warnings), 1)

    def test_simulator_rejects_malformed_tagged_action(self):
        class Env:
            accepted_action_dims = (7,)

            def step(self, _received):
                raise AssertionError("malformed action must not reach the environment")

        logger = _Logger()
        node = types.SimpleNamespace(
            state=RUNNING,
            numeric_dtype=np.float64,
            env=Env(),
            episode_index=2,
            step_index=9,
            plan_epoch=1,
            get_logger=lambda: logger,
        )
        message = Float64MultiArray(data=[0.0] * 7)
        message.layout.dim = [MultiArrayDimension(label="episode=2;observation=9", size=7, stride=7)]

        SimulatorNode.on_action(node, message)

        self.assertEqual(len(logger.warnings), 1)

    def test_official_image_orientation_survives_ros_round_trip(self):
        agentview = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)
        wrist = np.bitwise_xor(agentview, np.uint8(255))
        env = object.__new__(LiberoEnv)
        env.contract = types.SimpleNamespace(cameras=("agentview", "wrist"))
        env.observer = types.SimpleNamespace(
            obs={
                "agentview_image": agentview,
                "robot0_eye_in_hand_image": wrist,
                "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "robot0_gripper_qpos": np.zeros(2),
            }
        )

        observation = env._observation()
        for camera, source in (("agentview", agentview), ("wrist", wrist)):
            message = rgb_to_image_msg(observation.images[camera], Time(), camera, 0, 0)
            np.testing.assert_array_equal(image_msg_to_rgb(message), source[::-1, ::-1])

    def test_resize_matches_official_openpi_client(self):
        client_source = PROJECT_ROOT.parent / "openpi/packages/openpi-client/src"
        sys.path.insert(0, str(client_source))
        from openpi_client import image_tools

        raw = np.arange(157 * 256 * 3, dtype=np.uint8).reshape(157, 256, 3)
        ros_image = image_msg_to_rgb(rgb_to_image_msg(np.ascontiguousarray(raw[::-1, ::-1]), Time(), "agentview", 2, 19))
        expected = image_tools.resize_with_pad(raw[::-1, ::-1], 224, 224)
        actual = _resize_with_pad(ros_image, 224)

        np.testing.assert_array_equal(actual, expected)

    def test_action_unnormalization_and_ros_transport_match_official_formula(self):
        q01 = np.linspace(-0.8, -0.2, 7, dtype=np.float64)
        q99 = np.linspace(0.3, 1.1, 7, dtype=np.float64)
        stats = {
            "norm_stats": {
                "state": {"q01": [0.0] * 8, "q99": [1.0] * 8},
                "actions": {"q01": q01.tolist(), "q99": q99.tolist()},
            }
        }
        normalized = np.linspace(-1.0, 1.0, 10 * 32, dtype=np.float32).reshape(1, 10, 32)

        with tempfile.TemporaryDirectory() as directory:
            stats_path = Path(directory) / "norm_stats.json"
            stats_path.write_text(json.dumps(stats), encoding="utf-8")
            import torch

            actions = OpenPIPostInfer(stats_path).infer(torch.from_numpy(normalized))

        expected = (normalized[0, :, :7] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        self.assertEqual(actions.dtype, np.float64)
        np.testing.assert_array_equal(actions, expected)

        node = _policy_harness()
        OpenPINode._publish_action(node, actions[0], 2, 19)
        transported = np.asarray(node.action_pub.messages[-1].data, dtype=np.float64)
        np.testing.assert_array_equal(transported, expected[0])


if __name__ == "__main__":
    unittest.main()
