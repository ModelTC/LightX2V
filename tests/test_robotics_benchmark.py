"""CPU checks for evaluation protocols; no ROS, model weights or simulator needed."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "lightx2v_ros/src/simulator"), str(ROOT / "lightx2v_ros/src/common")]
from simulator.libero_node import benchmark as libero  # noqa: E402
from simulator.libero_node.observer import LiberoActionObserver, observation_components  # noqa: E402
from simulator.robotwin_node import benchmark as robotwin  # noqa: E402
from simulator.sim import benchmark, evaluator  # noqa: E402


class RoboticsBenchmarkTest(unittest.TestCase):
    def config(self, name="robotwin", *args):
        with patch.dict(os.environ, {}, clear=True):
            return benchmark.load_config(name, list(args))

    def test_config_defaults_and_override_precedence(self):
        cfg = self.config()
        self.assertEqual(cfg["EVALUATION"]["replan_steps"], 24)
        self.assertEqual(cfg["model"]["native_config"]["actions_per_plan"], 24)
        self.assertEqual(cfg["EVALUATION"]["num_inference_steps"], 1)
        self.assertTrue(cfg["model"]["native_config"]["cuda_graph"])
        cfg = self.config("robotwin", "EVALUATION.replan_steps=8", "model.action_horizon=16")
        self.assertEqual(cfg["model"]["native_config"]["actions_per_plan"], 8)
        self.assertEqual(cfg["model"]["native_config"]["action_chunk_size"], 16)
        for name, trials in [("libero", 50), ("libero_plus", 1)]:
            cfg = self.config(name)
            self.assertEqual(cfg["EVALUATION"]["num_trials"], trials)
            self.assertEqual(cfg["EVALUATION"]["replan_steps"], 10)
        cfg = self.config("robotwin", "model=fastwam")
        self.assertEqual(cfg["model"]["native_config"]["model_cls"], "fastwam")
        self.assertEqual(cfg["EVALUATION"]["num_inference_steps"], 20)

    def test_invalid_config_fails_before_model_load(self):
        for arg in ["EVALUATION.replan_steps=33", "EVALUATION.num_inference_steps=0", "MULTIRUN.num_gpus=0", "seed=-1", "model=fasterwam"]:
            with self.subTest(arg=arg), self.assertRaises(ValueError):
                self.config("robotwin", arg)
        with self.assertRaises(ValueError):
            self.config("robotwin", f"config_json={ROOT}/configs/realtimewam/libero_i2va.json")
        with self.assertRaises(Exception):
            self.config("robotwin", "lora_path=/unsupported")

    def test_gpu_slots_respect_visible_devices(self):
        cfg = self.config("robotwin", "MULTIRUN.num_gpus=2")
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="3,7"):
            self.assertEqual(benchmark.gpu_slots(cfg), ["3", "3", "7", "7"])
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="3,3"), self.assertRaises(ValueError):
            benchmark.gpu_slots(cfg)

    def test_libero_observation_orientation_and_state(self):
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        raw = {"agentview_image": image, "robot0_eye_in_hand_image": image, "robot0_eef_pos": [1, 2, 3], "robot0_eef_quat": [0, 0, 0, 1], "robot0_gripper_qpos": [0.1, 0.2]}
        images, state = observation_components(raw)
        np.testing.assert_array_equal(images["agentview"], image[::-1, ::-1])
        self.assertTrue(images["wrist"].flags.c_contiguous)
        np.testing.assert_allclose(state, [1, 2, 3, 0, 0, 0, 0.1, 0.2])

    def test_libero_reset_owns_initial_state_and_settles(self):
        runtime = LiberoActionObserver.__new__(LiberoActionObserver)
        runtime.initial_states = np.array([[1.0, 2.0], [3.0, 4.0]])
        runtime.num_init_states = 2
        runtime.task = SimpleNamespace(name="test")
        env = runtime.env = Mock()

        def init_state(state):
            state[:] = 100
            return "initial"

        env.set_init_state.side_effect = init_state
        env.step.return_value = ("settled", 0, False, {})
        action = [0, 0, 0, 0, 0, 0, -1]
        self.assertEqual(runtime.reset(1, settle_steps=5, settle_action=action), "settled")
        np.testing.assert_array_equal(runtime.init_state, [3, 4])
        np.testing.assert_array_equal(runtime.initial_states[1], [3, 4])
        self.assertEqual(env.step.call_count, 5)
        with self.assertRaises(ValueError):
            runtime.reset(2)

    def test_plus_categories_are_required(self):
        cfg = self.config("libero_plus")
        with self.assertRaisesRegex(ValueError, "category"):
            libero.LiberoAdapter(cfg, {"category": None})

    def test_action_validation(self):
        for actions, space in [(np.zeros((32, 7)), "other"), (np.zeros((32, 14)), "libero_delta_eef"), (np.full((1, 7), np.nan), "libero_delta_eef")]:
            with self.assertRaises(ValueError):
                evaluator.ActionChunk(actions, space).validate("libero_delta_eef", 7)

    def test_ros_node_selects_existing_realtimewam_policy(self):
        class Node:
            def __init__(self, name):
                self.params = {"model_cls": "realtimewam", "config_json": "/profile.json", "model_path": "/model"}

            def declare_parameter(self, key, default):
                self.params.setdefault(key, default)

            def get_parameter(self, key):
                return SimpleNamespace(value=self.params[key])

            def get_logger(self):
                return Mock()

            def create_publisher(self, *args):
                return Mock()

            def create_subscription(self, *args):
                return Mock()

        modules = {}

        def module(name, **attrs):
            modules[name] = ModuleType(name)
            modules[name].__dict__.update(attrs)

        fast, realtime = Mock(), Mock()
        startup = Mock(return_value={"action_dim": 7, "robot_state_dim": 8})
        module("rclpy")
        module("rclpy.node", Node=Node)
        module("sensor_msgs")
        module("sensor_msgs.msg", Image=object)
        module("std_msgs")
        module("std_msgs.msg", **dict.fromkeys(["Bool", "Float32MultiArray", "Int32", "String"], object))
        module("lightx2v.models.runners.wan.fastwam_runner", FastWAMPolicy=fast)
        module("lightx2v.models.runners.wan.realtimewam_runner", RealtimeWAMPolicy=realtime)
        module("lightx2v.utils.set_config", build_startup_config=startup)
        spec = importlib.util.spec_from_file_location("test_wam_ros_node", ROOT / "lightx2v_ros/src/inference/inference/fastwam_node/main.py")
        with patch.dict(sys.modules, modules):
            node_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(node_module)
            node = node_module.FastWAMNode()
        realtime.from_config.assert_called_once()
        fast.from_config.assert_not_called()
        self.assertEqual(startup.call_args.args[0]["model_cls"], "realtimewam")
        self.assertIs(node.policy, realtime.from_config.return_value)

    def test_failed_rollout_records_error_and_closes_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config("libero", f"EVALUATION.output_dir={tmp}", "EVALUATION.num_trials=1")
            task = {"key": "test"}
            env = Mock()
            env.reset.side_effect = RuntimeError("simulator failed")
            with patch.object(evaluator, "make_environment", return_value=env), self.assertRaisesRegex(RuntimeError, "simulator failed"):
                evaluator.run_task(cfg, task, Mock())
            env.close.assert_called_once()
            result = json.loads(benchmark.result_path(tmp, task).read_text())
            self.assertEqual(result["status"], "error")
            self.assertIn("simulator failed", result["error"])

    def test_rollout_replans_at_24_and_stops_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config("robotwin", f"EVALUATION.output_dir={tmp}", "EVALUATION.eval_num_episodes=1")
            task = {"key": "clean/test", "phase": "clean"}
            obs = evaluator.Observation({}, np.zeros(14), "test")
            env = Mock(max_steps=100, action_space="robotwin_joint_position", action_dim=14)
            env.reset.return_value = obs
            env.episode_metadata.return_value = {"environment_seed": 4300000}
            calls = []

            def step(action, *, observe=True):
                calls.append((float(action[0]), observe))
                success = len(calls) == 26
                return obs if observe else None, success, success

            env.step.side_effect = step
            policy = Mock()
            actions = np.broadcast_to(np.arange(32)[:, None], (32, 14))
            policy.predict_action_chunk.return_value = evaluator.ActionChunk(actions, "robotwin_joint_position")
            with patch.object(evaluator, "make_environment", return_value=env):
                evaluator.run_task(cfg, task, policy)
            self.assertEqual([a for a, _ in calls], list(range(24)) + [0, 1])
            self.assertEqual([i for i, (_, capture) in enumerate(calls) if capture], [23])
            self.assertEqual(policy.predict_action_chunk.call_count, 2)
            env.close.assert_called_once()
            result = json.loads(benchmark.result_path(tmp, task).read_text())
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["episodes"][0]["success"])

    def test_robotwin_success_does_not_add_checks_or_observations(self):
        adapter = robotwin.RoboTwinAdapter.__new__(robotwin.RoboTwinAdapter)
        task = Mock(eval_success=False, take_action_cnt=1)
        task.check_success.side_effect = AssertionError("extra success check")
        adapter.env = Mock(env=task, task_description="test")
        adapter.max_steps, adapter.step_index = 100, 0
        raw = SimpleNamespace(images={}, state=np.zeros(14))
        adapter.env._observation.return_value = raw
        self.assertEqual(adapter.step(np.zeros(14), observe=False), (None, False, False))
        adapter.env._observation.assert_not_called()
        task.eval_success = True
        self.assertEqual(adapter.step(np.zeros(14)), (None, True, True))
        task.check_success.assert_not_called()
        adapter.env._observation.assert_not_called()

    def test_seed_cache_revalidates_and_removes_rejected_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config("robotwin", "EVALUATION.reuse_seed_cache=true", f"EVALUATION.seed_cache_dir={tmp}")
            path = Path(tmp) / "clean/test_seed.json"
            benchmark.atomic_json(path, [11, 12, 13])
            cache = robotwin.SeedCache(cfg, {"phase": "clean", "task_name": "test"})
            self.assertEqual(cache.next_candidate(4300000), 11)
            cache.reject(11)
            self.assertEqual(cache.next_candidate(4300000), 12)
            cache.record_validated(12)
            self.assertEqual(json.loads(path.read_text()), [12, 13])

    def test_summary_is_micro_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = [{"key": "a", "category": "small"}, {"key": "b", "category": "large"}]
            for task, successes in zip(tasks, [[True], [True, False, False]]):
                benchmark.atomic_json(benchmark.result_path(tmp, task), {"status": "complete", "episodes": [{"success": x} for x in successes]})
            summary = benchmark.summarize(tmp, tasks, 3)
            self.assertEqual(summary["overall"]["success_rate"], 0.5)
            self.assertEqual(summary["completed_trials"], 4)

    def test_cli_dry_run_and_resume_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "robotwin"
            (root / "envs").mkdir(parents=True)
            (root / "description/task_instruction").mkdir(parents=True)
            (root / "description/task_instruction/test.json").write_text("{}")
            output = Path(tmp) / "out"
            cfg = self.config("robotwin", f"EVALUATION.robotwin_root={root}", f"EVALUATION.output_dir={output}", "model.factory=test:factory", "dry_run=true")
            with patch.dict(os.environ, {}, clear=True):
                benchmark.run(cfg)
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(len(manifest["tasks"]), 2)
                self.assertFalse((output / "jobs").exists())
                with self.assertRaisesRegex(ValueError, "Output already exists"):
                    benchmark.run(cfg)
                cfg["EVALUATION"]["resume"] = True
                benchmark.run(cfg)
                cfg["EVALUATION"]["replan_steps"] = 8
                with self.assertRaises(ValueError):
                    benchmark.run(cfg)
            result = subprocess.run([sys.executable, "scripts/bench/robotics/run.py", "--help"], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("libero_plus", result.stdout)


if __name__ == "__main__":
    unittest.main()
