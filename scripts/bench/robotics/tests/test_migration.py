"""Shared-runtime parity and canonical entrypoints without loading models or ROS."""

import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.bench.robotics.common.config import ROOT, load_config
from scripts.bench.robotics.common.simulator import bootstrap_simulator

bootstrap_simulator()
from common.contract import LIBERO_CONTRACT  # noqa: E402
from simulator.libero_node import runtime  # noqa: E402
from simulator.libero_node.env import LiberoEnv  # noqa: E402
from simulator.libero_node.observer import LiberoActionObserver  # noqa: E402

from scripts.bench.robotics.libero.adapter import LiberoAdapter, observation  # noqa: E402


def raw_observation():
    image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    return {
        **{key: image.copy() for key in runtime.CAMERA_OBS_KEYS.values()},
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.6, 0.0, 0.8]),
        "robot0_gripper_qpos": np.array([0.01, -0.01]),
    }


class FakeEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def seed(self, seed):
        self.calls.append(("seed", seed))

    def reset(self):
        self.calls.append(("reset",))

    def set_init_state(self, state):
        self.calls.append(("init", state.copy()))
        state[:] = -100  # simulator mutation must not corrupt the saved state
        return raw_observation()

    def step(self, action):
        self.calls.append(("step", np.asarray(action).copy()))
        return raw_observation(), 0.0, False, {}

    def close(self):
        self.calls.append(("close",))


class MigrationTests(unittest.TestCase):
    def test_legacy_sources_removed(self):
        self.assertEqual(list((ROOT / "experiments").rglob("*.py")), [])
        self.assertFalse((ROOT / "experiments/configs/eval.yaml").exists())

    def test_entrypoints_from_other_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("libero", "libero_plus", "robotwin"):
                entry = f"scripts/bench/robotics/run_{name}.py"
                result = subprocess.run([sys.executable, str(ROOT / entry), "--help"], cwd=directory, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("configs/bench/robotics/eval.yaml", result.stdout)

    def test_shared_observation_exact_batch_parity(self):
        raw = raw_observation()
        interactive = LiberoEnv.__new__(LiberoEnv)
        interactive.contract = replace(LIBERO_CONTRACT, cameras=("agentview", "wrist"))
        interactive.observer = SimpleNamespace(obs=raw)
        # Reference is the pre-migration batch conversion, including precision.
        for quat in ([0.0, 0.6, 0.0, 0.8], [0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 1.0001]):
            raw["robot0_eef_quat"] = quat
            q = np.asarray(quat, dtype=np.float32).copy()
            w = np.clip(q[3], -1.0, 1.0)
            den = np.sqrt(1.0 - w * w)
            axis = np.zeros(3, dtype=np.float32) if np.isclose(den, 0) else q[:3] * 2 * np.arccos(w) / den
            expected = np.concatenate([raw["robot0_eef_pos"], axis, raw["robot0_gripper_qpos"]]).astype(np.float32)
            batch, ros = observation(raw, "task", 2), interactive._observation()
            np.testing.assert_array_equal(batch.state, expected)
            np.testing.assert_array_equal(batch.state, ros.state)
            for camera in batch.images:
                np.testing.assert_array_equal(batch.images[camera], ros.images[camera])
                np.testing.assert_array_equal(batch.images[camera], raw[runtime.CAMERA_OBS_KEYS[camera]][::-1, ::-1])
                self.assertTrue(batch.images[camera].flags.c_contiguous)

    def test_entrypoint_does_not_shadow_libero_namespace(self):
        # Executing a script puts its own directory on sys.path. The adapter's
        # libero/__init__.py must not mask the simulator's namespace package.
        for name in ("libero", "libero_plus", "robotwin"):
            entry = ROOT / f"scripts/bench/robotics/run_{name}.py"
            code = (
                "import sys, runpy, importlib.util; from pathlib import Path; "
                "entry = Path(sys.argv[1]); sys.path.insert(0, str(entry.parent)); "
                "sys.argv = [str(entry), '--help']; runpy.run_path(str(entry), run_name='__main__'); "
                "assert str(entry.parent) not in sys.path; "
                "spec = importlib.util.find_spec('libero'); "
                "assert spec is None or spec.origin is None or not Path(spec.origin).is_relative_to(entry.parent)"
            )
            result = subprocess.run([sys.executable, "-c", code, str(entry)], cwd="/tmp", capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_shared_runtime_reset_and_protocol(self):
        task = SimpleNamespace(name="fake", language="pick", problem_folder="scene", bddl_file="task.bddl")
        states = np.arange(8).reshape(2, 4)
        suite = SimpleNamespace(get_num_tasks=lambda: 1, get_task=lambda _: task, get_task_init_states=lambda _: states)
        benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": lambda: suite})
        with patch.object(runtime, "load_libero", return_value=(benchmark, lambda _: "/bddl", FakeEnv)), patch.dict(os.environ, {}, clear=True):
            interactive = LiberoActionObserver()
            self.assertIsInstance(interactive, runtime.LiberoRuntime)
            self.assertEqual([c[0] for c in interactive.env.calls], ["seed", "reset", "init"])
            self.assertEqual(len(interactive.env.kwargs["camera_names"]), 4)
            self.assertEqual(interactive.env.kwargs["camera_heights"], 224)
            interactive.reset()
            np.testing.assert_array_equal(interactive.env.calls[-1][1], states[0])
            with self.assertRaises(ValueError):
                runtime.LiberoRuntime(init_state_id=2)
            cfg = load_config("libero", ["EVALUATION.output_dir=/tmp/fake", "EVALUATION.max_steps=1"])
            adapter = LiberoAdapter(cfg, {"suite": "libero_spatial", "task_id": 0, "prompt": "pick"})
            self.assertEqual([c[0] for c in adapter.env.calls], ["seed"])  # no extra RNG-consuming reset
            self.assertNotIn("camera_names", adapter.env.kwargs)
            self.assertEqual(adapter.env.kwargs["camera_heights"], 256)
            adapter.reset(3)  # cyclic evaluation indices, unlike interactive strict selection
            self.assertEqual(adapter.episode_metadata()["init_state_index"], 1)
            np.testing.assert_array_equal(adapter.env.calls[2][1], states[1])
            self.assertEqual([c[0] for c in adapter.env.calls], ["seed", "reset", "init"] + ["step"] * 5)
            for call in adapter.env.calls[3:]:
                np.testing.assert_array_equal(call[1], [0, 0, 0, 0, 0, 0, -1])
            _, success, done = adapter.step(np.zeros(7))
            self.assertFalse(success)
            self.assertTrue(done)
            adapter.close()
            interactive.close()
            np.testing.assert_array_equal(states, np.arange(8).reshape(2, 4))

    def test_plus_initial_states_owned_by_suite(self):
        import torch

        # A perturbation can map to a differently named initial-state file.
        suite = SimpleNamespace(get_task_init_states=lambda _: torch.load("suite_resolved.pt"))
        with patch.object(torch, "load", return_value=[np.ones(4)]) as loader:
            states = runtime.load_suite_init_states(suite, 3)
            loader.assert_called_once_with("suite_resolved.pt", weights_only=False, map_location="cpu")
            self.assertIs(torch.load, loader)  # scoped loader has been restored
            self.assertEqual(len(states), 1)
        with self.assertRaisesRegex(ValueError, "Empty"):
            runtime.load_suite_init_states(SimpleNamespace(get_task_init_states=lambda _: []), 0)
