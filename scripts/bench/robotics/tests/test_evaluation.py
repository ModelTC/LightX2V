import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf.errors import ConfigKeyError

from lightx2v.models.networks.wan.realtimewam_checkpoint import merge_linear_lora
from scripts.bench.robotics.common.config import ROBOTWIN_ROOT, ROOT, default_libero_root, load_config
from scripts.bench.robotics.common.interfaces import ActionChunk, Observation
from scripts.bench.robotics.common.manager import gpu_slots
from scripts.bench.robotics.common.results import atomic_json, result_path, summarize


class EvaluationTests(unittest.TestCase):
    def test_robotwin_planner_assets_relocate(self):
        from pathlib import Path

        from scripts.bench.robotics.robotwin.planner_adapter import relocate_asset_paths

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embodiment = root / "assets/embodiments/aloha-agilex"
            embodiment.mkdir(parents=True)
            (embodiment / "robot.urdf").touch()
            (embodiment / "spheres.yml").touch()
            config = {"robot_cfg": {"urdf_path": "/old/machine/RoboTwin/assets/embodiments/aloha-agilex/robot.urdf", "collision_spheres": "spheres.yml", "asset_root_path": None, "base_link": "base"}}
            relocated = relocate_asset_paths(config, root, embodiment)["robot_cfg"]
            self.assertEqual(relocated["urdf_path"], str(embodiment / "robot.urdf"))
            self.assertEqual(relocated["collision_spheres"], str(embodiment / "spheres.yml"))
            self.assertIsNone(relocated["asset_root_path"])
            with self.assertRaises(ValueError):
                relocate_asset_paths({"urdf_path": "/outside/robot.urdf"}, root, embodiment)
            with self.assertRaises(FileNotFoundError):
                relocate_asset_paths({"urdf_path": "missing.urdf"}, root, embodiment)

    def test_robotwin_defaults_and_cache_config(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = load_config("robotwin", ["EVALUATION.reuse_seed_cache=true"])
            self.assertEqual(cfg["EVALUATION"]["robotwin_root"], str(ROBOTWIN_ROOT.resolve()))
            self.assertEqual(cfg["EVALUATION"]["seed_cache_dir"], str(ROBOTWIN_ROOT.resolve() / "my_seeds"))
            with self.assertRaises(ValueError):
                load_config("libero", ["EVALUATION.reuse_seed_cache=true"])
        with patch.dict(os.environ, {"ROBOTWIN_ROOT": "/tmp/robotwin", "ROBOTWIN_SEED_DIR": "/tmp/cache"}, clear=True):
            cfg = load_config("robotwin", ["EVALUATION.reuse_seed_cache=true"])
            self.assertEqual(cfg["EVALUATION"]["robotwin_root"], "/tmp/robotwin")
            self.assertEqual(cfg["EVALUATION"]["seed_cache_dir"], "/tmp/cache")
            cfg = load_config("robotwin", ["EVALUATION.reuse_seed_cache=true", "EVALUATION.seed_cache_dir=/tmp/override"])
            self.assertEqual(cfg["EVALUATION"]["seed_cache_dir"], "/tmp/override")

    def test_seed_cache_legacy_revalidation_and_persistence(self):
        from pathlib import Path
        from types import SimpleNamespace

        from scripts.bench.robotics.robotwin.seed_cache import SeedCache, find_validated_seed, load_seeds

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / "clean/adjust_bottle_seed.json"
            atomic_json(path, [100, 102, 104])
            cfg = load_config("robotwin", ["EVALUATION.reuse_seed_cache=true", f"EVALUATION.seed_cache_dir={directory}", "seed=123"])
            cache = SeedCache(cfg, {"phase": "clean", "task_name": "adjust_bottle"})
            attempted = []
            env = SimpleNamespace(seed=12400000, _setup_demo=lambda: None, _close_task_env=lambda: None, _log=lambda _: None)

            def play():
                attempted.append(env.seed)
                return {"info": {"seed": env.seed}}

            env.env = SimpleNamespace(play_once=play, plan_success=True, check_success=lambda: env.seed != 100)
            info = find_validated_seed(env, cache, 3)
            self.assertEqual(attempted, [100, 102])  # cached candidates still run expert
            self.assertEqual(info["info"]["seed"], 102)
            self.assertTrue(cache.current_hit)
            cache.record_validated(env.seed)
            self.assertEqual(load_seeds(path), [102, 104])  # rejected removed, unused retained
            self.assertEqual(cache.next_candidate(0), 104)
            self.assertEqual(cache.next_candidate(0), 105)
            self.assertFalse(cache.current_hit)
            # A second writer's additions survive the first writer's next save.
            atomic_json(path, [102, 104, 108])
            cache.record_validated(105)
            self.assertEqual(load_seeds(path), [102, 104, 105, 108])
            random_cache = SeedCache(cfg, {"phase": "random", "task_name": "adjust_bottle"})
            self.assertEqual(random_cache.next_candidate(0), 12400000)
            random_cache.record_validated(12400000)
            self.assertEqual(load_seeds(Path(directory) / "random/adjust_bottle_seed.json"), [12400000])
            self.assertEqual(load_seeds(path), [102, 104, 105, 108])
            with patch("scripts.bench.robotics.robotwin.seed_cache.atomic_json", side_effect=OSError("read only")), self.assertWarns(UserWarning):
                cache.record_validated(110)
            self.assertEqual(load_seeds(path), [102, 104, 105, 108])

    def test_seed_cache_disabled_and_invalid_input(self):
        from pathlib import Path

        from scripts.bench.robotics.robotwin.seed_cache import SeedCache, load_seeds

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / "clean/adjust_bottle_seed.json"
            atomic_json(path, [100])
            cfg = load_config("robotwin", [f"EVALUATION.seed_cache_dir={directory}"])
            cache = SeedCache(cfg, {"phase": "clean", "task_name": "adjust_bottle"})
            self.assertEqual(cache.next_candidate(4300000), 4300000)
            cache.record_validated(4300001)
            self.assertEqual(load_seeds(path), [100])
            for invalid in ([True], [-1], [2, 2], [2, 1], {"seeds": [1]}):
                atomic_json(path, invalid)
                with self.assertWarns(UserWarning):
                    self.assertEqual(load_seeds(path), [])

    def test_repo_local_benchmark_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            for benchmark, name in (("libero", "LIBERO"), ("libero_plus", "LIBERO-plus")):
                expected = ROOT / "lightx2v_ros/src/simulator/simulator/libero_node" / name
                self.assertEqual(default_libero_root(benchmark), expected)
                self.assertEqual(load_config(benchmark, [])["EVALUATION"]["libero_root"], str(expected.resolve()))
                # Explicit null also opts back in to the repository default.
                self.assertEqual(load_config(benchmark, ["EVALUATION.libero_root=null"])["EVALUATION"]["libero_root"], str(expected.resolve()))
            cfg = load_config("libero_plus", ["EVALUATION.libero_root=/tmp/custom-plus"])
            self.assertEqual(cfg["EVALUATION"]["libero_root"], "/tmp/custom-plus")
        with patch.dict(os.environ, {"LIBERO_SOURCE_DIR": "/tmp/plain", "LIBERO_PLUS_SOURCE_DIR": "/tmp/plus"}, clear=True):
            self.assertEqual(load_config("libero", [])["EVALUATION"]["libero_root"], "/tmp/plain")
            self.assertEqual(load_config("libero_plus", [])["EVALUATION"]["libero_root"], "/tmp/plus")
            self.assertEqual(load_config("libero", ["EVALUATION.libero_root=/tmp/override"])["EVALUATION"]["libero_root"], "/tmp/override")

    def test_missing_benchmark_initialization_hint(self):
        from scripts.bench.robotics.libero.adapter import load_libero

        with patch.dict(os.environ, {}, clear=True):
            cfg = load_config("libero_plus", ["EVALUATION.output_dir=/tmp/unused-missing-libero"])
        with patch("pathlib.Path.is_dir", return_value=False), self.assertRaisesRegex(FileNotFoundError, "git submodule update --init --recursive"):
            load_libero(cfg)

    def test_libero_namespace_package_reload(self):
        import sys
        from pathlib import Path
        from types import ModuleType

        from scripts.bench.robotics.libero.adapter import load_libero

        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {}, clear=True):
            package = Path(root) / "libero/libero"
            (package / "bddl_files").mkdir(parents=True)
            outer, inner, envs = (ModuleType(name) for name in ("libero", "libero.libero", "libero.libero.envs"))
            outer.__file__ = None
            inner.__file__ = str(package / "__init__.py")
            inner.benchmark, inner.get_libero_path, envs.OffScreenRenderEnv = object(), object(), object()
            cfg = load_config("libero", [f"EVALUATION.libero_root={root}", f"EVALUATION.output_dir={root}/output"])
            with patch.dict(sys.modules, {"libero": outer, "libero.libero": inner, "libero.libero.envs": envs}), patch.object(sys, "path", sys.path.copy()):
                self.assertEqual(load_libero(cfg), load_libero(cfg))
                inner.__file__ = "/other/libero/libero/__init__.py"
                with self.assertRaisesRegex(RuntimeError, "separate processes"):
                    load_libero(cfg)

    def test_config_defaults_and_strict_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = load_config("robotwin", ["model=realtimewam", "base_ckpt=base.pt", "lora_path=lora", "seed=123"])
            self.assertEqual(cfg["seed"], 123)
            self.assertEqual(cfg["model"]["backbone"], "fastwam")
            self.assertEqual(cfg["EVALUATION"]["action_infer_mode"], "first_frame")
            with self.assertRaises(ConfigKeyError):
                load_config("libero", ["EVALUATION.num_trails=3"])
            with self.assertRaises(ValueError):
                load_config("libero", ["base_ckpt=x", "ckpt=y"])
            with self.assertRaises(ValueError):
                load_config("libero", ["model.sampler=consistency_baseline"])

    def test_gpu_mapping(self):
        cfg = {"MULTIRUN": {"num_gpus": 2, "max_tasks_per_gpu": 2}}
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,7"}):
            self.assertEqual(gpu_slots(cfg), ["4", "4", "7", "7"])
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}), self.assertRaises(ValueError):
            gpu_slots(cfg)

    def test_lora_merge(self):
        base = {"mixtures.action.q.weight": torch.ones(3, 4)}
        a, b = torch.arange(8.0).reshape(2, 4), torch.ones(3, 2)
        adapter = {"base_model.model.q.lora_A.weight": a, "base_model.model.q.lora_B.weight": b}
        count = merge_linear_lora(base, adapter, {"r": 2, "lora_alpha": 4}, "mixtures.action")
        self.assertEqual(count, 1)
        torch.testing.assert_close(base["mixtures.action.q.weight"], 1 + 2 * b @ a)
        adapter["unused"] = torch.ones(1)
        with self.assertRaises(ValueError):
            merge_linear_lora(base, adapter, {"r": 2, "lora_alpha": 4}, "mixtures.action")

    def test_invalid_actions(self):
        for array in (np.zeros((0, 7)), np.zeros((3, 14)), np.full((3, 7), np.nan)):
            with self.assertRaises(ValueError):
                ActionChunk(array, "libero_delta_eef").validate("libero_delta_eef", 7)
        with self.assertRaises(ValueError):
            ActionChunk(np.zeros((3, 7)), "robotwin_joint_position").validate("libero_delta_eef", 7)

    def test_bfloat16_merge_matches_peft_safe_merge(self):
        torch.manual_seed(8)
        weight = torch.randn(8, 8).bfloat16()
        a, b = torch.randn(2, 8), torch.randn(8, 2)
        expected = weight.clone()
        expected += (b @ a * 0.5).to(expected.dtype)
        state = {"mixtures.action.q.weight": weight}
        adapter = {"base_model.model.q.lora_A.weight": a, "base_model.model.q.lora_B.weight": b}
        merge_linear_lora(state, adapter, {"r": 2, "lora_alpha": 1}, "mixtures.action")
        torch.testing.assert_close(state["mixtures.action.q.weight"], expected, rtol=0, atol=0)

    def test_future_video_mask_and_fusion(self):
        from types import SimpleNamespace as NS

        from lightx2v.models.networks.wan.realtimewam_model import RealtimeWAMTransformerInfer

        cfg = {"num_layers": 3, "num_heads": 1, "dim": 4, "condition_layers": [0, 2], "video_kv_fusion": "interval_weighted_sum"}
        infer = RealtimeWAMTransformerInfer(cfg)
        masks = []

        class Attention:
            def apply(self, q, k, v, attn_mask=None):
                masks.append(attn_mask.clone())
                return q

        blocks = [NS(index=i, self_attn=NS(attn=Attention())) for i in range(3)]

        def build(block, x, freqs, mod):
            kv = torch.full((4, 1, 4), float(block.index + 1))
            return kv, kv, kv, x, None, None, None, None

        infer._build_self_attention_io = build
        infer._post_block = lambda block, residual, *args: residual
        weights = NS(video=NS(blocks=blocks), fusion_logits=[NS(tensor=torch.zeros(1)), NS(tensor=torch.zeros(2))])
        pre = NS(tokens=torch.zeros(4, 4), tokens_per_frame=2, freqs=None, t_mod=None, context=None, context_mask=None)
        cache = infer.prefill_video_cache(weights, pre)
        self.assertIsNone(cache[1])
        torch.testing.assert_close(cache[2]["k"], torch.full((4, 1, 4), 2.5))
        self.assertFalse(masks[0][:2, 2:].any())
        self.assertTrue(masks[0][2:, :].all())

    def test_trial_weighted_summary(self):
        with tempfile.TemporaryDirectory() as out:
            tasks = [{"key": "a", "category": "Camera"}, {"key": "b", "category": "Robot"}]
            atomic_json(result_path(out, tasks[0]), {"status": "complete", "episodes": [{"success": True}], "error": None})
            atomic_json(result_path(out, tasks[1]), {"status": "error", "episodes": [{"success": False}] * 3, "error": "crashed"})
            result = summarize(out, tasks, 4)
            self.assertEqual(result["overall"]["success_rate"], 0.25)
            self.assertEqual(result["planned_trials"], 8)
            self.assertFalse(result["complete"])

    def test_replan_and_reset(self):
        from scripts.bench.robotics.common.evaluator import run_task

        class Env:
            action_dim, action_space, max_steps = 7, "libero_delta_eef", 5

            def reset(self, index):
                self.count = 0
                return Observation({}, np.zeros(8), "task")

            def episode_metadata(self):
                return {"environment_seed": 42}

            def step(self, action):
                self.count += 1
                return Observation({}, np.zeros(8), "task", self.count), self.count == 5, self.count == 5

            def close(self):
                pass

        class Policy:
            resets, calls = 0, 0

            def reset_episode(self, metadata):
                self.resets += 1

            def predict_action_chunk(self, obs):
                self.calls += 1
                return ActionChunk(np.zeros((32, 7)), "libero_delta_eef")

        with tempfile.TemporaryDirectory() as out, patch.dict(os.environ, {}, clear=True):
            cfg = load_config("libero", [f"EVALUATION.output_dir={out}", "EVALUATION.num_trials=2", "EVALUATION.replan_steps=2"])
            policy = Policy()
            with patch("scripts.bench.robotics.common.evaluator.make_environment", return_value=Env()):
                run_task(cfg, {"key": "test"}, policy)
            self.assertEqual(policy.calls, 6)
            self.assertEqual(policy.resets, 2)

    def test_robotwin_step_contract(self):
        from types import SimpleNamespace

        from scripts.bench.robotics.robotwin.adapter import RoboTwinAdapter

        adapter = RoboTwinAdapter.__new__(RoboTwinAdapter)
        adapter.step_index, adapter.max_steps = 0, 20
        raw = SimpleNamespace(images={}, state=np.zeros(14))
        adapter.env = SimpleNamespace(step=lambda _: (raw, False, False), task_description="test")
        observation, success, done = adapter.step(np.zeros(14))
        self.assertEqual(observation.step, 1)
        self.assertFalse(success)
        self.assertFalse(done)

    def test_robotwin_observation_flag(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(load_config("robotwin", [])["EVALUATION"]["skip_get_obs_within_replan"])
            self.assertFalse(load_config("robotwin", ["EVALUATION.skip_get_obs_within_replan=false"])["EVALUATION"]["skip_get_obs_within_replan"])
            self.assertFalse(load_config("libero", [])["EVALUATION"]["skip_get_obs_within_replan"])
            for benchmark, value in (("libero", "true"), ("libero_plus", "true"), ("robotwin", "1")):
                with self.assertRaises(ValueError):
                    load_config(benchmark, [f"EVALUATION.skip_get_obs_within_replan={value}"])

    def test_robotwin_chunk_observation_equivalence(self):
        import json

        from scripts.bench.robotics.common.evaluator import run_task

        class Env:
            action_dim, action_space, max_steps = 14, "robotwin_joint_position", 19

            def __init__(self, stop, success):
                self.stop, self.success = stop, success
                self.actions, self.observations = [], []

            def reset(self, index):
                self.count = 0
                return Observation({}, np.zeros(14), "task", 0)

            def episode_metadata(self):
                return {"environment_seed": 42}

            def step(self, action, *, observe=True):
                self.actions.append(action.copy())
                self.count += 1
                self.observations.append(observe)
                obs = Observation({}, np.full(14, self.count), "task", self.count) if observe else None
                done = self.count == self.stop
                return obs, done and self.success, done

            def close(self):
                pass

        class Policy:
            def __init__(self, horizon):
                self.horizon, self.inputs = horizon, []

            def reset_episode(self, metadata):
                pass

            def predict_action_chunk(self, obs):
                self.inputs.append((obs.step, obs.state.copy()))
                return ActionChunk(np.stack([obs.state + i + 1 for i in range(self.horizon)]), "robotwin_joint_position")

        # Covers early success, early unsuccessful termination, rollout cap,
        # a short returned action chunk, replan=1, and multiple episode resets.
        for replan, horizon, stop, success in ((8, 32, 3, True), (8, 32, 9, False), (8, 32, None, False), (8, 3, None, False), (1, 32, None, False)):
            runs = []
            for mode in (False, True, "legacy"):
                with self.subTest(replan=replan, horizon=horizon, stop=stop, mode=mode), tempfile.TemporaryDirectory() as out, patch.dict(os.environ, {}, clear=True):
                    cfg = load_config("robotwin", [f"EVALUATION.output_dir={out}", "EVALUATION.eval_num_episodes=2", f"EVALUATION.replan_steps={replan}"])
                    if mode == "legacy":
                        cfg["EVALUATION"].pop("skip_get_obs_within_replan")
                    else:
                        cfg["EVALUATION"]["skip_get_obs_within_replan"] = mode
                    env, policy = Env(stop, success), Policy(horizon)
                    with patch("scripts.bench.robotics.common.evaluator.make_environment", return_value=env):
                        run_task(cfg, {"key": "test"}, policy)
                    result = json.loads(result_path(out, {"key": "test"}).read_text())
                    runs.append((env, policy, [(e["steps"], e["success"], e["inference_calls"]) for e in result["episodes"]]))
            baseline, optimized, legacy = runs
            for candidate in (optimized, legacy):
                np.testing.assert_array_equal(baseline[0].actions, candidate[0].actions)
                self.assertEqual(baseline[2], candidate[2])
                self.assertEqual([x[0] for x in baseline[1].inputs], [x[0] for x in candidate[1].inputs])
                np.testing.assert_array_equal([x[1] for x in baseline[1].inputs], [x[1] for x in candidate[1].inputs])
            self.assertTrue(all(legacy[0].observations))
            if stop is None:
                # One reset observation per episode is outside step().
                self.assertEqual(sum(optimized[0].observations), len(optimized[1].inputs) - 2)
            if replan > 1:
                self.assertLess(sum(optimized[0].observations), sum(baseline[0].observations))

    def test_robotwin_native_skip_preserves_execution(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock

        from scripts.bench.robotics.robotwin.adapter import RoboTwinAdapter

        with patch.object(sys, "path", [str(ROOT / "lightx2v_ros/src/common"), str(ROOT / "lightx2v_ros/src/simulator"), *sys.path]):
            from simulator.robotwin_node.env import RoboTwinEnv

        env = RoboTwinEnv.__new__(RoboTwinEnv)
        env.contract = SimpleNamespace(action_dim=14)
        env.env = SimpleNamespace(take_action=Mock(), eval_success=False, check_success=Mock(return_value=False))
        env._observation = Mock(return_value=SimpleNamespace(images={}, state=np.zeros(14)))
        env._task_description = "test"
        adapter = RoboTwinAdapter.__new__(RoboTwinAdapter)
        adapter.env, adapter.step_index, adapter.max_steps = env, 0, 2
        obs, success, done = adapter.step(np.zeros(14), observe=False)
        self.assertIsNone(obs)
        self.assertFalse(success)
        self.assertFalse(done)
        env._observation.assert_not_called()
        env.env.check_success.assert_called_once()
        env.env.take_action.assert_called_once()
        obs, success, done = adapter.step(np.zeros(14))
        self.assertEqual(obs.step, 2)
        self.assertTrue(done)  # adapter's step limit still enforced
        self.assertFalse(success)
        self.assertEqual(env.env.check_success.call_count, 2)
        env._observation.assert_called_once()
        env.env.eval_success = True
        self.assertEqual(env.step(np.zeros(14), observe=False), (None, True, True))

    def test_custom_policy_options(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = load_config("libero", ["model=my_model", "model.factory=my_package:build", "model.options.endpoint=localhost"])
            self.assertEqual(cfg["model"]["options"]["endpoint"], "localhost")


if __name__ == "__main__":
    unittest.main()
