"""FL2AV official PDMD: recipes, legacy cache routing, gradients and launchers."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import default_collate

from lightx2v_train.data.minimax_h3_cache_dataset import MiniMaxH3TaskCycleSampler, _strip_condition_batch
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability import MiniMaxH3DistributionMatchingCapability
from lightx2v_train.model_zoo.minimax_h3.tests.test_ref2av_capability import model_wrapper
from lightx2v_train.runtime.config import load_config
from lightx2v_train.trainers.dmd.official_pdmd_core import critic_objective, full_rollout, joint_noise_like, rollout_grid, student_objective
from lightx2v_train.trainers.dmd.official_pdmd_training import validate_official_pdmd_config

TRAIN_ROOT = Path(__file__).resolve().parents[4]


def recipe(steps):
    return TRAIN_ROOT / f"configs/train/dmd/minimax_h3_fl2av_pdmd{steps}_official_fsdp32.yaml"


def sampler(rank, start=0, accum=1):
    # Uneven and non-multiple-of-16 pools exercise shuffle-boundary windows.
    rows = [{"type": "metadata", "row": {"task": "fl2av", "aspect_bucket": bucket, "id": f"{bucket}-{i}"}} for bucket, size in (("landscape", 101), ("portrait", 83)) for i in range(size)]
    result = MiniMaxH3TaskCycleSampler(
        SimpleNamespace(samples=rows),
        task_order=["fl2av"],
        num_replicas=32,
        rank=rank,
        route_mode="stratified",
        balance_route_loss=True,
        require_all_routes_per_optimizer_step=True,
        seed=42,
    )
    result.configure_from_microbatch_offset(start_microbatch=start, gradient_accumulation_iters=accum, fake_update_ratio=5)
    return result


class TinyTransformer(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))
        self.conditions = []

    def forward(self, hidden_states, audio_hidden_states, **kwargs):
        self.conditions.append(hidden_states[:, :12].detach().clone())
        return self.weight * hidden_states.sin(), self.weight * audio_hidden_states.cos()


class Fl2avPdmdTest(unittest.TestCase):
    def test_recipes_only_differ_in_steps_and_retain_official_hyperparameters(self):
        env = {"H3_MODEL_PATH": "/model", "H3_FL2AV_CACHE": "/old-cache", "H3_FL2AV_PDMD_OUTPUT": "/output"}
        with patch.dict(os.environ, env):
            four, eight = load_config(recipe(4)), load_config(recipe(8))
        for config in (four, eight):
            validate_official_pdmd_config(config)
            self.assertEqual(config["data"]["train"]["tasks"], ["fl2av"])
            self.assertEqual(config["model"]["name"], "minimax_h3_t2av")
            self.assertEqual(config["training"]["student"]["lora"]["param_dtype"], "fp32")
            self.assertEqual(config["training"]["student"]["lora"]["rank"], 128)
            self.assertEqual(config["training"]["student"]["lora"]["alpha"], 8)
            self.assertEqual(config["training"]["fake"]["train_type"], "full")
            self.assertEqual(config["training"]["fake"]["optimizer"]["learning_rate"], 1e-5)
            self.assertEqual(config["training"]["max_train_iters"], 250000)
            self.assertEqual(config["training"]["save_every_iters"], 50)
            self.assertEqual(config["distributed"]["fsdp2"]["size"], 32)
        four.pop("config_path")
        eight.pop("config_path")
        eight["training"]["dmd"]["num_inference_steps"] = 4
        self.assertEqual(four, eight)
        bad = copy.deepcopy(four)
        bad["data"]["train"]["route_mode"] = "homogeneous"
        with self.assertRaisesRegex(ValueError, "stratified"):
            validate_official_pdmd_config(bad)

    def test_cursor_batches_balanced_disjoint_cover_every_row_and_resume(self):
        samplers = [sampler(rank) for rank in range(32)]
        seen = set()
        for offset in range(24):
            indices = [s._cursor_sample_index(offset) for s in samplers]
            self.assertEqual(len(set(indices)), 32)
            rows = [samplers[0].dataset.samples[i]["row"] for i in indices]
            self.assertEqual(Counter(r["aspect_bucket"] for r in rows), {"landscape": 16, "portrait": 16})
            self.assertTrue(all(r["task"] == "fl2av" for r in rows))
            seen.update(indices)
        self.assertEqual(len(seen), 184)
        for rank in (0, 1, 30, 31):
            original = sampler(rank, accum=2)
            for start in (5, 10, 19):
                resumed = sampler(rank, start=start, accum=2)
                for epoch in (0, 1):
                    resumed.set_epoch(epoch)
                    offset = start + epoch * len(resumed)
                    self.assertEqual(list(resumed), [original._cursor_sample_index(offset + i) for i in range(len(resumed))])
                meta = resumed.checkpoint_metadata()
                self.assertEqual(meta["resume_cursor_unit"], "absolute_global_microbatch")
                self.assertEqual(meta["student_data_source"], "reuse_preceding_critic_trajectory")
        original.configure(start_iteration=0, gradient_accumulation_iters=1, fake_update_ratio=1)
        self.assertIsNone(original.start_microbatch)
        self.assertNotIn("resume_cursor_unit", original.checkpoint_metadata())

    def test_fl2av_packed_condition_survives_rollout_and_both_backward_paths(self):
        torch.manual_seed(123)
        roles = []
        for coefficient in (0.05, 0.1, 0.2):
            model = model_wrapper(reference=False)
            model.transformer = TinyTransformer(coefficient)
            roles.append(MiniMaxH3DistributionMatchingCapability(model, {"official_pdmd": True, "geometry_from_metadata": True}))
        student, fake, teacher = roles
        teacher.model.transformer.requires_grad_(False)
        cached = {
            "task": "fl2av",
            "prompt_embeds": torch.randn(1, 4, 5120),
            "text_token_tags": torch.ones(4, dtype=torch.long),
            "target_height": 64,
            "target_width": 96,
            "target_num_frames": 124,
            "keyframe_anchors": torch.tensor([0, 1]),
            "condition_video_latents": torch.randn(12, 96),
        }
        batch = default_collate([{"inputs": {}, "conditioning": {"positive": _strip_condition_batch(cached)}, "meta": {}}])
        condition, _ = student.encode_conditions(batch, "", 1.0, lambda x: x)
        self.assertEqual(condition["keyframe_anchors"], ("first", "last"))
        shape = student.latent_shape(batch, [], lambda x: x)
        initial = student.initial_latents(shape, torch.float32, lambda x: x)
        for steps in (4, 8):
            trajectory = full_rollout(student, initial, condition, rollout_grid(steps), rollout_grid(steps))
            self.assertEqual(len(trajectory.states), steps + 1)
            noise = joint_noise_like(initial, torch.Generator().manual_seed(42))
            critic = critic_objective(student, fake, trajectory, torch.tensor([0.6, 0.4]), noise)
            critic.backward()
            self.assertTrue(torch.isfinite(fake.model.transformer.weight.grad))
            self.assertIsNone(student.model.transformer.weight.grad)
            fake.model.transformer.zero_grad(set_to_none=True)
            loss, _ = student_objective(student, fake, teacher, trajectory, steps - 1, 0.5)
            loss.backward()
            self.assertTrue(torch.isfinite(student.model.transformer.weight.grad))
            self.assertNotEqual(float(student.model.transformer.weight.grad), 0.0)
            self.assertIsNone(fake.model.transformer.weight.grad)
            self.assertIsNone(teacher.model.transformer.weight.grad)
            student.model.transformer.zero_grad(set_to_none=True)
        for role in roles:
            for rows in role.model.transformer.conditions:
                torch.testing.assert_close(rows[0], condition["noised_condition_video_latents"], rtol=0, atol=0)

    def test_both_launchers_dry_run_and_reject_wrong_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model/transformer").mkdir(parents=True)
            (root / "model/transformer/config.json").write_text("{}")
            (root / "condition.pt").write_bytes(b"not loaded by dry-run")
            for bucket, height, width in (("landscape", 768, 1344), ("portrait", 1344, 768)):
                folder = root / "cache/fl2av" / bucket
                folder.mkdir(parents=True)
                row = {"task": "fl2av", "target_height": height, "target_width": width, "target_num_frames": 124, "condition_path": str(root / "condition.pt")}
                (folder / "metadata.jsonl").write_text((json.dumps(row) + "\n") * 32)
            env = {
                **os.environ,
                "H3_CODE_ROOT": str(TRAIN_ROOT.parent),
                "H3_PYTHON": sys.executable,
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29999",
                "H3_MODEL_PATH": str(root / "model"),
                "H3_FL2AV_CACHE": str(root / "cache"),
                "H3_FL2AV_PDMD_OUTPUT": str(root / "output"),
                "H3_CONFIG_PATH": "/stale/ref-config.yaml",
                "LOCAL_KERNELS": "",
                "KERNELS_CACHE": "",
                "H3_KERNEL_SNAPSHOT": "",
            }
            for steps in (4, 8):
                launcher = TRAIN_ROOT / f"scripts/run_minimax_h3_fl2av_pdmd{steps}_official_fsdp32_32gpu_acp.sh"
                result = subprocess.run(["bash", str(launcher), "--dry-run"], env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                for value in (f"steps={steps}", recipe(steps).name, "total=64", "landscape=16, portrait=16", "--nnodes=4 --nproc_per_node=8"):
                    self.assertIn(value, result.stdout)
                self.assertFalse((root / "output").exists())
                bad = subprocess.run(["bash", str(launcher), "--dry-run"], env={**env, "H3_FL2AV_CACHE": str(root / "missing")}, text=True, capture_output=True)
                self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
