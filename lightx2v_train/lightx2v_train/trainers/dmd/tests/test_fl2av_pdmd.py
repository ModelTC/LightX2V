"""Base/keyframe PDMD: three-task recipes, cache routing and launchers."""

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
TASKS = ("t2av", "i2av", "fl2av")


def recipe(steps):
    return TRAIN_ROOT / f"configs/train/dmd/minimax_h3_fl2av_pdmd{steps}_official_fsdp32.yaml"


def sampler(rank, start=0, accum=1, tasks=("fl2av",), balance_orientation=True):
    # Uneven and non-multiple-of-16 pools exercise shuffle-boundary windows.
    rows = [
        {"type": "metadata", "row": {"task": task, "aspect_bucket": bucket, "id": f"{task}-{bucket}-{i}"}}
        for task in tasks
        for bucket, size in (("landscape", 101), ("portrait", 83))
        for i in range(size)
    ]
    result = MiniMaxH3TaskCycleSampler(
        SimpleNamespace(samples=rows),
        task_order=tasks,
        num_replicas=32,
        rank=rank,
        route_mode="stratified",
        balance_route_loss=True,
        require_all_routes_per_optimizer_step=True,
        balance_orientation=balance_orientation,
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
            data = config["data"]["train"]
            self.assertEqual(data["tasks"], list(TASKS))
            self.assertEqual(data["task_order"], list(TASKS))
            self.assertEqual(data["task_bucket_order"], {task: ["landscape", "portrait"] for task in TASKS})
            self.assertFalse(data["balance_orientation"])
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

    def test_three_tasks_random_orientation_cover_every_row_without_quotas(self):
        samplers = [sampler(rank, tasks=TASKS, balance_orientation=False) for rank in range(32)]
        seen = set()
        task_rows = {task: [] for task in TASKS}
        task_cycle = Counter()
        landscape_counts = []
        for offset in range(24):
            indices = [s._cursor_sample_index(offset) for s in samplers]
            self.assertEqual(len(set(indices)), 32)
            rows = [samplers[0].dataset.samples[i]["row"] for i in indices]
            counts = Counter(row["task"] for row in rows)
            self.assertEqual(sorted(counts.values()), [10, 11, 11])
            task_cycle.update(counts)
            if offset % 3 == 2:
                self.assertEqual(task_cycle, {task: 32 for task in TASKS})
                task_cycle.clear()
            landscape_counts.append(sum(row["aspect_bucket"] == "landscape" for row in rows))
            weighted = Counter()
            for s, row in zip(samplers, rows):
                weighted[row["task"]] += s.cursor_microbatch_loss_scale(offset)
                task_rows[row["task"]].append(row)
            for task in TASKS:
                self.assertAlmostEqual(weighted[task], 32 / 3)
            seen.update(indices)
        self.assertEqual(len(seen), 3 * 184)
        self.assertGreater(len(set(landscape_counts)), 1)
        self.assertTrue(any(count != 16 for count in landscape_counts))
        for rows in task_rows.values():
            first_pass = rows[:184]
            self.assertEqual(len({row["id"] for row in first_pass}), 184)
            self.assertEqual(Counter(row["aspect_bucket"] for row in first_pass), {"landscape": 101, "portrait": 83})

    def test_three_task_cursor_weights_accumulation_and_exact_resume(self):
        for balance_orientation in (True, False):
            for accum in (1, 2, 3):
                samplers = [sampler(rank, accum=accum, tasks=TASKS, balance_orientation=balance_orientation) for rank in range(32)]
                for update in range(3):
                    weighted = Counter()
                    for micro in range(accum):
                        offset = update * accum + micro
                        for s in samplers:
                            index = s._cursor_sample_index(offset)
                            row = s.dataset.samples[index]["row"]
                            key = (row["task"], row["aspect_bucket"] if balance_orientation else "mixed")
                            weighted[key] += s.cursor_microbatch_loss_scale(offset)
                    for value in weighted.values():
                        self.assertAlmostEqual(value / (32 * accum), 1 / len(weighted))
                for rank in (0, 1, 30, 31):
                    original = samplers[rank]
                    for start in (5, 10, 19):
                        resumed = sampler(rank, start=start, accum=accum, tasks=TASKS, balance_orientation=balance_orientation)
                        for epoch in (0, 1):
                            resumed.set_epoch(epoch)
                            offset = start + epoch * len(resumed)
                            self.assertEqual(list(resumed), [original._cursor_sample_index(offset + i) for i in range(len(resumed))])
                            self.assertEqual(resumed.cursor_microbatch_loss_scale(offset), original.cursor_microbatch_loss_scale(offset))
                if not balance_orientation:
                    self.assertFalse(samplers[0].checkpoint_metadata()["balance_orientation"])

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
            for task in TASKS:
                for bucket, height, width in (("landscape", 768, 1344), ("portrait", 1344, 768)):
                    folder = root / "cache" / task / bucket
                    folder.mkdir(parents=True)
                    row = {"task": task, "target_height": height, "target_width": width, "target_num_frames": 124, "condition_path": str(root / "condition.pt")}
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
                for value in (f"steps={steps}", recipe(steps).name, "total=192", "tasks=T2AV/I2AV/FL2AV", "orientations=natural_random", "11/11/10", "--nnodes=4 --nproc_per_node=8"):
                    self.assertIn(value, result.stdout)
                self.assertNotIn("landscape=16, portrait=16", result.stdout)
                self.assertFalse((root / "output").exists())
                bad = subprocess.run(["bash", str(launcher), "--dry-run"], env={**env, "H3_FL2AV_CACHE": str(root / "missing")}, text=True, capture_output=True)
                self.assertNotEqual(bad.returncode, 0)
            (root / "cache/i2av/portrait/metadata.jsonl").unlink()
            bad = subprocess.run(["bash", str(launcher), "--dry-run"], env=env, text=True, capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("Need both orientations", bad.stderr)


if __name__ == "__main__":
    unittest.main()
