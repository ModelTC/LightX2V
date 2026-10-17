"""CPU config, schedule and ACP preflight checks for matched 6/3 PDMD shifts."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from lightx2v_train.runtime.config import load_config
from lightx2v_train.trainers.dmd.official_pdmd_core import interval_sigmas, rollout_grid

TRAIN_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_fl2av_pdmd_shift6_3_fsdp32_32gpu_acp.sh"


def recipe(steps, suffix="shift6_3"):
    return TRAIN_ROOT / f"configs/train/dmd/minimax_h3_fl2av_pdmd{steps}_{suffix}_fsdp32.yaml"


class Fl2avPdmdShift63Test(unittest.TestCase):
    def test_only_shifts_and_checkpoint_limit_change(self):
        env = {"H3_MODEL_PATH": "/model", "H3_FL2AV_CACHE": "/cache", "H3_FL2AV_PDMD_OUTPUT": "/output"}
        with patch.dict(os.environ, env):
            for steps in (4, 8):
                with self.subTest(steps=steps):
                    old = load_config(recipe(steps, "official"))
                    new = load_config(recipe(steps))
                    old.pop("config_path")
                    new.pop("config_path")
                    dmd = new["training"]["dmd"]
                    matching = new["model"]["capabilities"]["distribution_matching"]
                    self.assertEqual(dmd["num_inference_steps"], steps)
                    self.assertEqual((dmd["official_rollout_video_shift"], dmd["official_rollout_audio_shift"]), (6.0, 3.0))
                    self.assertEqual((matching["video_flow_shift"], matching["audio_flow_shift"]), (6.0, 3.0))
                    self.assertFalse(new["scheduler"]["time_shift_settings"]["do_time_shift"])
                    self.assertEqual(new["training"]["save_every_iters"], 50)
                    self.assertEqual(new["training"]["save_total_limit"], 2)
                    # Normalize intentional overrides to recover the original recipe exactly.
                    new["training"]["save_total_limit"] = old["training"]["save_total_limit"]
                    dmd["official_rollout_video_shift"] = 1.0
                    dmd["official_rollout_audio_shift"] = 1.0
                    matching["video_flow_shift"] = 12.0
                    self.assertEqual(new, old)

    def test_shifted_grids_and_interval_queries(self):
        for steps in (4, 8):
            with self.subTest(steps=steps):
                grids = []
                for shift in (6.0, 3.0):
                    grid = rollout_grid(steps, shift)
                    base = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
                    expected = (shift * base / (1 + (shift - 1) * base)).float()
                    torch.testing.assert_close(grid, expected)
                    self.assertEqual((grid[0].item(), grid[-1].item()), (1.0, 0.0))
                    self.assertTrue(torch.all(grid[:-1] > grid[1:]))
                    grids.append(grid)
                for index in range(steps):
                    query = interval_sigmas(*grids, index, 0.5)
                    torch.testing.assert_close(query, torch.stack([(grid[index] + grid[index + 1]) / 2 for grid in grids]))

    def test_launcher_selects_shifted_recipes_without_starting_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model/transformer").mkdir(parents=True)
            (root / "model/transformer/config.json").write_text("{}")
            (root / "condition.pt").write_bytes(b"metadata-only preflight must not load this")
            for task in ("t2av", "i2av", "fl2av"):
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
                "H3_FL2AV_PDMD_CONFIG": "/stale/original-recipe.yaml",
                "H3_RDZV_ID": "",
                "LOCAL_KERNELS": "",
                "KERNELS_CACHE": "",
                "H3_KERNEL_SNAPSHOT": "",
            }
            for steps in (4, 8):
                with self.subTest(steps=steps):
                    result = subprocess.run(["bash", str(LAUNCHER), str(steps), "--dry-run"], env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    for expected in (
                        recipe(steps).name,
                        f"steps={steps}",
                        "total=192",
                        "Training rollout shifts: video=6.0, audio=3.0; inference shifts: video=6.0, audio=3.0",
                        "tasks=T2AV/I2AV/FL2AV",
                        "orientations=natural_random",
                        "11/11/10",
                        "--nnodes=4 --nproc_per_node=8",
                        f"h3_fl2av_3task_pdmd{steps}_shift6_3_lorafp32_32gpu",
                    ):
                        self.assertIn(expected, result.stdout)
                    self.assertNotIn("/stale/original-recipe.yaml", result.stdout)
                    self.assertFalse((root / "output").exists())
            bad = subprocess.run(["bash", str(LAUNCHER), "6", "--dry-run"], env=env, text=True, capture_output=True)
            self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
