"""The two-stage experiment must keep its budget and weight boundary explicit."""

import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from lightx2v_train.infer.wan_official_pdmd import WanOfficialPdmdInferencer
from lightx2v_train.runtime.config import load_config
from lightx2v_train.trainers.dmd.official_pdmd_core import critic_updates_before
from lightx2v_train.utils.registry import build_inferencer

TRAIN_ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("wan_dcm_pdmd_stages", TRAIN_ROOT / "scripts/run_wan21_dcm_pdmd_stages.py")
STAGES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGES)


class ExperimentTests(unittest.TestCase):
    def configs(self, directory):
        with patch.dict(os.environ, {"WAN_DCM_PDMD_ROOT": directory, "CUDA_VISIBLE_DEVICES": "0,1"}):
            dcm = load_config(str(TRAIN_ROOT / "configs/train/turbo/wan2_1_t2v_1_3b_dcm100_fp32_fsdp01.yaml"))
            pdmd = load_config(str(TRAIN_ROOT / "configs/train/dmd/wan2_1_t2v_1_3b_official_pdmd200_from_dcm_fp32_fsdp01.yaml"))
        return dcm, pdmd

    def test_full_budget_live_student_and_no_video_encoding_in_pdmd(self):
        with tempfile.TemporaryDirectory() as directory:
            dcm, pdmd = self.configs(directory)
            with patch.object(Path, "is_dir", return_value=True), patch.object(Path, "is_file", return_value=True):
                checkpoint = STAGES.preflight(dcm, pdmd)
            total = pdmd["training"]["max_train_iters"]
            critic = critic_updates_before(total, pdmd["training"]["dmd"]["fake_update_ratio"])
            self.assertEqual((total, total - critic, critic), (200, 33, 167))
            self.assertEqual(pdmd["training"]["student"]["checkpoint_path"], str(checkpoint / "student_ema"))
            self.assertFalse(pdmd["training"]["student"]["ema"]["enabled"])
            self.assertEqual(pdmd["data"]["train"]["name"], "prompt_dataset")
            self.assertEqual(pdmd["inference"]["infer_every_iters"], 100)
            self.assertEqual(pdmd["training"]["save_every_iters"], 100)
            self.assertEqual(dcm["training"]["turbo"]["scm_steps"], 0)
            self.assertEqual(dcm["training"]["turbo"]["joint_steps"], 0)
            self.assertIsInstance(build_inferencer(pdmd), WanOfficialPdmdInferencer)

    def test_preview_uses_all_euler_states_without_stochastic_renoising(self):
        _, pdmd = self.configs("/experiment")
        preview = WanOfficialPdmdInferencer(pdmd)
        sigmas = []

        class NativeWan:
            def set_training(self, enabled):
                pass

            def predict_velocity(self, state, sigma, condition):
                sigmas.append(torch.as_tensor(sigma).clone())
                # Native Wan v=noise-clean; exact constant field ends at x-2.
                return torch.ones_like(state) * 2

        noise = torch.tensor([[[3.0, 7.0]]])
        with patch("lightx2v_train.infer.turbo_video.sample_turbo_latents", side_effect=AssertionError("PDMD must not use CM re-noising")):
            result = preview._sample(NativeWan(), noise, {}, 8, 5)
        torch.testing.assert_close(result, noise - 2)
        self.assertEqual(len(sigmas), 8)
        self.assertEqual(float(sigmas[0].flatten()[0]), 1.0)

    def test_failed_dcm_process_never_starts_pdmd(self):
        import subprocess

        with tempfile.TemporaryDirectory() as directory:
            dcm, pdmd = self.configs(directory)
            error = subprocess.CalledProcessError(1, "dcm")
            with (
                patch("sys.argv", ["stages", "--dcm-config", "dcm.yaml", "--pdmd-config", "pdmd.yaml"]),
                patch.object(STAGES, "load_config", side_effect=[dcm, pdmd]),
                patch.object(STAGES, "preflight", return_value=Path(directory) / "checkpoint-000000100"),
                patch.object(STAGES.subprocess, "run", side_effect=error) as runner,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    STAGES.main()
            self.assertEqual(runner.call_count, 1)

    def test_stage_boundary_refreshes_budget_without_restarting_completed_dcm(self):
        with tempfile.TemporaryDirectory() as directory:
            dcm, pdmd = self.configs(directory)
            original_pdmd = copy.deepcopy(pdmd)
            original_pdmd["training"]["max_train_iters"] = 1200
            original_pdmd["inference"]["infer_every_iters"] = 600
            checkpoint = Path(dcm["training"]["output_dir"]) / "checkpoint-000000100"
            (checkpoint / "student_ema").mkdir(parents=True)
            (checkpoint / "student_ema/pytorch_lora_weights.safetensors").write_bytes(b"test exported adapter")
            (checkpoint / "turbo.complete.json").write_text(json.dumps({"iteration": 100}))
            with (
                patch("sys.argv", ["stages", "--dcm-config", "dcm.yaml", "--pdmd-config", "pdmd.yaml"]),
                patch.object(STAGES, "load_config", side_effect=[dcm, original_pdmd, dcm, pdmd]),
                patch.object(STAGES, "preflight", return_value=checkpoint),
                patch.object(STAGES.subprocess, "run") as runner,
            ):
                STAGES.main()
            self.assertEqual(runner.call_count, 2)
            self.assertEqual(runner.call_args_list[0].args[0][-1], "dcm.yaml")
            self.assertEqual(runner.call_args_list[1].args[0][-1], "pdmd.yaml")
            transition = json.loads((Path(directory) / "stage_transition.json").read_text())
            self.assertEqual((transition["pdmd_optimizer_updates"], transition["pdmd_student_updates"], transition["pdmd_fake_updates"]), (200, 33, 167))
            self.assertEqual(transition["pdmd_infer_every_role_updates"], 100)


if __name__ == "__main__":
    unittest.main()
