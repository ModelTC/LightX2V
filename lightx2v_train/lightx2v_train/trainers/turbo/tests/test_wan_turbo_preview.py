"""Wan local smoke recipe and deterministic periodic preview contract."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lightx2v_train.infer.turbo_video import WanTurboInferencer
from lightx2v_train.runtime.config import load_config
from lightx2v_train.trainers.turbo.config import TurboConfig

TRAIN_ROOT = Path(__file__).resolve().parents[4]
RECIPE = TRAIN_ROOT / "configs/train/turbo/wan2_1_t2v_1_3b_turbo8_fp32_gpu0_smoke.yaml"


class WanTurboPreviewTests(unittest.TestCase):
    def config(self, root):
        with patch.dict(os.environ, {"WAN_TURBO_ROOT": str(root)}):
            return load_config(str(RECIPE))

    def test_recipe_uses_online_fp32_components_and_100_step_previews(self):
        config = self.config("/experiment")
        turbo = TurboConfig.from_mapping(config)
        self.assertEqual(turbo.total_steps, 300)
        self.assertEqual((turbo.dcm_steps, turbo.scm_steps, turbo.joint_steps), (100, 50, 150))
        self.assertEqual(turbo.generation_shapes, [{"value": [81, 480, 832]}])
        self.assertEqual(config["data"]["train"]["name"], "video_dataset")
        self.assertEqual(config["data"]["val"]["name"], "prompt_dataset")
        self.assertEqual(config["training"]["teacher"]["guidance_scale"], 5.0)
        self.assertFalse(config["training"]["allow_tf32"])
        self.assertEqual(config["inference"]["infer_every_iters"], 100)
        self.assertFalse(config["inference"]["teacher_baseline_at_zero"])
        self.assertFalse(config["distributed"]["fsdp2"]["enabled"])
        self.assertTrue(config["model"]["load_text_encoder"])
        self.assertTrue(config["model"]["load_vae"])
        self.assertFalse(config["model"]["t5_cpu"])
        for key in ("running_dtype", "transformer_param_dtype", "vae_dtype", "t5_dtype"):
            self.assertEqual(config["model"][key], "fp32")

    def test_preview_keeps_rng_and_writes_four_fixed_seed_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            preview = WanTurboInferencer(config)
            prompt_calls, sampled = [], []

            def condition(prompt):
                prompt_calls.append(prompt)
                return {"prompt_embed": torch.ones(1, 2, 2)}

            model = SimpleNamespace(
                device=torch.device("cpu"),
                running_dtype=torch.float32,
                vae_dtype=torch.float32,
                ensure_capabilities=lambda: SimpleNamespace(require=lambda kind: object()),
                set_denoiser_eval=lambda: None,
                encode_prompt_condition=condition,
                prepare_infer_latents=lambda height, width: torch.randn(1, 1, 2, 2, 2),
                decode_latent=lambda value: [value],
            )
            preview.set_model(model)
            preview.set_data(SimpleNamespace(dataset=SimpleNamespace(samples=[{"prompt": f"case {i}"} for i in range(4)])))
            preview.output_infer_dir = str(Path(directory) / "infer/iter-000000100")

            def sample(capability, initial, positive, **options):
                sampled.append((initial.clone(), options))
                return initial + torch.randn_like(initial)

            def export(video, destination, **options):
                Path(destination).write_bytes(b"test preview placeholder")

            torch.manual_seed(127)
            before = torch.get_rng_state().clone()
            with patch("lightx2v_train.infer.turbo_video.sample_turbo_latents", side_effect=sample), patch("lightx2v_train.infer.turbo_video.export_to_video", side_effect=export):
                files = preview.infer()
                torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
                initial = [item[0] for item in sampled]
                sampled.clear()
                preview.infer()
            for value, repeated in zip(initial, sampled, strict=True):
                torch.testing.assert_close(value, repeated[0], rtol=0, atol=0)
                self.assertEqual(repeated[1]["num_steps"], 8)
                self.assertEqual(repeated[1]["video_shift"], 5.0)
            self.assertEqual(len(files), 4)
            self.assertEqual(prompt_calls, [f"case {i}" for i in range(4)] * 2)
            manifest = json.loads((Path(preview.output_infer_dir) / "manifest.json").read_text())
            self.assertEqual([record["seed"] for record in manifest], [42, 43, 44, 45])
            self.assertTrue(all(record["weights"] == "student_ema" and record["guidance_scale"] == 1.0 for record in manifest))
            self.assertTrue((Path(preview.output_infer_dir) / "index.html").is_file())

    def test_fsdp_recipe_uses_two_fp32_ranks_and_reuses_raw_selection(self):
        recipe = RECIPE.with_name("wan2_1_t2v_1_3b_turbo8_fp32_fsdp01_smoke.yaml")
        with patch.dict(os.environ, {"WAN_TURBO_ROOT": "/dual-output", "WAN_TURBO_DATA_ROOT": "/raw-selection"}):
            config = load_config(str(recipe))
        self.assertEqual(TurboConfig.from_mapping(config).total_steps, 300)
        fsdp = config["distributed"]["fsdp2"]
        self.assertTrue(fsdp["enabled"])
        self.assertEqual(fsdp["size"], 2)
        self.assertEqual(fsdp["mixed_precision"], {"param_dtype": "fp32", "reduce_dtype": "fp32", "output_dtype": "fp32", "cast_forward_inputs": False})
        self.assertFalse(config["inference"]["run_at_start"])
        self.assertEqual(config["inference"]["infer_every_iters"], 100)
        self.assertEqual(config["data"]["train"]["data_path"], ["/raw-selection/selection/train.jsonl"])
        self.assertEqual(config["training"]["output_dir"], "/dual-output/train")

    def test_fsdp_nonwriter_rank_still_executes_all_prompt_forwards(self):
        with tempfile.TemporaryDirectory() as directory:
            preview = WanTurboInferencer(self.config(directory))
            decode_calls = []
            preview.set_model(
                SimpleNamespace(
                    device=torch.device("cpu"),
                    running_dtype=torch.float32,
                    vae_dtype=torch.float32,
                    ensure_capabilities=lambda: SimpleNamespace(require=lambda kind: object()),
                    set_denoiser_eval=lambda: None,
                    encode_prompt_condition=lambda prompt: {"prompt_embed": torch.ones(1, 2, 2)},
                    prepare_infer_latents=lambda height, width: torch.randn(1, 1, 2, 2, 2),
                    decode_latent=lambda value: decode_calls.append(value) or [value],
                )
            )
            preview.set_data(SimpleNamespace(dataset=SimpleNamespace(samples=[{"prompt": f"case {i}"} for i in range(4)])))
            preview.output_infer_dir = str(Path(directory) / "infer/iter-000000100")
            with (
                patch("lightx2v_train.infer.turbo_video.is_main_process", return_value=False),
                patch("lightx2v_train.infer.turbo_video.get_world_size", return_value=2),
                patch("lightx2v_train.infer.turbo_video.sample_turbo_latents", side_effect=lambda capability, noise, condition, **options: noise) as sample,
                patch("lightx2v_train.infer.turbo_video.export_to_video") as export,
            ):
                self.assertEqual(preview.infer(), [])
            self.assertEqual(sample.call_count, 4)
            self.assertEqual(len(decode_calls), 4)
            export.assert_not_called()
            self.assertFalse((Path(preview.output_infer_dir) / "manifest.json").exists())
            self.assertFalse((Path(preview.output_infer_dir) / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
