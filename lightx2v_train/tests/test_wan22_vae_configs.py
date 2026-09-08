"""Wan22 dataset geometry and frame contracts for grouped Wan2.1 VAE training."""

import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from lightx2v_train.utils.registry import build_sample_processor


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs/train/vae"
COMPONENTS = ("encoder", "decoder")
PHASES = ("search", "recover")
DEFAULT_METADATA = "/mnt/lm_data_afs/wuzhuguanyu/LightX2V_train/lightx2v_train/data/wan22/metadata_vae.jsonl"
DEFAULT_WEIGHTS = "/mnt/devsft_afs_1/gushiqiao/Wan2.1_VAE.pth"


def config_name(component, phase, *, wan22=True):
    suffix = "_wan22_8gpu" if wan22 else ""
    return f"wan21_{component}_prune_{phase}_grouped_keep5{suffix}_ddp"


def load_config(component, phase, *, env=None, wan22=True):
    source = CONFIG_DIR / f"{config_name(component, phase, wan22=wan22)}.yaml"
    with patch.dict(os.environ, env or {}, clear=True):
        return OmegaConf.to_container(OmegaConf.load(source), resolve=True)


class Wan22VAEConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_paths_and_dataset_geometry(self):
        outputs = set()
        for component in COMPONENTS:
            for phase in PHASES:
                with self.subTest(component=component, phase=phase):
                    config = load_config(component, phase)
                    self.assertEqual(config["model"]["pretrained_model_name_or_path"], DEFAULT_WEIGHTS)
                    output = f"output_train/{config_name(component, phase)}"
                    self.assertEqual(config["training"]["output_dir"], output)
                    self.assertEqual(config["inference"]["output_dir"], f"{output}/previews")
                    outputs.add(output)
                    if phase == "search":
                        self.assertEqual(config["training"]["pruning"]["export_dir"], f"{output}/export")
                    else:
                        selection = config["model"][f"pruned_{component}"]["selection_path"]
                        self.assertEqual(selection, f"output_train/{config_name(component, 'search')}/export/kept_layers.json")
                    for split in ("train", "val"):
                        data = config["data"][split]
                        self.assertEqual(data["data_path"], DEFAULT_METADATA)
                        self.assertEqual((data["height"], data["width"], data["num_frames"]), (480, 832, 81))
                        self.assertEqual(data["frame_rate"], 16)
                        self.assertFalse(data["fix_frame_rate"])
                        self.assertFalse(data["geometry_from_metadata"])
                        self.assertFalse(data["skip_missing"])
                        self.assertEqual((data["time_division_factor"], data["time_division_remainder"]), (4, 1))
                    self.assertEqual(config["inference"]["fps"], 16)
        self.assertEqual(len(outputs), 4)

    def test_environment_overrides_keep_search_and_recovery_linked(self):
        for component in COMPONENTS:
            prefix = f"WAN_VAE_{component.upper()}_WAN22"
            env = {
                "WAN_VAE_GROUPED_PATH": "/override/Wan2.1_VAE.pth",
                "WAN_VAE_METADATA": "/override/metadata_vae.jsonl",
                f"{prefix}_SEARCH_OUTPUT": f"/override/{component}_search",
                f"{prefix}_RECOVER_OUTPUT": f"/override/{component}_recover",
            }
            for phase in PHASES:
                with self.subTest(component=component, phase=phase):
                    config = load_config(component, phase, env=env)
                    self.assertEqual(config["model"]["pretrained_model_name_or_path"], env["WAN_VAE_GROUPED_PATH"])
                    self.assertEqual(config["training"]["output_dir"], env[f"{prefix}_{phase.upper()}_OUTPUT"])
                    for split in ("train", "val"):
                        self.assertEqual(config["data"][split]["data_path"], env["WAN_VAE_METADATA"])
                    if phase == "recover":
                        self.assertEqual(
                            config["model"][f"pruned_{component}"]["selection_path"],
                            f"{env[f'{prefix}_SEARCH_OUTPUT']}/export/kept_layers.json",
                        )
            env[f"{prefix}_SELECTION"] = f"/override/{component}_selection.json"
            config = load_config(component, "recover", env=env)
            self.assertEqual(config["model"][f"pruned_{component}"]["selection_path"], env[f"{prefix}_SELECTION"])

    def test_training_schedule_and_loss_settings_are_preserved(self):
        for component in COMPONENTS:
            for phase in PHASES:
                with self.subTest(component=component, phase=phase):
                    config = load_config(component, phase)
                    baseline = load_config(component, phase, wan22=False)
                    training = config["training"]
                    distillation = training["vae_distillation"]
                    expected_crops = [(33, 256)] if phase == "search" else [(33, 256), (81, 384)]
                    self.assertEqual(
                        [(stage["crop_num_frames"], stage["crop_size"]) for stage in distillation["stages"]],
                        expected_crops,
                    )
                    self.assertEqual(distillation["perceptual_num_frames"], 32 if phase == "search" else 64)
                    self.assertEqual(training["max_train_iters"], 1000 if phase == "search" else 3000)
                    self.assertEqual(training["save_every_iters"], 100)
                    self.assertEqual(config["inference"]["infer_every_iters"], 100)
                    if phase == "recover":
                        self.assertEqual([spec["max_frames"] for spec in distillation["gan"]["crop_specs"]], [32, 32])

                    expected_training = copy.deepcopy(baseline["training"])
                    expected_training["output_dir"] = training["output_dir"]
                    if phase == "search":
                        expected_training["pruning"]["export_dir"] = training["pruning"]["export_dir"]
                    else:
                        expected_training["vae_distillation"]["stages"][1]["crop_num_frames"] = 81
                    self.assertEqual(training, expected_training)

    def test_processor_accepts_81_frames_without_padding_and_rejects_65(self):
        for component in COMPONENTS:
            for phase in PHASES:
                with self.subTest(component=component, phase=phase):
                    processor = build_sample_processor(load_config(component, phase))
                    self.assertEqual(processor.min_source_frames, 81)
                    self.assertFalse(processor.load_cached_latents)
                    pixels = torch.linspace(-1, 1, 3 * 81 * 8 * 16).reshape(3, 81, 8, 16)
                    output = processor({"inputs": {"video": pixels}, "meta": {}})
                    self.assertEqual(output["inputs"]["video"].shape, (3, 81, 8, 16))
                    self.assertEqual(output["meta"]["source_num_frames"], 81)
                    self.assertEqual(output["meta"]["num_frames"], 81)
                    torch.testing.assert_close(output["inputs"]["video"], (pixels + 1) * 0.5)
                    with self.assertRaisesRegex(ValueError, "at least 81 real source frames"):
                        processor({"inputs": {"video": torch.zeros(3, 65, 8, 16)}, "meta": {}})

    def test_recovery_processor_minimum_must_cover_81_frame_loss_crop(self):
        for component in COMPONENTS:
            with self.subTest(component=component):
                config = load_config(component, "recover")
                config["data"]["processor"]["min_source_frames"] = 65
                with self.assertRaisesRegex(ValueError, "largest training crop"):
                    build_sample_processor(config)


if __name__ == "__main__":
    unittest.main()
