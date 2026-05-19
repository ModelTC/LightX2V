# ruff: noqa: E402, I001
import os
import sys
import tempfile
import types
import unittest

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

lightx2v_pkg = types.ModuleType("lightx2v")
lightx2v_pkg.__path__ = [os.path.join(REPO_ROOT, "lightx2v")]
sys.modules.setdefault("lightx2v", lightx2v_pkg)

from lightx2v.models.networks.bagel.infer import transformer_infer
from lightx2v.models.runners.bagel.t2i_utils import BAGEL_T2I_ASPECT_RATIOS, resolve_bagel_t2i_image_shape, validate_bagel_model_assets
from lightx2v.models.schedulers.bagel.scheduler import BagelScheduler
from lightx2v.utils.input_info import T2IInputInfo
from lightx2v.utils.lockable_dict import LockableDict


def make_bagel_config(model_path="."):
    return LockableDict(
        {
            "model_path": model_path,
            "interpolate_pos": False,
            "latent_patch_size": 2,
            "max_latent_size_update": 64,
            "vae_config": {"downsample": 8, "z_channels": 16},
            "infer_steps": 4,
            "inference_hyper": {
                "cfg_text_scale": 4.0,
                "cfg_img_scale": 1.0,
                "cfg_interval": [0.4, 1.0],
                "timestep_shift": 3.0,
                "cfg_renorm_min": 0.0,
                "cfg_renorm_type": "global",
            },
            "llm_config": {
                "num_hidden_layers": 1,
                "layer_module": "Qwen2MoTDecoderLayer",
                "hidden_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "max_position_embeddings": 2048,
                "rope_scaling": None,
            },
            "llm_config_update": {},
            "visual_gen": True,
        }
    )


class BagelT2ISupportTest(unittest.TestCase):
    def test_default_aspect_ratio(self):
        self.assertEqual(resolve_bagel_t2i_image_shape(T2IInputInfo(), make_bagel_config()), (1024, 1024))

    def test_official_aspect_ratios(self):
        for aspect_ratio, expected_shape in BAGEL_T2I_ASPECT_RATIOS.items():
            with self.subTest(aspect_ratio=aspect_ratio):
                input_info = T2IInputInfo(aspect_ratio=aspect_ratio)
                self.assertEqual(resolve_bagel_t2i_image_shape(input_info, make_bagel_config()), expected_shape)

    def test_target_shape_overrides_aspect_ratio(self):
        input_info = T2IInputInfo(aspect_ratio="16:9", target_shape=[1024, 1024])
        self.assertEqual(resolve_bagel_t2i_image_shape(input_info, make_bagel_config()), (1024, 1024))

    def test_invalid_target_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "divisible by latent downsample"):
            resolve_bagel_t2i_image_shape(T2IInputInfo(target_shape=[577, 1024]), make_bagel_config())

        with self.assertRaisesRegex(ValueError, "must be \\[H W\\]"):
            resolve_bagel_t2i_image_shape(T2IInputInfo(target_shape=[1024]), make_bagel_config())

    def test_invalid_aspect_ratio_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported BAGEL aspect_ratio"):
            resolve_bagel_t2i_image_shape(T2IInputInfo(aspect_ratio="2:1"), make_bagel_config())

    def test_seed_controls_initial_noise(self):
        scheduler = BagelScheduler(make_bagel_config())
        kwargs = {
            "curr_kvlens": [0],
            "curr_rope": [0],
            "image_sizes": [(16, 16)],
            "new_token_ids": {"start_of_image": 1, "end_of_image": 2},
        }
        noise_a = scheduler.prepare_vae_latent(**kwargs, seed=123)["packed_init_noises"]
        noise_b = scheduler.prepare_vae_latent(**kwargs, seed=123)["packed_init_noises"]
        noise_c = scheduler.prepare_vae_latent(**kwargs, seed=124)["packed_init_noises"]

        self.assertTrue(torch.equal(noise_a, noise_b))
        self.assertFalse(torch.equal(noise_a, noise_c))

    def test_missing_flash_attn_error_is_clear(self):
        original_flash_attn_varlen_func = transformer_infer.flash_attn_varlen_func
        transformer_infer.flash_attn_varlen_func = None
        try:
            with self.assertRaisesRegex(ImportError, "flash-attn"):
                transformer_infer.BagelTransformerInfer({}, {})
        finally:
            transformer_infer.flash_attn_varlen_func = original_flash_attn_varlen_func

    def test_missing_model_weights_error_is_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError) as ctx:
                validate_bagel_model_assets(make_bagel_config(model_path=tmpdir), tmpdir)
        message = str(ctx.exception)
        self.assertIn("ema.safetensors", message)
        self.assertIn("ae.safetensors", message)


if __name__ == "__main__":
    unittest.main()
