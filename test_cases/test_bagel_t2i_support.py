# ruff: noqa: E402, I001
import os
import sys
import tempfile
import types
import unittest

import torch
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

lightx2v_pkg = types.ModuleType("lightx2v")
lightx2v_pkg.__path__ = [os.path.join(REPO_ROOT, "lightx2v")]
sys.modules.setdefault("lightx2v", lightx2v_pkg)

import lightx2v_platform.base.global_var as global_var

if global_var.AI_DEVICE is None:
    global_var.AI_DEVICE = "cuda"

from lightx2v.models.networks.bagel.infer import transformer_infer
from lightx2v.models.networks.bagel.model import BagelModel
from lightx2v.models.networks.bagel.vision import build_bagel_vit_config, extract_bagel_vit_state_dict
from lightx2v.models.runners.bagel.i2i_utils import resolve_bagel_i2i_image_shape
from lightx2v.models.runners.bagel.t2i_utils import BAGEL_T2I_ASPECT_RATIOS, resolve_bagel_t2i_image_shape, validate_bagel_model_assets
from lightx2v.models.schedulers.bagel.scheduler import BagelScheduler
from lightx2v.utils.input_info import I2IInputInfo, T2IInputInfo
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


def make_bagel_i2i_config(model_path="."):
    config = make_bagel_config(model_path=model_path)
    config.update(
        {
            "task": "i2i",
            "visual_und": True,
            "vit_max_num_patch_per_side": 70,
            "connector_act": "gelu_pytorch_tanh",
            "vit_config": {
                "hidden_size": 1152,
                "image_size": 980,
                "intermediate_size": 4304,
                "num_attention_heads": 16,
                "num_hidden_layers": 27,
                "patch_size": 14,
                "num_channels": 3,
            },
        }
    )
    return config


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

    def test_i2i_target_shape_overrides_input_shape(self):
        input_info = I2IInputInfo(target_shape=[576, 1024], aspect_ratio="1:1")
        self.assertEqual(resolve_bagel_i2i_image_shape(input_info, make_bagel_i2i_config(), (640, 480)), (576, 1024))

    def test_i2i_auto_shape_preserves_input_ratio_and_aligns(self):
        self.assertEqual(resolve_bagel_i2i_image_shape(I2IInputInfo(), make_bagel_i2i_config(), (2000, 1000)), (512, 1024))
        self.assertEqual(resolve_bagel_i2i_image_shape(I2IInputInfo(aspect_ratio="16:9"), make_bagel_i2i_config(), (641, 481)), (480, 640))

    def test_i2i_invalid_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "divisible by latent downsample"):
            resolve_bagel_i2i_image_shape(I2IInputInfo(target_shape=[577, 1024]), make_bagel_i2i_config(), (640, 480))

    def test_i2i_missing_config_error_is_clear(self):
        config = make_bagel_config()
        config["task"] = "i2i"
        with self.assertRaisesRegex(ValueError, "vit_config"):
            validate_bagel_model_assets(config, ".")

    def test_missing_vit_weights_error_is_clear(self):
        with self.assertRaisesRegex(ValueError, "vit_model"):
            extract_bagel_vit_state_dict({})

    def test_vit_config_uses_checkpoint_layer_count(self):
        weight_dict = {
            "vit_model.vision_model.encoder.layers.0.layer_norm1.weight": torch.empty(1),
            "vit_model.vision_model.encoder.layers.25.layer_norm1.weight": torch.empty(1),
        }
        vit_config = build_bagel_vit_config(make_bagel_i2i_config(), weight_dict=weight_dict)
        self.assertEqual(vit_config.num_hidden_layers, 26)

    def test_gen_text_raises_not_implemented(self):
        model = BagelModel.__new__(BagelModel)
        with self.assertRaisesRegex(NotImplementedError, "gen_text"):
            model.gen_text(None)

    def test_i2i_context_roles_match_mvp_plan(self):
        model = BagelModel.__new__(BagelModel)
        model.config = {"task": "i2i"}
        model.think = False
        model.understanding_output = False
        model.do_sample = False
        model.text_temperature = 0.3
        model.enable_taylorseer = False
        model.new_token_ids = {"start_of_image": 1, "end_of_image": 2}
        model.transformer_infer = types.SimpleNamespace()

        def init_gen_context():
            return {"images": [], "texts": [], "past_key_values": "pkv", "kv_lens": [0], "ropes": [0]}

        def update_context_text(text, context):
            context = dict(context)
            context["texts"] = list(context["texts"]) + [text]
            return context

        def update_context_image(image, context, vae_model, vae=True, vit=True):
            context = dict(context)
            context["images"] = list(context["images"]) + [image.size]
            return context

        class DummyScheduler:
            infer_steps = 4

            def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids, seed):
                return {"packed_init_noises": torch.zeros(1), "image_sizes": image_sizes}

            def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
                return {"cfg_image_sizes": image_sizes}

        model.init_gen_context = init_gen_context
        model.update_context_text = update_context_text
        model.update_context_image = update_context_image
        model.set_scheduler = lambda scheduler: None

        input_info = I2IInputInfo(
            seed=42,
            prompt="make it blue",
            image_shapes=[128, 128],
        )
        input_info.input_image = Image.new("RGB", (128, 128), "white")

        bagel_inputs, scheduler = model.prepare_inputs(input_info, DummyScheduler(), vae_model=object())

        self.assertEqual(bagel_inputs.gen_context["images"], [(128, 128)])
        self.assertEqual(bagel_inputs.gen_context["texts"], ["make it blue"])
        self.assertEqual(bagel_inputs.cfg_text_precontext["images"], [(128, 128)])
        self.assertEqual(bagel_inputs.cfg_text_precontext["texts"], [])
        self.assertEqual(bagel_inputs.cfg_img_precontext["images"], [])
        self.assertEqual(bagel_inputs.cfg_img_precontext["texts"], ["make it blue"])
        self.assertEqual(scheduler.latents.shape, torch.Size([1]))


if __name__ == "__main__":
    unittest.main()
