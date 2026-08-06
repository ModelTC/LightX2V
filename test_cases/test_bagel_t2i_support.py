# ruff: noqa: E402, I001
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

from lightx2v.common.ops.rope import TorchRealRope
from lightx2v.models.networks.bagel.infer.pre_infer import BagelPreInfer
from lightx2v.models.networks.bagel.infer import transformer_infer
from lightx2v.models.networks.bagel.model import GEN_THINK_SYSTEM_PROMPT, BagelModel
from lightx2v.models.networks.bagel.vision import build_bagel_vit_config, extract_bagel_vit_state_dict
from lightx2v.models.runners.bagel.i2i_utils import resolve_bagel_i2i_image_shape, resize_pil_for_vit
from lightx2v.models.runners.bagel.t2i_utils import BAGEL_T2I_ASPECT_RATIOS, resolve_bagel_t2i_image_shape, validate_bagel_model_assets
from lightx2v.models.schedulers.bagel.scheduler import BagelScheduler
from lightx2v.models.video_encoders.hf.bagel import vae as bagel_vae
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
            "rope_type": "torch_real_rope",
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
                "rope_theta": 1000000.0,
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

    def test_prepare_start_tokens_matches_bagel_layout(self):
        model = BagelModel.__new__(BagelModel)
        generation_input = model.prepare_start_tokens(
            curr_kvlens=[3, 2],
            curr_rope=[7, 11],
            new_token_ids={"bos_token_id": 151644},
        )

        self.assertEqual(generation_input["packed_start_tokens"].tolist(), [151644, 151644])
        self.assertEqual(generation_input["packed_query_position_ids"].tolist(), [7, 11])
        self.assertEqual(generation_input["key_values_lens"].tolist(), [3, 2])
        self.assertEqual(generation_input["packed_key_value_indexes"].tolist(), [0, 1, 2, 3, 4])

    def test_generate_text_reuses_x2v_inference_and_stops_at_eos(self):
        class FakeLmHead:
            def __init__(self):
                self.call_index = 0

            def apply(self, hidden_states):
                next_token = [5, 2][self.call_index]
                self.call_index += 1
                logits = torch.full((hidden_states.shape[0], 8), -100.0)
                logits[:, next_token] = 100.0
                return logits

        model = BagelModel.__new__(BagelModel)
        model.pre_weight = SimpleNamespace(lm_head=FakeLmHead())
        model.pre_infer = SimpleNamespace(
            embed_tokens=lambda weights, token_ids: token_ids.float().unsqueeze(-1),
        )
        forward_calls = []

        def forward_inference(**kwargs):
            forward_calls.append(kwargs)
            return kwargs["packed_query_sequence"], kwargs["past_key_values"]

        model.forward_inference = forward_inference
        cache = object()
        generated_tokens = model.generate_text(
            past_key_values=cache,
            packed_key_value_indexes=torch.tensor([0, 1, 2], dtype=torch.long),
            key_values_lens=torch.tensor([3], dtype=torch.int),
            packed_start_tokens=torch.tensor([1], dtype=torch.long),
            packed_query_position_ids=torch.tensor([7], dtype=torch.long),
            max_length=8,
            do_sample=False,
            end_token_id=2,
        )

        self.assertEqual(generated_tokens[:, 0].tolist(), [1, 5])
        self.assertEqual(len(forward_calls), 2)
        self.assertEqual(forward_calls[0]["mode"], "und")
        self.assertTrue(forward_calls[0]["is_causal"])
        self.assertTrue(forward_calls[0]["update_past_key_values"])
        self.assertEqual(forward_calls[0]["packed_query_position_ids"].tolist(), [7])
        self.assertEqual(forward_calls[1]["packed_query_position_ids"].tolist(), [8])
        self.assertIs(forward_calls[0]["past_key_values"], cache)

    def test_gen_text_decodes_official_bagel_markers(self):
        model = BagelModel.__new__(BagelModel)
        model.new_token_ids = {"bos_token_id": 1, "eos_token_id": 2}
        model.tokenizer = SimpleNamespace(
            decode=lambda token_ids: "<|im_start|><think>plan</think><|im_end|>ignored",
        )
        model.prepare_start_tokens = lambda curr_kvlens, curr_rope, new_token_ids: {
            "packed_key_value_indexes": torch.tensor([], dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_start_tokens": torch.tensor([new_token_ids["bos_token_id"]], dtype=torch.long),
            "packed_query_position_ids": torch.tensor(curr_rope, dtype=torch.long),
        }
        model.generate_text = lambda **kwargs: torch.tensor([[1], [7]], dtype=torch.long)

        output = model.gen_text(
            {
                "past_key_values": object(),
                "kv_lens": [0],
                "ropes": [0],
            },
            max_length=16,
            do_sample=False,
            temperature=0.3,
        )

        self.assertEqual(output, "<think>plan</think>")

    def test_cot_uses_configured_limit_and_injects_plan(self):
        model = BagelModel.__new__(BagelModel)
        model.config = {"task": "t2i"}
        model.think = True
        model.understanding_output = False
        model.do_sample = False
        model.text_temperature = 0.3
        model.max_think_token_n = 37
        model.enable_taylorseer = False
        model.new_token_ids = {"start_of_image": 1, "end_of_image": 2}
        model.transformer_infer = types.SimpleNamespace()

        def init_gen_context():
            return {"texts": [], "past_key_values": "pkv", "kv_lens": [0], "ropes": [0]}

        def update_context_text(text, context):
            context = dict(context)
            context["texts"] = list(context["texts"]) + [text]
            context["kv_lens"] = [len(context["texts"])]
            context["ropes"] = [len(context["texts"])]
            return context

        gen_text_calls = []

        def gen_text(context, **kwargs):
            gen_text_calls.append((context, kwargs))
            return "<think>compose the scene</think>"

        class DummyScheduler:
            infer_steps = 4

            def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids, seed):
                return {"packed_init_noises": torch.zeros(1), "image_sizes": image_sizes}

            def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
                return {"cfg_image_sizes": image_sizes}

        model.init_gen_context = init_gen_context
        model.update_context_text = update_context_text
        model.gen_text = gen_text
        model.set_scheduler = lambda scheduler: None

        input_info = T2IInputInfo(
            seed=42,
            prompt="a glass city",
            image_shapes=[128, 128],
        )
        bagel_inputs, _ = model.prepare_inputs(input_info, DummyScheduler())

        self.assertEqual(len(gen_text_calls), 1)
        self.assertEqual(gen_text_calls[0][1]["max_length"], 37)
        self.assertFalse(gen_text_calls[0][1]["do_sample"])
        self.assertEqual(gen_text_calls[0][1]["temperature"], 0.3)
        self.assertEqual(
            bagel_inputs.gen_context["texts"],
            [
                GEN_THINK_SYSTEM_PROMPT,
                "a glass city",
                "<think>compose the scene</think>",
            ],
        )

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

    def test_resize_pil_for_vit_respects_stride_and_max_pixels(self):
        image = Image.new("RGB", (1920, 1080), "white")
        resized = resize_pil_for_vit(image)
        width, height = resized.size

        self.assertEqual(width % 14, 0)
        self.assertEqual(height % 14, 0)
        self.assertLessEqual(width, 980)
        self.assertLessEqual(height, 980)
        self.assertLessEqual(width * height, 14 * 14 * 9 * 1024)


class BagelRopeTest(unittest.TestCase):
    @staticmethod
    def _legacy_qwen2_cos_sin(x, position_ids, head_dim, rope_theta):
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def test_freqs_are_bitwise_equal_to_legacy_qwen2_path(self):
        config = make_bagel_config()
        pre_infer = BagelPreInfer(config, config["llm_config"])
        packed_sequence = torch.randn(7, 16, dtype=torch.bfloat16)
        position_ids = torch.tensor(
            [[0, 1, 2, 31, 127, 511, 1023]],
            dtype=torch.long,
        )

        actual_cos, actual_sin = pre_infer._compute_rope_cos_sin(packed_sequence, position_ids)
        expected_cos, expected_sin = self._legacy_qwen2_cos_sin(
            packed_sequence,
            position_ids,
            head_dim=8,
            rope_theta=1000000.0,
        )

        self.assertTrue(torch.equal(actual_cos, expected_cos))
        self.assertTrue(torch.equal(actual_sin, expected_sin))

    def test_prepare_and_apply_use_x2v_module_without_drift(self):
        config = make_bagel_config()
        pre_infer = BagelPreInfer(config, config["llm_config"])
        rope = TorchRealRope(layout="split_half", compute_dtype=torch.float32)
        pre_infer.set_rope(rope)

        packed_sequence = torch.randn(7, 16, dtype=torch.bfloat16)
        position_ids = torch.tensor([0, 1, 2, 31, 127, 511, 1023], dtype=torch.long)
        packed_rope_freqs, packed_rope_positions = pre_infer.prepare_rope(
            packed_sequence,
            position_ids,
            device="cpu",
        )
        cos, sin = packed_rope_freqs
        self.assertIsNone(packed_rope_positions)
        self.assertEqual(cos.shape, (7, 8))
        self.assertEqual(sin.shape, (7, 8))
        self.assertEqual(cos.dtype, torch.bfloat16)
        self.assertEqual(sin.dtype, torch.bfloat16)

        query = torch.randn(7, 2, 8, dtype=torch.bfloat16)
        key = torch.randn(7, 2, 8, dtype=torch.bfloat16)
        actual_query, actual_key = rope.apply(
            query,
            key,
            packed_rope_freqs,
            rotary_dim=8,
            unsqueeze_dim=1,
        )

        def legacy_apply(x):
            x_float = x.float()
            rotated = torch.cat((-x_float[..., 4:], x_float[..., :4]), dim=-1)
            return (x_float * cos.unsqueeze(1).float() + rotated * sin.unsqueeze(1).float()).to(x.dtype)

        self.assertTrue(torch.equal(actual_query, legacy_apply(query)))
        self.assertTrue(torch.equal(actual_key, legacy_apply(key)))

    def test_model_wires_attention_rope_into_pre_infer(self):
        rope = object()

        class FakePreInfer:
            def __init__(self, config, llm_config):
                self.rope = None

            def set_rope(self, value):
                self.rope = value

        class FakeInfer:
            def __init__(self, config, llm_config):
                pass

        model = BagelModel.__new__(BagelModel)
        model.config = {}
        model.llm_config = {}
        model.pre_infer_class = FakePreInfer
        model.transformer_infer_class = FakeInfer
        model.post_infer_class = FakeInfer
        model.transformer_weights = SimpleNamespace(blocks=[SimpleNamespace(self_attn=SimpleNamespace(rope=rope))])

        model._init_infer()

        self.assertIs(model.pre_infer.rope, rope)

    def test_scaled_rope_fails_with_clear_error(self):
        config = make_bagel_config()
        config["llm_config"]["rope_scaling"] = {"rope_type": "dynamic", "factor": 2.0}
        with self.assertRaisesRegex(NotImplementedError, "only default RoPE"):
            BagelPreInfer(config, config["llm_config"])


class BagelVaeTest(unittest.TestCase):
    def test_vae_owns_fp32_device_boundary_for_encode_and_decode(self):
        class FakeAutoEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(1))
                self.encode_input = None
                self.decode_input = None

            def encode(self, images):
                self.encode_input = images
                return images[:, :1]

            def decode(self, latents):
                self.decode_input = latents
                return torch.zeros(
                    (latents.shape[0], 3, latents.shape[2], latents.shape[3]),
                    device=latents.device,
                    dtype=latents.dtype,
                )

        fake_ae = FakeAutoEncoder()
        with (
            patch.object(bagel_vae.os.path, "exists", return_value=True),
            patch.object(
                bagel_vae,
                "load_ae",
                return_value=(fake_ae, object()),
            ),
            patch.object(bagel_vae, "AI_DEVICE", "cpu"),
        ):
            vae = bagel_vae.BagelVae({"model_path": "."})

        parameter = next(vae.vae_model.parameters())
        self.assertFalse(vae.vae_model.training)
        self.assertEqual(parameter.dtype, torch.float32)
        self.assertEqual(parameter.device.type, "cpu")

        images = torch.ones(1, 3, 2, 2, dtype=torch.bfloat16)
        vae.encode(images)
        self.assertEqual(fake_ae.encode_input.dtype, torch.float32)
        self.assertEqual(fake_ae.encode_input.device.type, "cpu")

        decode_info = {
            "packed_seqlens": torch.tensor([6]),
            "image_shape": (2, 2),
            "latent_downsample": 1,
            "latent_channel": 1,
            "latent_patch_size": 1,
            "return_result_tensor": True,
        }
        vae.decode(
            torch.ones(4, 1, dtype=torch.bfloat16),
            decode_info,
        )
        self.assertEqual(fake_ae.decode_input.dtype, torch.float32)
        self.assertEqual(fake_ae.decode_input.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
