"""PDD head layout, native H3 conventions, fusion and adapter persistence."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from lightx2v_train.model_capabilities import DistributionMatchingCapability
from lightx2v_train.model_zoo.minimax_h3.tests import test_ref2av_capability as ref_fixtures
from lightx2v_train.model_zoo.minimax_h3.tests.test_ref2av_capability import model_wrapper, reference_condition
from lightx2v_train.model_zoo.minimax_h3.tests.test_turbo_jvp import tiny_setup
from lightx2v_train.model_zoo.minimax_h3_image.model import MiniMaxH3ImageModel
from lightx2v_train.model_zoo.minimax_h3_image.tests.test_model import model_config, save_tiny_model
from lightx2v_train.model_zoo.native.minimax_h3.modeling import _transformer_class
from lightx2v_train.model_zoo.pdd import ParallelDecodeLinear
from safetensors.torch import save_file


def image_fixture(directory):
    original = save_tiny_model(directory)
    native = Path(directory) / "transformer"
    native.mkdir()
    values = {f"final_layer.video_out.{name}": value for name, value in original.pdd_proj_out[2].state_dict().items()}
    save_file(values, str(native / "model-00001-of-00001.safetensors"))
    (native / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in values}}))
    config = model_config(directory)
    config["model"]["image_head_mode"] = "base"
    config["training"]["method"] = "pdd"
    return config


class ParallelLinearTests(unittest.TestCase):
    def test_copy_layout_fusion_and_selected_head_gradient(self):
        torch.manual_seed(9)
        original = torch.nn.Linear(5, 6)
        bank = ParallelDecodeLinear.from_linear(original, 4)
        value = torch.randn(2, 3, 5)
        prediction = bank(value).unflatten(-1, (4, 6))
        for head in range(4):
            torch.testing.assert_close(prediction[:, :, head], original(value), rtol=0, atol=0)
        with torch.no_grad():
            bank.weight.add_(torch.randn_like(bank.weight) * 0.1)
        weights = torch.tensor([0.0, 0.25, 0.75, 0.0], dtype=torch.float64)
        prediction = bank(value).unflatten(-1, (4, 6))
        with bank.fuse(1, 3, weights):
            fused = bank(value)
        torch.testing.assert_close(fused, (prediction * weights.float()[None, None, :, None]).sum(2), rtol=2e-6, atol=2e-6)
        prediction[:, :, 2].square().mean().backward()
        gradient = bank.weight.grad.unflatten(0, (4, 6))
        self.assertGreater(float(gradient[2].abs().sum()), 0)
        self.assertEqual(float(gradient[[0, 1, 3]].abs().sum()), 0)


class H3PDDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def check_av(self, model, state, condition):
        model.config["training"] = {"method": "pdd", "pdd": {"video_shift": 12, "audio_shift": 3}}
        sigma = torch.tensor(0.7, dtype=torch.float64)
        teacher = model.predict_pdd_teacher(state, sigma, condition)
        model.add_lora(2, 2, ["to_q", "to_k", "to_v", "to_out.0"])
        model.set_lora_trainable()
        model.enable_pdd(4)
        model.set_lora_trainable()  # Resume must retain new full-head trainability.
        prediction = model.predict_pdd_heads(state, 0, condition, sigma)
        for name in ("video", "audio"):
            values = getattr(prediction, name)
            self.assertEqual(values.shape, (1, 4, *getattr(state, name).shape[1:]))
            for head in range(4):
                torch.testing.assert_close(values[:, head], getattr(teacher, name), rtol=2e-5, atol=2e-5)
        grid = torch.tensor([0.7, 0.5, 0.4, 0.1, 0.0], dtype=torch.float64)
        schedules = model.pdd_sigma_schedule(grid)
        self.assertEqual(schedules["audio"].dtype, torch.float64)
        torch.testing.assert_close(schedules["audio"], grid / (4 - 3 * grid))
        displacement = model.predict_pdd_block(state, 0, 3, condition, grid)
        for name in ("video", "audio"):
            weights = schedules[name].diff()[:3].to(getattr(prediction, name))
            explicit = (getattr(prediction, name)[:, :3] * weights[None, :, None, None]).sum(1)
            torch.testing.assert_close(getattr(displacement, name), explicit, rtol=2e-5, atol=2e-5)
        (prediction.video[:, 1].square().mean() + prediction.audio[:, 2].square().mean()).backward()
        for name in model.pdd_auxiliary_parameter_names():
            parameter = model.transformer.get_parameter(name)
            self.assertTrue(parameter.requires_grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        with tempfile.TemporaryDirectory() as directory:
            expected = {name: model.transformer.get_parameter(name).detach().clone() for name in model.pdd_auxiliary_parameter_names()}
            model.save_pdd_weights(directory)
            with torch.no_grad():
                for name in expected:
                    model.transformer.get_parameter(name).zero_()
            model.load_pdd_weights(directory)
            for name, value in expected.items():
                torch.testing.assert_close(model.transformer.get_parameter(name), value, rtol=0, atol=0)

    def test_t2av_heads_fusion_and_save(self):
        model, _, state, _, condition = tiny_setup()
        self.check_av(model, state, condition)

    def test_ref2av_keeps_reference_rows_outside_head_outputs(self):
        model = model_wrapper()
        model.running_dtype = torch.float32
        model.text_dim = 12
        model.video_latent_channels = model.audio_latent_channels = 2
        model.transformer = _transformer_class()(
            num_attention_heads=2,
            attention_head_dim=8,
            hidden_size=16,
            num_layers=1,
            num_refiner_layers=1,
            ffn_dim=32,
            in_channels=2,
            audio_in_channels=2,
            patch_size=(1, 2, 2),
            text_dim=12,
            freq_dim=4,
            time_embed_hidden_dim=16,
            time_embed_dim=8,
            rope_freq_dim=1,
        )
        cached = reference_condition()
        cached["prompt_embeds"] = torch.randn(1, 4, 12)
        cached["references"][0]["video_latents"] = torch.randn(6, 8)
        cached["references"][1]["audio_latents"] = torch.randn(4, 2)
        capability = model.ensure_capabilities().require(DistributionMatchingCapability)
        sample, condition = ref_fixtures.Ref2AVCapabilityTest().prepare(capability, cached)
        shape = capability.latent_shape(sample, [{"value": [124, 64, 96]}], lambda value: value)
        state = capability.initial_latents(shape, torch.float32, lambda value: value)
        frozen = {name: condition[name].clone() for name in ("noised_condition_video_latents", "condition_audio_latents")}
        self.check_av(model, state, condition)
        for name, value in frozen.items():
            torch.testing.assert_close(condition[name], value, rtol=0, atol=0)

    def test_image_retained_base_head_then_widening(self):
        with tempfile.TemporaryDirectory() as directory:
            config = image_fixture(directory)
            with patch("torch.cuda.is_available", return_value=False):
                model = MiniMaxH3ImageModel(config)
                model.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
            condition = model.prepare_text_condition({"prompt_embeds": torch.randn(1, 3, 12), "text_token_tags": torch.ones(3, dtype=torch.long)})
            state = torch.randn(1, 2, 1, 4, 4)
            teacher = model.predict_pdd_teacher(state, torch.tensor(0.9), condition)
            model.add_lora(2, 2, None)
            model.set_lora_trainable()
            model.enable_pdd(4)
            heads = model.predict_pdd_heads(state, 0, condition, torch.tensor(0.9))
            self.assertEqual(heads.shape, (1, 4, 2, 1, 4, 4))
            for index in range(4):
                torch.testing.assert_close(heads[:, index], teacher)
            grid = torch.tensor([0.9, 0.5, 0.3, 0.1, 0.0], dtype=torch.float64)
            fused = model.predict_pdd_block(state, 0, 2, condition, grid)
            explicit = heads[:, 0] * float(grid[1] - grid[0]) + heads[:, 1] * float(grid[2] - grid[1])
            torch.testing.assert_close(fused, explicit, rtol=2e-5, atol=2e-5)
            heads[:, 1].square().mean().backward()
            self.assertTrue(torch.isfinite(model.transformer.proj_out.weight.grad).all())
            with self.assertRaises(ValueError):
                model.predict_pdd_block(state, 2, 2, condition, grid)


if __name__ == "__main__":
    unittest.main()
