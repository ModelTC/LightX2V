import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from lightx2v_train.model_capabilities import DistributionMatchingCapability
from lightx2v_train.model_zoo.minimax_h3_image.model import MiniMaxH3ImageModel
from lightx2v_train.model_zoo.native.minimax_h3_image import PDDHeadSchedule, load_transformer, packed_t2i_layout, patchify_video_latents, transformer_class, unpatchify_video_latents

SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)


def tiny_transformer():
    return transformer_class()(
        num_attention_heads=2,
        attention_head_dim=8,
        hidden_size=16,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=2,
        patch_size=(1, 2, 2),
        text_dim=12,
        freq_dim=4,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        rope_freq_dim=1,
        num_pdd_heads=6,
    )


def save_tiny_model(directory):
    model = tiny_transformer()
    model.save_pretrained(Path(directory) / "transformer_diffusers", safe_serialization=True, max_shard_size="30KB")
    (Path(directory) / "pdd_config.json").write_text(json.dumps({"num_inference_steps": 6, "sigmas": SIGMAS}))
    return model


def model_config(directory):
    return {
        "model": {
            "name": "minimax_h3_image",
            "pretrained_model_name_or_path": str(directory),
            "running_dtype": "fp32",
            "transformer_param_dtype": "fp32",
            "capabilities": {"distribution_matching": {"official_pdmd": True, "projected_dmd": True}},
        },
        "training": {"method": "dmd", "dmd": {"official_pdmd": True, "official_rollout_sigmas": [1.0, 0.75, 0.0]}},
    }


class ImageGeometryTests(unittest.TestCase):
    def test_head_selection_including_exact_boundaries(self):
        schedule = PDDHeadSchedule(SIGMAS)
        physical = schedule.physical_sigmas()
        self.assertEqual([schedule.head_index(value) for value in physical], [0, 1, 2, 3, 4, 5, 5])
        self.assertEqual([schedule.head_index((left + right) / 2) for left, right in zip(physical[:-1], physical[1:])], list(range(6)))
        self.assertEqual(schedule.head_index(0.75), 3)
        for value in (-0.1, 1.1, float("nan"), torch.tensor([0.1, 0.2])):
            with self.assertRaises(ValueError):
                schedule.head_index(value)

    def test_patch_roundtrip_preserves_gradients(self):
        values = torch.randn((1, 2, 1, 4, 6), requires_grad=True)
        rows = patchify_video_latents(values)
        restored = unpatchify_video_latents(rows, values.shape)
        torch.testing.assert_close(restored, values, rtol=0, atol=0)
        restored.square().sum().backward()
        torch.testing.assert_close(values.grad, 2 * values)

    def test_official_image_coordinates(self):
        layout = packed_t2i_layout(3, 4, 4)
        expected = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [3, 0, 16], [3, 16, 0], [3, 16, 16]], dtype=torch.float64)
        torch.testing.assert_close(layout["position_ids"], expected, rtol=0, atol=0)
        self.assertEqual(layout["image_indices"].tolist(), [3, 4, 5, 6])
        self.assertEqual(layout["token_tags"].tolist(), [1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(layout["timestep_indices"].tolist(), [0] * 7)

    def test_velocity_sign_time_and_head(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            model = MiniMaxH3ImageModel(model_config("unused"))
        model.pdd_schedule = PDDHeadSchedule(SIGMAS)
        model.image_head_mode = "released"
        model.latent_channels = 2
        model.patch_size = (1, 2, 2)
        model._layout_key = None
        model.transformer = mock.Mock(side_effect=lambda **kwargs: (kwargs["hidden_states"] * 2,))
        values = torch.randn(1, 2, 1, 4, 4, requires_grad=True)
        condition = {"prompt_embeds": torch.randn(1, 3, 12)}
        prediction = model.predict_denoiser_output(values, torch.tensor([0.75]), condition)
        torch.testing.assert_close(prediction, -2 * values)
        call = model.transformer.call_args.kwargs
        self.assertEqual(call["pdd_step_index"], 3)
        torch.testing.assert_close(call["timestep"], torch.tensor([0.25]))
        prediction.sum().backward()
        torch.testing.assert_close(values.grad, torch.full_like(values, -2))


class ImageModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            transformer_class()
        except ImportError as error:
            raise unittest.SkipTest(str(error))
        torch.set_num_threads(1)

    def test_native_forward_lora_gradient_and_meta_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            native = save_tiny_model(directory)
            with mock.patch("torch.cuda.is_available", return_value=False):
                model = MiniMaxH3ImageModel(model_config(directory))
                model.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
            condition = model.prepare_text_condition({"prompt_embeds": torch.randn(1, 3, 12), "text_token_tags": torch.ones(3, dtype=torch.long)})
            values = torch.randn(1, 2, 1, 4, 4, requires_grad=True)
            baseline = model.predict_denoiser_output(values, torch.tensor([0.75]), condition)
            native_prediction = native(
                hidden_states=patchify_video_latents(values).unsqueeze(0),
                encoder_hidden_states=condition["prompt_embeds"],
                timestep=torch.tensor([0.25]),
                pdd_step_index=3,
                **packed_t2i_layout(3, 4, 4),
                return_dict=False,
            )[0]
            torch.testing.assert_close(baseline, -unpatchify_video_latents(native_prediction, values.shape))
            model.add_lora(4, 4, None, rank_pattern={"adaln_proj.linear": 2, "norm_out.linear": 2}, alpha_pattern={"adaln_proj.linear": 2, "norm_out.linear": 2})
            model.set_lora_trainable()
            model.set_lora_param_dtype("fp32")
            model.enable_gradient_checkpointing()
            model.set_denoiser_eval()
            prediction = model.predict_denoiser_output(values, torch.tensor([0.75]), condition)
            torch.testing.assert_close(prediction, baseline)
            prediction.square().mean().backward()
            parameters = dict(model.transformer.named_parameters())
            self.assertTrue(torch.isfinite(values.grad).all())
            self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for name, parameter in parameters.items() if ".lora_B." in name))
            self.assertTrue(all(parameter.grad is None for name, parameter in parameters.items() if ".lora_" not in name))
            for name, parameter in parameters.items():
                if ".lora_A." in name:
                    self.assertEqual(parameter.shape[0], 2 if "adaln_proj.linear" in name or "norm_out.linear" in name else 4)
            meta = load_transformer(directory, torch_dtype=torch.bfloat16, meta=True)
            for name, parameter in meta.named_parameters():
                self.assertTrue(parameter.is_meta)
                expected = torch.float32 if name.startswith(("proj_in.", "time_embedder.", "pdd_proj_out.")) else torch.bfloat16
                self.assertEqual(parameter.dtype, expected, name)

    def test_capability_loss_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            save_tiny_model(directory)
            with mock.patch("torch.cuda.is_available", return_value=False):
                model = MiniMaxH3ImageModel(model_config(directory))
                model.load_components(load_transformer=False, load_vae=False, load_condition_encoder=False)
            cap = model.ensure_capabilities().require(DistributionMatchingCapability)
            shape = cap.latent_shape({"conditioning": {"prompt": "test"}}, [{"value": [2048, 2048]}], lambda value: value)
            self.assertEqual(shape, (1, 2, 1, 128, 128))
            image = torch.randn(1, 2, 1, 4, 4, requires_grad=True)
            noise = torch.randn_like(image)
            noised = cap.add_noise(None, image, noise, torch.tensor([0.25]))
            torch.testing.assert_close(noised, image * 0.75 + noise * 0.25)
            loss = cap.dmd_loss(image, torch.zeros_like(image), torch.ones_like(image))
            loss.backward()
            self.assertTrue(torch.isfinite(image.grad).all())
            self.assertTrue(cap.extra_checkpoint_metadata()["h3_image_pdmd"]["score_off_grid_is_experimental"])


if __name__ == "__main__":
    unittest.main()
