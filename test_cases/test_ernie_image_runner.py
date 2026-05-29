# ruff: noqa: E402
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

loguru_stub = ModuleType("loguru")
loguru_stub.logger = SimpleNamespace(
    debug=lambda *args, **kwargs: None,
    error=lambda *args, **kwargs: None,
    info=lambda *args, **kwargs: None,
    warning=lambda *args, **kwargs: None,
)
sys.modules.setdefault("loguru", loguru_stub)

imageio_ffmpeg_stub = ModuleType("imageio_ffmpeg")
imageio_ffmpeg_stub.get_ffmpeg_exe = lambda: "ffmpeg"
sys.modules.setdefault("imageio_ffmpeg", imageio_ffmpeg_stub)

lightx2v_pkg = ModuleType("lightx2v")
lightx2v_pkg.__path__ = [os.path.join(REPO_ROOT, "lightx2v")]
sys.modules.setdefault("lightx2v", lightx2v_pkg)

import lightx2v_platform.base.global_var as global_var

global_var.AI_DEVICE = "cpu"

from lightx2v.utils.registry_factory import RUNNER_REGISTER


class FakeImage:
    def __init__(self):
        self.saved_path = None

    def save(self, path):
        self.saved_path = str(path)
        Path(path).write_text("fake image")


class FakeTransformer:
    def __init__(self):
        self.dtype = torch.float32
        self.config = SimpleNamespace(in_channels=128, text_in_dim=4)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return (torch.zeros_like(kwargs["hidden_states"]),)


class FakeScheduler:
    def __init__(self):
        self.timesteps = []
        self.set_timesteps_calls = []
        self.step_calls = []

    def set_timesteps(self, sigmas, device):
        self.set_timesteps_calls.append((sigmas, device))
        self.timesteps = sigmas

    def step(self, pred, timestep, latents):
        self.step_calls.append((pred, timestep, latents))
        return SimpleNamespace(prev_sample=latents)


class FakeVAE:
    def __init__(self):
        self.bn = SimpleNamespace(
            running_mean=torch.zeros(128),
            running_var=torch.ones(128),
        )
        self.decode_calls = []
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(torch.device(device))
        return self

    def decode(self, latents, return_dict=False):
        self.decode_calls.append((latents, return_dict))
        batch_size = latents.shape[0]
        return (torch.zeros(batch_size, 3, 16, 16),)


class FakeErnieImagePipeline:
    from_pretrained_calls = []

    def __init__(self):
        self.cpu_offload_enabled = False
        self.device = None
        self.calls = []
        self.image = FakeImage()
        self.transformer = FakeTransformer()
        self.scheduler = FakeScheduler()
        self.vae = FakeVAE()
        self.vae_scale_factor = 16
        self.pe = object()
        self.pe_tokenizer = object()
        self.encoded_prompts = []
        self.enhanced_prompts = []
        self.freed_hooks = False
        self._execution_device = torch.device("cpu")

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        cls.from_pretrained_calls.append((model_path, kwargs))
        return cls()

    def enable_model_cpu_offload(self):
        self.cpu_offload_enabled = True

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[self.image])

    def _enhance_prompt_with_pe(self, prompt, device, width=1024, height=1024):
        self.enhanced_prompts.append((prompt, device, width, height))
        return f"enhanced {prompt}"

    def encode_prompt(self, prompt, device, num_images_per_prompt=1):
        self.encoded_prompts.append((list(prompt), device, num_images_per_prompt))
        hiddens = []
        for _ in prompt:
            for _ in range(num_images_per_prompt):
                hiddens.append(torch.ones(2, self.transformer.config.text_in_dim, device=device))
        return hiddens

    @staticmethod
    def _pad_text(text_hiddens, device, dtype, text_in_dim):
        lens = torch.tensor([hidden.shape[0] for hidden in text_hiddens], device=device, dtype=torch.long)
        text_bth = torch.zeros(len(text_hiddens), int(lens.max().item()), text_in_dim, device=device, dtype=dtype)
        for index, hidden in enumerate(text_hiddens):
            text_bth[index, : hidden.shape[0]] = hidden.to(device=device, dtype=dtype)
        return text_bth, lens

    @staticmethod
    def _unpatchify_latents(latents):
        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(batch_size, channels // 4, height * 2, width * 2)

    def maybe_free_model_hooks(self):
        self.freed_hooks = True


def make_config(**overrides):
    config = {
        "model_cls": "ernie_image",
        "model_path": "baidu/ERNIE-Image",
        "task": "t2i",
        "cpu_offload": True,
        "infer_steps": 8,
        "sample_guide_scale": 1.0,
        "target_height": 512,
        "target_width": 768,
        "use_pe": False,
    }
    config.update(overrides)
    return config


class ErnieImageRunnerTest(unittest.TestCase):
    def setUp(self):
        FakeErnieImagePipeline.from_pretrained_calls.clear()

    def test_vae_wrapper_decodes_pil_and_preserves_latent_output(self):
        from lightx2v.models.video_encoders.hf.ernie_image.vae import AutoencoderKLErnieImageVAE

        fake_vae = FakeVAE()
        wrapper = AutoencoderKLErnieImageVAE({}, fake_vae)
        latents = torch.zeros(1, 128, 4, 4)

        images = wrapper.decode(latents, output_type="pil")

        self.assertEqual(fake_vae.decode_calls[0][0].shape, torch.Size([1, 32, 8, 8]))
        self.assertEqual(len(images), 1)
        self.assertIsInstance(images[0], Image.Image)

        latent_result = wrapper.decode(latents, output_type="latent")

        self.assertIs(latent_result, latents)
        self.assertEqual(len(fake_vae.decode_calls), 1)

    def test_vae_wrapper_delegates_diffusers_cpu_offload_to_pipeline_hooks(self):
        from lightx2v.models.video_encoders.hf.ernie_image.vae import AutoencoderKLErnieImageVAE

        fake_vae = FakeVAE()
        wrapper = AutoencoderKLErnieImageVAE({"cpu_offload": True}, fake_vae, diffusers_cpu_offload=True)

        wrapper.decode(torch.zeros(1, 128, 4, 4), output_type="pil")

        self.assertTrue(wrapper.cpu_offload)
        self.assertTrue(wrapper.diffusers_cpu_offload)
        self.assertFalse(wrapper.manage_cpu_offload)
        self.assertEqual(fake_vae.to_calls, [])

    def test_vae_wrapper_can_manage_standalone_cpu_offload(self):
        from lightx2v.models.video_encoders.hf.ernie_image.vae import AutoencoderKLErnieImageVAE

        fake_vae = FakeVAE()
        latents = torch.zeros(1, 128, 4, 4)
        wrapper = AutoencoderKLErnieImageVAE({"vae_cpu_offload": True, "cpu_offload": False}, fake_vae)

        wrapper.decode(latents, output_type="pil")

        self.assertTrue(wrapper.cpu_offload)
        self.assertFalse(wrapper.diffusers_cpu_offload)
        self.assertTrue(wrapper.manage_cpu_offload)
        self.assertEqual(fake_vae.to_calls[0], torch.device("cpu"))
        self.assertEqual(fake_vae.to_calls[-2:], [latents.device, torch.device("cpu")])

    def test_scheduler_wrapper_prepares_sigmas_and_returns_prev_sample(self):
        from lightx2v.models.schedulers.ernie_image.scheduler import ErnieImageScheduler

        fake_scheduler = FakeScheduler()
        wrapper = ErnieImageScheduler({}, fake_scheduler)
        device = torch.device("cpu")
        latent_shape = (1, 128, 4, 4)

        timesteps = wrapper.prepare(4, device, latent_shape=latent_shape, dtype=torch.float32, seed=123)
        pred = torch.zeros_like(wrapper.latents)
        wrapper.step_pre(0)
        next_latents = wrapper.step(pred, timesteps[0])

        sigmas, call_device = fake_scheduler.set_timesteps_calls[0]
        self.assertEqual(call_device, device)
        self.assertTrue(torch.equal(sigmas, torch.tensor([1.0, 0.75, 0.5, 0.25])))
        self.assertIs(wrapper.timesteps, fake_scheduler.timesteps)
        self.assertEqual(wrapper.infer_steps, 4)
        self.assertEqual(wrapper.latents.shape, torch.Size(latent_shape))
        self.assertIsInstance(wrapper.generator, torch.Generator)
        self.assertEqual(wrapper.step_index, 0)
        self.assertIs(wrapper.noise_pred, pred)
        self.assertIs(next_latents, wrapper.latents)

    def test_text_encoder_wrapper_handles_pe_cfg_and_padding(self):
        from lightx2v.models.input_encoders.hf.ernie_image.text_encoder import ErnieImageTextEncoder

        pipe = FakeErnieImagePipeline()
        wrapper = ErnieImageTextEncoder({}, pipe, diffusers_cpu_offload=True)
        generation_kwargs = {
            "prompt": "a glass pavilion",
            "negative_prompt": "low quality",
            "width": 768,
            "height": 512,
            "use_pe": True,
            "num_images_per_prompt": 1,
        }

        batch_size, text_bth, text_lens = wrapper.encode(
            generation_kwargs,
            torch.device("cpu"),
            torch.float32,
            do_classifier_free_guidance=True,
        )

        self.assertEqual(batch_size, 1)
        self.assertEqual(pipe.enhanced_prompts[0], ("a glass pavilion", torch.device("cpu"), 768, 512))
        self.assertEqual(wrapper.revised_prompts, ["enhanced a glass pavilion"])
        self.assertEqual(pipe.encoded_prompts[0][0], ["enhanced a glass pavilion"])
        self.assertEqual(pipe.encoded_prompts[1][0], ["low quality"])
        self.assertEqual(text_bth.shape, torch.Size([2, 2, 4]))
        self.assertTrue(torch.equal(text_lens, torch.tensor([2, 2])))

    def test_transformer_model_wrapper_handles_cfg_and_sets_scheduler_noise_pred(self):
        from lightx2v.models.networks.ernie_image.model import ErnieImageTransformerModel

        class CfgTransformer(FakeTransformer):
            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                batch_size = kwargs["hidden_states"].shape[0]
                values = torch.arange(batch_size, dtype=kwargs["hidden_states"].dtype).view(batch_size, 1, 1, 1)
                return (values.expand_as(kwargs["hidden_states"]),)

        scheduler = SimpleNamespace(latents=torch.zeros(1, 128, 4, 4), noise_pred=None)
        transformer = CfgTransformer()
        wrapper = ErnieImageTransformerModel({}, transformer)
        wrapper.set_scheduler(scheduler)
        text_bth = torch.zeros(2, 2, 4)
        text_lens = torch.tensor([2, 2])

        pred = wrapper.infer(
            {
                "timestep": torch.tensor(0.5),
                "text_bth": text_bth,
                "text_lens": text_lens,
                "device": torch.device("cpu"),
                "dtype": torch.float32,
                "total_batch_size": 1,
                "guidance_scale": 4.0,
                "do_classifier_free_guidance": True,
            }
        )

        self.assertTrue(torch.equal(pred, torch.full_like(scheduler.latents, 4.0)))
        self.assertIs(scheduler.noise_pred, pred)
        self.assertEqual(transformer.calls[0]["hidden_states"].shape[0], 2)
        self.assertEqual(transformer.calls[0]["timestep"].shape, torch.Size([2]))
        self.assertIs(transformer.calls[0]["text_bth"], text_bth)
        self.assertIs(transformer.calls[0]["text_lens"], text_lens)

    def test_pipeline_components_expose_diffusers_modules_and_hooks(self):
        from lightx2v.models.runners.ernie_image.components import ErnieImagePipelineComponents

        with mock.patch("diffusers.ErnieImagePipeline", FakeErnieImagePipeline, create=True):
            components = ErnieImagePipelineComponents.from_pretrained(
                "baidu/ERNIE-Image",
                {"torch_dtype": torch.bfloat16},
                target_device="cpu",
                cpu_offload=True,
            )

        self.assertEqual(FakeErnieImagePipeline.from_pretrained_calls[0][0], "baidu/ERNIE-Image")
        self.assertEqual(FakeErnieImagePipeline.from_pretrained_calls[0][1]["torch_dtype"], torch.bfloat16)
        self.assertTrue(components.pipeline.cpu_offload_enabled)
        self.assertIs(components.transformer, components.pipeline.transformer)
        self.assertIs(components.scheduler, components.pipeline.scheduler)
        self.assertIs(components.vae, components.pipeline.vae)
        self.assertEqual(components.vae_scale_factor, 16)
        self.assertEqual(components.execution_device, torch.device("cpu"))

        revised_prompt = components.enhance_prompt_with_pe("a glass pavilion", torch.device("cpu"), width=768, height=512)
        text_hiddens = components.encode_prompt([revised_prompt], torch.device("cpu"), 1)
        text_bth, text_lens = components.pad_text(text_hiddens, torch.device("cpu"), torch.float32)

        self.assertEqual(revised_prompt, "enhanced a glass pavilion")
        self.assertEqual(text_bth.shape, torch.Size([1, 2, 4]))
        self.assertTrue(torch.equal(text_lens, torch.tensor([2])))

    def test_runner_is_registered(self):
        from lightx2v.models.runners.ernie_image.ernie_image_runner import ErnieImageRunner

        self.assertIs(RUNNER_REGISTER["ernie_image"], ErnieImageRunner)

    def test_auto_calc_config_allows_huggingface_repo_id(self):
        from lightx2v.utils.set_config import auto_calc_config, get_default_config

        config = get_default_config()
        config.update(
            {
                "config_json": None,
                "model_cls": "ernie_image",
                "model_path": "baidu/ERNIE-Image",
                "task": "t2i",
            }
        )

        result = auto_calc_config(config)

        self.assertEqual(result["model_path"], "baidu/ERNIE-Image")

    def test_load_model_uses_diffusers_pipeline_and_cpu_offload(self):
        from lightx2v.models.input_encoders.hf.ernie_image.text_encoder import ErnieImageTextEncoder
        from lightx2v.models.networks.ernie_image.model import ErnieImageTransformerModel
        from lightx2v.models.runners.ernie_image.components import ErnieImagePipelineComponents
        from lightx2v.models.runners.ernie_image.ernie_image_runner import ErnieImageRunner
        from lightx2v.models.schedulers.ernie_image.scheduler import ErnieImageScheduler
        from lightx2v.models.video_encoders.hf.ernie_image.vae import AutoencoderKLErnieImageVAE

        with mock.patch("diffusers.ErnieImagePipeline", FakeErnieImagePipeline, create=True):
            runner = ErnieImageRunner(make_config())
            runner.init_modules()

        self.assertEqual(FakeErnieImagePipeline.from_pretrained_calls[0][0], "baidu/ERNIE-Image")
        self.assertEqual(FakeErnieImagePipeline.from_pretrained_calls[0][1]["torch_dtype"], torch.bfloat16)
        self.assertTrue(runner.pipe.cpu_offload_enabled)
        self.assertIsNone(runner.pipe.device)
        self.assertIsInstance(runner.components, ErnieImagePipelineComponents)
        self.assertIs(runner.components.pipeline, runner.pipe)
        self.assertIsInstance(runner.text_encoder, ErnieImageTextEncoder)
        self.assertIs(runner.text_encoder.components, runner.components)
        self.assertTrue(runner.text_encoder.diffusers_cpu_offload)
        self.assertIsInstance(runner.model, ErnieImageTransformerModel)
        self.assertIs(runner.model.transformer, runner.components.transformer)
        self.assertIs(runner.model.scheduler, runner.scheduler)
        self.assertTrue(runner.model.diffusers_cpu_offload)
        self.assertIsInstance(runner.vae, AutoencoderKLErnieImageVAE)
        self.assertIs(runner.vae.model, runner.components.vae)
        self.assertTrue(runner.vae.cpu_offload)
        self.assertTrue(runner.vae.diffusers_cpu_offload)
        self.assertFalse(runner.vae.manage_cpu_offload)
        self.assertIsInstance(runner.scheduler, ErnieImageScheduler)
        self.assertIs(runner.scheduler.scheduler, runner.components.scheduler)

    def test_builtin_16_by_9_shape_is_allowed(self):
        from lightx2v.models.runners.ernie_image.ernie_image_runner import resolve_ernie_image_shape

        input_info = SimpleNamespace(target_shape=[], aspect_ratio="16:9")
        height, width = resolve_ernie_image_shape(input_info, make_config(target_height=None, target_width=None))

        self.assertEqual((height, width), (720, 1280))

    def test_progress_callback_runs_from_decomposed_step_loop(self):
        from lightx2v.models.runners.ernie_image.ernie_image_runner import ErnieImageRunner

        progress_calls = []
        input_info = SimpleNamespace(
            seed=123,
            prompt="a glass pavilion",
            save_result_path="/tmp/unused.png",
            return_result_tensor=True,
            target_shape=[640, 640],
            aspect_ratio="16:9",
        )

        with mock.patch("diffusers.ErnieImagePipeline", FakeErnieImagePipeline, create=True):
            runner = ErnieImageRunner(make_config())
            runner.set_progress_callback(lambda percent, total: progress_calls.append((percent, total)))
            runner.init_modules()
            runner.run_pipeline(input_info)

        self.assertEqual(runner.pipe.calls, [])
        self.assertEqual(len(progress_calls), 8)
        self.assertEqual(progress_calls[0], (12.5, 100))
        self.assertEqual(progress_calls[-1], (100.0, 100))

    def test_run_pipeline_passes_generation_parameters_and_saves_image(self):
        from lightx2v.models.runners.ernie_image.ernie_image_runner import ErnieImageRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "ernie.png"
            input_info = SimpleNamespace(
                seed=123,
                prompt="a compact observatory on a snowy ridge",
                negative_prompt="low quality",
                save_result_path=str(output_path),
                return_result_tensor=False,
                target_shape=[],
                aspect_ratio="1:1",
            )

            with mock.patch("diffusers.ErnieImagePipeline", FakeErnieImagePipeline, create=True):
                runner = ErnieImageRunner(make_config())
                runner.init_modules()
                result = runner.run_pipeline(input_info)

            self.assertEqual(runner.pipe.calls, [])
            self.assertEqual(runner.pipe.encoded_prompts[0][0], [input_info.prompt])
            self.assertEqual(len(runner.pipe.transformer.calls), 8)
            self.assertEqual(len(runner.pipe.scheduler.step_calls), 8)
            self.assertEqual(len(runner.pipe.vae.decode_calls), 1)
            self.assertEqual(runner.scheduler.latents.shape, torch.Size([1, 128, 32, 48]))
            self.assertIsInstance(runner.scheduler.generator, torch.Generator)
            self.assertEqual(runner.scheduler.step_index, 7)
            self.assertTrue(runner.pipe.freed_hooks)
            with Image.open(output_path) as image:
                self.assertIsInstance(image, Image.Image)
            self.assertTrue(output_path.exists())
            self.assertEqual(result, {"images": None})

    def test_decomposed_runner_handles_cfg_and_progress_without_calling_pipeline(self):
        from lightx2v.models.runners.ernie_image.ernie_image_runner import ErnieImageRunner

        progress_calls = []
        input_info = SimpleNamespace(
            seed=123,
            prompt="a glass pavilion",
            negative_prompt="low quality",
            save_result_path="/tmp/unused.png",
            return_result_tensor=True,
            target_shape=[64, 64],
            aspect_ratio="1:1",
        )

        with mock.patch("diffusers.ErnieImagePipeline", FakeErnieImagePipeline, create=True):
            runner = ErnieImageRunner(make_config(target_height=None, target_width=None, sample_guide_scale=4.0, use_pe=True))
            runner.set_progress_callback(lambda percent, total: progress_calls.append((percent, total)))
            runner.init_modules()
            result = runner.run_pipeline(input_info)

        self.assertEqual(runner.pipe.calls, [])
        self.assertEqual(runner.pipe.enhanced_prompts[0][0], input_info.prompt)
        self.assertEqual(runner.pipe.encoded_prompts[0][0], [f"enhanced {input_info.prompt}"])
        self.assertEqual(runner.pipe.encoded_prompts[1][0], [input_info.negative_prompt])
        self.assertEqual(len(runner.pipe.transformer.calls), 8)
        self.assertEqual(runner.pipe.transformer.calls[0]["hidden_states"].shape[0], 2)
        self.assertEqual(progress_calls[-1], (100.0, 100))
        self.assertEqual(len(progress_calls), 8)
        self.assertEqual(len(result["images"]), 1)

    def test_decomposed_runner_delegates_transformer_inference_to_model_boundary(self):
        from lightx2v.models.runners.ernie_image.ernie_image_runner import ErnieImageRunner

        input_info = SimpleNamespace(
            seed=123,
            prompt="a glass pavilion",
            negative_prompt="low quality",
            save_result_path="/tmp/unused.png",
            return_result_tensor=True,
            target_shape=[64, 64],
            aspect_ratio="1:1",
        )

        with mock.patch("diffusers.ErnieImagePipeline", FakeErnieImagePipeline, create=True):
            runner = ErnieImageRunner(make_config(target_height=None, target_width=None, sample_guide_scale=4.0, use_pe=True))
            runner.init_modules()

            def infer_through_boundary(inputs):
                del inputs
                runner.scheduler.noise_pred = torch.zeros_like(runner.scheduler.latents)
                return runner.scheduler.noise_pred

            runner.model.infer = mock.Mock(side_effect=infer_through_boundary)
            runner.run_pipeline(input_info)

        self.assertEqual(runner.pipe.transformer.calls, [])
        self.assertEqual(runner.model.infer.call_count, 8)

    def test_return_tensor_skips_save(self):
        from lightx2v.models.runners.ernie_image.ernie_image_runner import ErnieImageRunner

        input_info = SimpleNamespace(
            seed=123,
            prompt="a glass pavilion",
            save_result_path="/tmp/unused.png",
            return_result_tensor=True,
            target_shape=[640, 640],
            aspect_ratio="16:9",
        )

        with mock.patch("diffusers.ErnieImagePipeline", FakeErnieImagePipeline, create=True):
            runner = ErnieImageRunner(make_config())
            runner.init_modules()
            result = runner.run_pipeline(input_info)

        self.assertEqual(runner.pipe.calls, [])
        self.assertEqual(len(result["images"]), 1)
        self.assertIsInstance(result["images"][0], Image.Image)
        self.assertIsNone(runner.pipe.image.saved_path)

    def test_return_tensor_still_unloads_pipeline_when_configured(self):
        from lightx2v.models.runners.ernie_image.ernie_image_runner import ErnieImageRunner

        input_info = SimpleNamespace(
            seed=123,
            prompt="a glass pavilion",
            save_result_path="/tmp/unused.png",
            return_result_tensor=True,
            target_shape=[640, 640],
            aspect_ratio="16:9",
        )

        with mock.patch("diffusers.ErnieImagePipeline", FakeErnieImagePipeline, create=True):
            runner = ErnieImageRunner(make_config(unload_modules=True))
            runner.init_modules()
            result = runner.run_pipeline(input_info)

        self.assertEqual(len(result["images"]), 1)
        self.assertIsInstance(result["images"][0], Image.Image)
        self.assertIsNone(runner.pipe)
        self.assertIsNone(runner.components)
        self.assertIsNone(runner.model)
        self.assertIsNone(runner.text_encoder)
        self.assertIsNone(runner.vae)
        self.assertIsNone(runner.scheduler)


if __name__ == "__main__":
    unittest.main()
