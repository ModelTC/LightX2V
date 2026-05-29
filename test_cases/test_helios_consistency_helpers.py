import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class HeliosPromptPackingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        transformers_module = types.ModuleType("transformers")
        transformers_module.AutoTokenizer = object
        transformers_module.UMT5EncoderModel = object
        sys.modules.setdefault("transformers", transformers_module)

        envs_module = types.ModuleType("lightx2v.utils.envs")
        envs_module.GET_DTYPE = lambda: torch.bfloat16
        sys.modules.setdefault("lightx2v.utils.envs", envs_module)

        global_var_module = types.ModuleType("lightx2v_platform.base.global_var")
        global_var_module.AI_DEVICE = "cpu"
        sys.modules.setdefault("lightx2v_platform.base.global_var", global_var_module)

        cls.text_module = load_module(
            "test_helios_text_model",
            REPO_ROOT / "lightx2v/models/input_encoders/hf/helios/model.py",
        )

    def test_pack_prompt_embeds_reapplies_sequence_lengths_before_padding(self):
        hidden_state = torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0], [999.0, 999.0], [999.0, 999.0]],
                [[3.0, 30.0], [4.0, 40.0], [5.0, 50.0], [999.0, 999.0]],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 0],
            ]
        )

        prompt_embeds, mask = self.text_module.pack_t5_prompt_embeds(
            hidden_state,
            attention_mask,
            max_sequence_length=4,
            num_videos_per_prompt=2,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(prompt_embeds.shape), (4, 4, 2))
        self.assertEqual(prompt_embeds.dtype, torch.bfloat16)
        self.assertTrue(mask.dtype == torch.bool)
        self.assertTrue(torch.equal(mask, attention_mask.bool()))
        self.assertTrue(torch.equal(prompt_embeds[0, 2:], torch.zeros((2, 2), dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(prompt_embeds[1], prompt_embeds[0]))
        self.assertTrue(torch.equal(prompt_embeds[2, 3:], torch.zeros((1, 2), dtype=torch.bfloat16)))
        self.assertEqual(prompt_embeds[2, 2, 0].item(), 5.0)


class HeliosRuntimeUtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime_utils = load_module(
            "test_helios_runtime_utils",
            REPO_ROOT / "lightx2v/models/runners/helios/runtime_utils.py",
        )

    def test_apply_image_condition_noise_uses_distinct_sigmas_and_generator_order(self):
        image_latents = torch.ones((1, 1, 1, 1, 1), dtype=torch.float32)
        fake_image_latents = torch.full((1, 1, 1, 1, 1), 2.0, dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(123)

        noisy_image, noisy_fake = self.runtime_utils.apply_image_condition_noise(
            image_latents=image_latents,
            fake_image_latents=fake_image_latents,
            generator=generator,
            device=torch.device("cpu"),
            image_noise_sigma_min=0.111,
            image_noise_sigma_max=0.135,
            video_noise_sigma_min=0.211,
            video_noise_sigma_max=0.235,
        )

        ref_generator = torch.Generator(device="cpu").manual_seed(123)
        image_sigma = torch.rand(1, device="cpu", generator=ref_generator) * (0.135 - 0.111) + 0.111
        ref_noisy_image = image_sigma * torch.randn(image_latents.shape, generator=ref_generator) + (1 - image_sigma) * image_latents
        fake_sigma = torch.rand(1, device="cpu", generator=ref_generator) * (0.235 - 0.211) + 0.211
        ref_noisy_fake = fake_sigma * torch.randn(fake_image_latents.shape, generator=ref_generator) + (1 - fake_sigma) * fake_image_latents

        self.assertTrue(torch.allclose(noisy_image, ref_noisy_image))
        self.assertTrue(torch.allclose(noisy_fake, ref_noisy_fake))

    def test_trim_and_postprocess_video_matches_helios_frame_rule(self):
        history_video = torch.arange(1 * 3 * 99 * 2 * 2, dtype=torch.float32).reshape(1, 3, 99, 2, 2)

        class DummyVideoProcessor:
            def __init__(self):
                self.called = False
                self.last_shape = None

            def postprocess_video(self, video, output_type="np"):
                self.called = True
                self.last_shape = tuple(video.shape)
                return {"frames": video.clone(), "output_type": output_type}

        processor = DummyVideoProcessor()
        result = self.runtime_utils.finalize_video_output(
            history_video=history_video,
            video_processor=processor,
            temporal_scale_factor=4,
            output_type="np",
        )

        self.assertTrue(processor.called)
        self.assertEqual(processor.last_shape, (1, 3, 97, 2, 2))
        self.assertEqual(tuple(result["frames"].shape), (1, 3, 97, 2, 2))
        self.assertEqual(result["output_type"], "np")

    def test_pt_video_output_is_converted_to_comfy_frame_layout(self):
        pt_video = torch.arange(1 * 2 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 2, 3, 2, 2)
        frames = self.runtime_utils.pt_video_output_to_comfy_frames(pt_video)
        self.assertEqual(tuple(frames.shape), (2, 2, 2, 3))
        self.assertTrue(torch.equal(frames[0, 0, 0], torch.tensor([0.0, 4.0, 8.0])))


class HeliosI2VGeneratorContinuityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_scheduler_module = types.ModuleType("lightx2v.models.schedulers.scheduler")

        class BaseScheduler:
            def __init__(self, config):
                self.config = config

        base_scheduler_module.BaseScheduler = BaseScheduler
        sys.modules.setdefault("lightx2v.models.schedulers.scheduler", base_scheduler_module)

        fake_dmd_module = types.ModuleType("lightx2v.models.schedulers.helios.helios_dmd")

        class FakeInnerScheduler:
            config = types.SimpleNamespace()

            @classmethod
            def from_pretrained(cls, _path):
                return cls()

        fake_dmd_module.HeliosDMDScheduler = FakeInnerScheduler
        sys.modules.setdefault("lightx2v.models.schedulers.helios.helios_dmd", fake_dmd_module)

        global_var_module = types.ModuleType("lightx2v_platform.base.global_var")
        global_var_module.AI_DEVICE = "cpu"
        sys.modules["lightx2v_platform.base.global_var"] = global_var_module

        cls.scheduler_module = load_module(
            "test_helios_scheduler_module",
            REPO_ROOT / "lightx2v/models/schedulers/helios/scheduler.py",
        )

    def test_prepare_reuses_external_generator_for_i2v_rng_continuity(self):
        scheduler = self.scheduler_module.HeliosDistilledScheduler(
            {
                "scheduler_path": "/tmp/unused",
                "pyramid_num_inference_steps_list": [2, 2, 2],
                "sample_guide_scale": 1.0,
            }
        )
        external_generator = torch.Generator(device="cpu").manual_seed(42)
        scheduler.prepare(seed=999, latent_shape=[16, 25, 48, 80], image_encoder_output={}, generator=external_generator)
        self.assertIs(scheduler.generator, external_generator)

    def test_helios_runner_i2v_no_longer_reseeds_a_second_generator(self):
        runner_path = REPO_ROOT / "lightx2v/models/runners/helios/helios_runner.py"
        source = runner_path.read_text(encoding="utf-8")
        i2v_block = source.split("def _run_input_encoder_local_i2v", 1)[1].split("def sample_block_noise", 1)[0]
        self.assertNotIn("manual_seed(self.input_info.seed)", i2v_block)


if __name__ == "__main__":
    unittest.main()
