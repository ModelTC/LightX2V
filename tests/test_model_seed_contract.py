import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner
from lightx2v.models.schedulers.bagel.scheduler import BagelScheduler, get_flattened_position_ids_extrapolate
from lightx2v.models.schedulers.hunyuan3d.flow_match_euler import FlowMatchEulerDiscreteScheduler
from lightx2v.models.schedulers.hunyuan3d.scheduler import Hunyuan3DShapeScheduler
from lightx2v.models.schedulers.hunyuan_image3.scheduler import HunyuanImage3Scheduler
from lightx2v.models.schedulers.wan.infinitetalk.scheduler import InfiniteTalkScheduler


@pytest.fixture(params=[({}, 42), ({"seed": None}, 42), ({"seed": 0}, 0), ({"seed": 12}, 12)])
def resolved_input(request):
    request_data, expected_seed = request.param
    runner = BaseRunner.__new__(BaseRunner)
    runner.config = {"task": "t2i", "seed": 37}
    input_info = runner.create_input_info({"task": "t2i", **request_data})
    assert input_info.seed == expected_seed
    return input_info


def test_bagel_keeps_one_cpu_stream_for_multiple_images(resolved_input):
    scheduler = BagelScheduler.__new__(BagelScheduler)
    scheduler.config = {"seed": 999}
    scheduler.latent_downsample = 2
    scheduler.max_latent_size = 8
    scheduler.latent_channel = 2
    scheduler.latent_patch_size = 1
    scheduler.get_flattened_position_ids = get_flattened_position_ids_extrapolate
    args = {
        "curr_kvlens": [3, 0],
        "curr_rope": [4, 5],
        "image_sizes": [(4, 4), (4, 8)],
        "new_token_ids": {"start_of_image": 1, "end_of_image": 2},
    }

    result = scheduler.prepare_vae_latent(**args, seed=resolved_input.seed)
    expected_generator = torch.Generator(device="cpu").manual_seed(resolved_input.seed)
    expected = torch.cat([torch.randn(4, 2, generator=expected_generator), torch.randn(8, 2, generator=expected_generator)])
    assert scheduler.generator.device.type == "cpu"
    assert torch.equal(result["packed_init_noises"], expected)

    scheduler.prepare_vae_latent(**args, seed=123)
    repeated = scheduler.prepare_vae_latent(**args, seed=resolved_input.seed)
    assert torch.equal(repeated["packed_init_noises"], expected)


def test_hunyuan_image3_uses_resolved_request_seed(resolved_input, monkeypatch):
    monkeypatch.setattr("lightx2v.models.schedulers.hunyuan_image3.scheduler.AI_DEVICE", "cpu")
    scheduler = HunyuanImage3Scheduler({"infer_steps": 1, "seed": 999})
    scheduler.prepare(resolved_input)

    expected_generator = torch.Generator(device="cpu").manual_seed(resolved_input.seed)
    assert torch.equal(torch.randn(8, generator=scheduler.generator), torch.randn(8, generator=expected_generator))


@pytest.mark.parametrize(("config_seed", "resolved_seed"), [(None, 0), (999, 27182)])
@pytest.mark.parametrize(("text_config", "generation_options", "text_seed"), [({}, None, None), ({"text_seed": 7}, None, 7), ({"text_seed": 7}, {"text_seed": 13}, 13)])
def test_hunyuan_image3_cot_uses_request_seed_with_text_overrides(config_seed, resolved_seed, text_config, generation_options, text_seed):
    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    runner.config = {"task": "t2i", "seed": config_seed, "max_new_tokens": 3, "text_do_sample": True, **text_config}
    runner._gc_frozen = True
    runner._close_after_run = False
    runner.hunyuan_generation_config = SimpleNamespace()
    runner._hunyuan_text_kv_cache_enabled = lambda: False
    runner._get_ar_cuda_graph_controller = lambda: SimpleNamespace(enabled=False)
    runner._build_text_model_inputs = lambda *args, **kwargs: {}
    runner._broadcast_parallel_tensor = lambda tensor: tensor
    runner._is_output_rank = lambda: True
    logits = torch.tensor([[[0.0, 0.5, 1.0, 1.5]]])
    runner.model = SimpleNamespace(infer=lambda inputs: {"logits": logits})
    plan = SimpleNamespace(stage_transitions=[], final_stop_tokens=set())
    observed_seeds = []
    sample_token = runner._sample_text_token

    def sample(logits, generator, generation_options=None):
        observed_seeds.append(generator.initial_seed())
        return sample_token(logits, generator, generation_options)

    runner._sample_text_token = sample
    runner.generate_t2i = lambda info: runner._generate_text_tokens(torch.tensor([[1]]), SimpleNamespace(), plan, generation_options=generation_options)
    input_info = SimpleNamespace(seed=resolved_seed, return_result_tensor=True)

    result = runner.run_pipeline(input_info)

    expected_seed = resolved_seed if text_seed is None else text_seed
    expected_generator = torch.Generator(device="cpu").manual_seed(expected_seed)
    expected_tokens = [torch.multinomial(torch.softmax(logits[0, 0], dim=-1), 1, generator=expected_generator).item() for _ in range(3)]
    assert runner.input_info is input_info
    assert observed_seeds == [expected_seed] * 3
    assert result == {"image": expected_tokens}


def test_hunyuan3d_noise_uses_resolved_request_seed(resolved_input):
    scheduler = Hunyuan3DShapeScheduler.__new__(Hunyuan3DShapeScheduler)
    scheduler.config = {"infer_steps": 2, "seed": 999}
    scheduler.flow_scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.device = torch.device("cpu")
    scheduler.dtype = torch.float32
    scheduler.prepare(resolved_input.seed, latent_shape=(1, 3, 4))

    expected_generator = torch.Generator(device="cpu").manual_seed(resolved_input.seed)
    expected = torch.randn((1, 3, 4), generator=expected_generator)
    assert torch.equal(scheduler.latents, expected)


def test_infinitetalk_preserves_global_rng_and_determinism(resolved_input, monkeypatch):
    cuda_seeds = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_seeds.append)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    scheduler = InfiniteTalkScheduler({"infer_steps": 1, "sample_shift": 5})
    python_state, numpy_state = random.getstate(), np.random.get_state()

    try:
        with torch.random.fork_rng(devices=[]):
            assert scheduler.seed_everything(resolved_input.seed) == resolved_input.seed
            expected_generator = torch.Generator(device="cpu").manual_seed(resolved_input.seed)
            assert torch.equal(torch.randn(8), torch.randn(8, generator=expected_generator))
            assert random.random() == random.Random(resolved_input.seed).random()
            assert np.random.random() == np.random.RandomState(resolved_input.seed).random_sample()
            assert cuda_seeds[-1] == resolved_input.seed
            assert torch.backends.cudnn.deterministic
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
