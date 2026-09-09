from types import SimpleNamespace

import pytest
import torch

from lightx2v.disagg.conn import DataPoll
from lightx2v.disagg.disagg_mixin import DisaggMixin, _estimate_encoder_buffer_sizes
from lightx2v.disagg.utils import estimate_transformer_buffer_sizes
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.utils.input_info import I2IInputInfo


class MemoryTransfer:
    def __init__(self, source, destination, data_args=None):
        self.source = source
        self.destination = destination
        self.bootstrap_room = 0
        self.ready = False
        self.rearms = 0
        self.data_args = data_args

    def send(self, buffer_ptrs):
        for index, (source, destination) in enumerate(zip(self.source, self.destination)):
            size = self.data_args.data_item_lens[index] if self.data_args is not None else source.numel()
            destination[:size].copy_(source[:size])
        self.ready = True

    def poll(self):
        assert self.ready, "The CPU test must send before receiving"
        return DataPoll.Success

    def init(self):
        self.ready = False
        self.rearms += 1


@pytest.mark.parametrize(("runner_cls", "model_cls", "task"), [(WanRunner, "wan2.1", "t2v"), (QwenImageRunner, "qwen_image", "t2i")])
def test_static_stages_transfer_one_resolved_seed_across_requests(runner_cls, model_cls, task, monkeypatch):
    monkeypatch.setattr("lightx2v.disagg.disagg_mixin.AI_DEVICE", "cpu")
    monkeypatch.setattr("lightx2v.disagg.disagg_mixin.GET_DTYPE", lambda: torch.float32)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    seeded = []

    def seed_cpu(seed):
        seeded.append(seed)
        torch.manual_seed(seed)

    monkeypatch.setattr("lightx2v.models.runners.base_runner.seed_all", seed_cpu)
    monkeypatch.setattr("lightx2v.disagg.disagg_mixin.seed_all", seed_cpu)

    config = {
        "model_cls": model_cls,
        "task": task,
        "seed": 17,
        "infer_steps": 1,
        "target_video_length": 1,
        "target_height": 8,
        "target_width": 8,
        "vae_stride": [1, 2, 2],
        "vae_z_dim": 1,
        "vae_scale_factor": 2,
        "text_len": 2,
        "text_encoder_dim": 2,
        "use_image_encoder": False,
        "enable_cfg": True,
        "disagg_config": {"bootstrap_room": 0},
    }
    runners = []
    noises = {role: [] for role in ("encoder", "transformer", "decode")}
    latent_shape = [1, 1, 4, 4]
    sent_latents, decoded_latents = [], []

    for role in noises:
        runner = runner_cls.__new__(runner_cls)
        BaseRunner.__init__(runner, {**config, "disagg_mode": role})
        runner._gc_frozen = True
        runner._disagg_mode = role
        runner._disagg_decentralized = False
        runner._disagg_bootstrap_room = 0
        runner._disagg_request_config = None
        runner._disagg_rdma_buffers = [torch.zeros(size, dtype=torch.uint8) for size in _estimate_encoder_buffer_sizes(config)]
        runner._disagg_p2_rdma_buffers = [torch.zeros(size, dtype=torch.uint8) for size in estimate_transformer_buffer_sizes(config)]
        runner._disagg_data_mgr = SimpleNamespace(data_args={0: SimpleNamespace(data_item_lens=None)})
        runner._disagg_p2_data_mgr = SimpleNamespace(data_args={0: SimpleNamespace(data_item_lens=None)})
        runner.end_run = lambda: None
        runners.append(runner)

    encoder, transformer, decoder = runners
    phase1 = MemoryTransfer(encoder._disagg_rdma_buffers, transformer._disagg_rdma_buffers)
    phase2 = MemoryTransfer(transformer._disagg_p2_rdma_buffers, decoder._disagg_p2_rdma_buffers)
    encoder._disagg_sender = transformer._disagg_receiver = phase1
    transformer._disagg_p2_sender = decoder._disagg_p2_receiver = phase2

    def encode():
        noises["encoder"].append(torch.randn(8))
        encoder.input_info.latent_shape = latent_shape.copy()
        context = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
        if runner_cls is WanRunner:
            return {"text_encoder_output": {"context": context, "context_null": -context}}
        return {"text_encoder_output": {"prompt_embeds": context, "negative_prompt_embeds": -context}}

    def denoise():
        noises["transformer"].append(torch.randn(8))
        latents = torch.randn(latent_shape)
        sent_latents.append(latents.clone())
        return latents

    def decode(latents):
        noises["decode"].append(torch.randn(8))
        decoded_latents.append(latents.clone())
        return latents

    encoder.run_input_encoder = encode
    decoder.run_vae_decoder = decode
    if runner_cls is WanRunner:
        transformer._run_transformer_role = lambda: transformer.send_transformer_outputs(denoise())
        decoder.process_images_after_vae_decoder = lambda: decoder.gen_video
    else:
        for runner in runners:
            runner.set_latent_shape = lambda runner=runner: setattr(runner.input_info, "latent_shape", latent_shape.copy())
        transformer.run_dit = lambda: (denoise(), None)
        decoder._save_images = lambda *args, **kwargs: None

    with torch.random.fork_rng(devices=[]):
        for index, (request_data, expected_seed) in enumerate([({}, 42), ({"seed": 0}, 0), ({"seed": 29}, 29)]):
            source_input = encoder.prepare_request({"task": encoder.config["task"], "return_result_tensor": True, **request_data})
            transformer_input = transformer.prepare_request({"task": transformer.config["task"], "seed": 999, "return_result_tensor": True})
            decoder_input = decoder.prepare_request({"task": decoder.config["task"], "seed": 999, "return_result_tensor": True})
            assert source_input.seed == expected_seed
            assert transformer_input.seed is None
            assert decoder_input.seed is None

            encoder.run_request(source_input)
            transformer.run_request(transformer_input)
            decoder.run_request(decoder_input)

            assert seeded[-3:] == [expected_seed] * 3
            assert len(seeded) == (index + 1) * 3
            assert all(runner.input_info.seed == expected_seed for runner in runners)
            assert all(runner.config["seed"] == 17 for runner in runners)
            assert torch.equal(noises["encoder"][-1], noises["transformer"][-1])
            assert torch.equal(noises["encoder"][-1], noises["decode"][-1])
            assert torch.equal(sent_latents[-1], decoded_latents[-1])

    assert phase1.rearms == phase2.rearms == 3
    assert not torch.equal(noises["encoder"][0], noises["encoder"][1])
    assert not torch.equal(noises["encoder"][0], noises["encoder"][2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("enable_cfg", [False, True])
def test_i2i_transfers_reference_latents_larger_than_output(dtype, enable_cfg, monkeypatch):
    monkeypatch.setattr("lightx2v.disagg.disagg_mixin.AI_DEVICE", "cpu")
    monkeypatch.setattr("lightx2v.disagg.disagg_mixin.GET_DTYPE", lambda: dtype)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    config = {
        "model_cls": "qwen_image",
        "task": "i2i",
        "target_video_length": 1,
        "target_height": 1664,
        "target_width": 1664,
        "vae_stride": [1, 8, 8],
        "vae_z_dim": 16,
        "text_len": 2,
        "text_encoder_dim": 2,
        "clip_embed_dim": 2,
        "enable_cfg": enable_cfg,
        "use_image_encoder": True,
    }
    runners = []
    for _ in range(2):
        runner = object.__new__(DisaggMixin)
        runner.config = config.copy()
        runner._disagg_decentralized = False
        runner._disagg_bootstrap_room = 0
        runner._disagg_request_config = None
        runner._disagg_rdma_buffers = [torch.zeros(size, dtype=torch.uint8) for size in _estimate_encoder_buffer_sizes(config)]
        runner._disagg_data_mgr = SimpleNamespace(data_args={0: SimpleNamespace(data_item_lens=None)})
        runners.append(runner)
    encoder, transformer = runners
    data_args = encoder._disagg_data_mgr.data_args[0]
    transfer = MemoryTransfer(encoder._disagg_rdma_buffers, transformer._disagg_rdma_buffers, data_args)
    encoder._disagg_sender = transformer._disagg_receiver = transfer
    original_ptrs = [buffer.data_ptr() for buffer in encoder._disagg_rdma_buffers]
    reference = torch.arange(4096 * 64, dtype=torch.float32).reshape(1, 4096, 64).to(dtype)
    context = torch.ones(1, 2, 2, dtype=dtype)
    inputs = {
        "text_encoder_output": {"prompt_embeds": context, "negative_prompt_embeds": -context if enable_cfg else None},
        "image_encoder_output": [{"image_latents": reference}],
    }

    for output_size in (512, 768):
        encoder.input_info = I2IInputInfo(seed=42, target_shape=[output_size, output_size])
        transformer.input_info = I2IInputInfo(seed=42, target_shape=[output_size, output_size])
        request_config = encoder.build_disagg_request_config(encoder.input_info)
        latent_shape = [1, 1, 16, output_size // 8, output_size // 8]
        encoder.send_encoder_outputs(inputs, latent_shape, request_config)
        received = transformer.receive_encoder_outputs(request_config)

        assert torch.equal(received["image_encoder_output"][0]["image_latents"], reference)
        assert received["latent_shape"] == latent_shape
        assert data_args.data_item_lens[2 + int(enable_cfg)] == reference.nbytes
        assert [buffer.data_ptr() for buffer in encoder._disagg_rdma_buffers] == original_ptrs

    assert transfer.rearms == 2

    encoder._disagg_rdma_buffers[2 + int(enable_cfg)] = torch.empty(reference.nbytes - 1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="Phase 1 request exceeds"):
        encoder.send_encoder_outputs(inputs, latent_shape, request_config)
