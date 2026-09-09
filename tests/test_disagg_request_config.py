from types import SimpleNamespace

import pytest
import torch

from lightx2v.disagg.conn import DataPoll
from lightx2v.disagg.disagg_mixin import (
    DisaggMixin,
    _estimate_encoder_buffer_sizes,
    validate_disagg_buffer_capacity,
    wait_for_disagg_transfer,
)
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS
from lightx2v.server.schema import ImageTaskRequest, VideoTaskRequest
from lightx2v.utils.input_info import I2IInputInfo, T2VInputInfo
from lightx2v.utils.lockable_dict import LockableDict


def create_disagg_runner(mode="encoder", decentralized=True):
    runner = object.__new__(DisaggMixin)
    runner.config = LockableDict(
        {
            "task": "t2v",
            "model_cls": "wan2.1",
            "infer_steps": 40,
            "target_video_length": 81,
            "target_height": 480,
            "target_width": 832,
            "disagg_config": {
                "bootstrap_room": 10,
                "sender_engine_rank": 0,
                "receiver_engine_rank": 3,
            },
        }
    )
    runner.config.lock()
    runner._disagg_mode = mode
    runner._disagg_decentralized = decentralized
    runner._disagg_bootstrap_room = 10
    runner._disagg_sender_rank = 0
    runner._disagg_receiver_rank = 3
    runner._disagg_request_config = None
    return runner


class SeedDisaggRunner(DisaggMixin, BaseRunner):
    supported_request_fields_by_task = {"t2v": COMMON_REQUEST_FIELDS}

    def run_pipeline(self, input_info):
        input_info.update(self._disagg_request_config or {})
        return input_info.seed


@pytest.mark.parametrize("selected_seed", [0, 73])
def test_disagg_uses_upstream_seed_before_seeding_each_stage(selected_seed, monkeypatch):
    seeded = []
    monkeypatch.setattr("lightx2v.models.runners.base_runner.seed_all", seeded.append)
    encoder = SeedDisaggRunner({"task": "t2v", "seed": selected_seed})
    # CLI prepares its request before init_disagg creates the dispatch state.
    encoder_input = encoder.prepare_request({"task": encoder.config["task"], "seed": selected_seed})
    assert encoder_input.seed == selected_seed
    request_config = {"seed": encoder_input.seed}

    for mode in ("transformer", "decode"):
        runner = SeedDisaggRunner({"task": "t2v", "seed": 999, "disagg_mode": mode})
        runner._gc_frozen = True
        runner._disagg_request_config = request_config.copy()
        input_info = runner.prepare_request({"task": runner.config["task"], "seed": 111})
        assert input_info.seed == selected_seed
        assert runner.run_request(input_info) == selected_seed
        assert runner.config["seed"] == 999
        request_config = runner._disagg_request_config.copy()

    assert seeded == [selected_seed, selected_seed]

    runner._disagg_request_config = None
    assert runner.prepare_request({"task": runner.config["task"], "seed": 9}).seed is None
    assert runner.prepare_request({"task": runner.config["task"]}).seed is None
    assert encoder.prepare_request({"task": encoder.config["task"], "seed": 9}).seed == 9
    assert encoder.prepare_request({"task": encoder.config["task"]}).seed == 42


@pytest.mark.parametrize("request_cls", [ImageTaskRequest, VideoTaskRequest])
def test_public_requests_do_not_expose_disagg_routing(request_cls):
    routing_fields = {
        "data_bootstrap_room",
        "disagg_bootstrap_room",
        "disagg_decoder_bootstrap_room",
        "disagg_phase1_receiver_engine_rank",
    }

    assert routing_fields.isdisjoint(request_cls.model_fields)


def test_disagg_request_config_keeps_startup_config_unchanged():
    runner = create_disagg_runner()
    first = T2VInputInfo(
        prompt="first",
        seed=1,
        target_video_length=49,
        target_shape=[480, 832],
        return_result_tensor=True,
    )
    second = T2VInputInfo(
        prompt="second",
        seed=2,
        target_video_length=81,
        target_shape=[720, 1280],
    )

    first_config = runner.build_disagg_request_config(
        first,
        {
            "data_bootstrap_room": 101,
            "disagg_phase1_receiver_engine_rank": 3,
        },
    )
    second_config = runner.build_disagg_request_config(
        second,
        {
            "data_bootstrap_room": 102,
            "disagg_phase1_receiver_engine_rank": 4,
        },
    )

    assert first_config["data_bootstrap_room"] == 101
    assert second_config["data_bootstrap_room"] == 102
    assert first_config["disagg_phase1_receiver_engine_rank"] == 3
    assert second_config["disagg_phase1_receiver_engine_rank"] == 4
    assert first_config["target_video_length"] == 49
    assert first_config["target_height"] == 480
    assert first_config["target_width"] == 832
    assert first_config["return_result_tensor"] is True
    assert second_config["target_video_length"] == 81
    assert second_config["target_height"] == 720
    assert second_config["target_width"] == 1280

    assert "data_bootstrap_room" not in runner.config
    assert runner.config["target_video_length"] == 81
    assert runner.config["target_height"] == 480
    assert runner.config["target_width"] == 832
    with pytest.raises(TypeError, match="Dictionary is locked"):
        runner.config["target_video_length"] = 49


def test_disagg_payload_carries_request_geometry():
    runner = create_disagg_runner()
    request_config = runner.build_disagg_request_config(
        T2VInputInfo(
            prompt="request",
            seed=7,
            target_video_length=49,
            target_shape=[480, 832],
        )
    )

    payload = runner._disagg_build_request_config_snapshot(request_config)

    assert payload["target_video_length"] == 49
    assert payload["target_shape"] == [480, 832]
    assert payload["target_height"] == 480
    assert payload["target_width"] == 832
    assert payload["return_result_tensor"] is False
    assert "infer_steps" not in payload
    assert "model_cls" not in payload


def test_i2i_dispatch_preserves_reference_buffer_size_and_denoise_strength(monkeypatch):
    class MemoryDataManager:
        def __init__(self):
            self.data_args = {}

        def init(self, args, room):
            self.data_args[room] = args

        def get_localhost(self):
            return "127.0.0.1"

        def get_session_id(self):
            return "test-session"

    def allocate_buffers(runner, sizes):
        runner._disagg_rdma_buffers = [torch.zeros(size, dtype=torch.uint8) for size in sizes]

    def allocate_p2_buffers(runner, sizes):
        runner._disagg_p2_rdma_buffers = [torch.zeros(size, dtype=torch.uint8) for size in sizes]

    monkeypatch.setattr(DisaggMixin, "_disagg_alloc_buffers", allocate_buffers)
    monkeypatch.setattr(DisaggMixin, "_disagg_alloc_p2_buffers", allocate_p2_buffers)
    monkeypatch.setattr("lightx2v.disagg.disagg_mixin.DataSender", lambda mgr, host, room: SimpleNamespace(bootstrap_room=room))
    monkeypatch.setattr("lightx2v.disagg.disagg_mixin.DataReceiver", lambda mgr, host, room: SimpleNamespace(init=lambda: None))
    encoder = create_disagg_runner()
    transformer = create_disagg_runner(mode="transformer")
    for runner in (encoder, transformer):
        runner.config = {
            **runner.config,
            "task": "i2i",
            "target_video_length": 1,
            "vae_stride": [1, 8, 8],
            "text_len": 2,
            "text_encoder_dim": 2,
            "use_image_encoder": False,
            "disagg_config": {**runner.config["disagg_config"], "decoder_engine_rank": 2},
        }
        runner._disagg_data_mgr = MemoryDataManager()
        runner._disagg_bootstrap_addr = "127.0.0.1"
    encoder._disagg_active_encoder_room = None
    transformer._disagg_active_transformer_room = None
    transformer._disagg_p2_data_mgr = MemoryDataManager()
    phase1_packets, phase2_packets = [], []
    encoder._disagg_phase1_queue = SimpleNamespace(produce=phase1_packets.append)
    transformer._disagg_phase2_queue = SimpleNamespace(produce=phase2_packets.append)

    input_info = I2IInputInfo(seed=42, target_shape=[512, 512], i2i_denoise_strength=0.5)
    request_config = encoder.build_disagg_request_config(input_info, {"data_bootstrap_room": 101})
    sizes = _estimate_encoder_buffer_sizes(encoder._disagg_effective_config(request_config))
    # A 1024x1024 reference image encoded into bfloat16 packed latents.
    sizes[1] = 4096 * 64 * 2
    encoder._disagg_encoder_setup_room(101, request_config, sizes)
    encoder._disagg_produce_phase1_for_encoder(request_config)
    packet = phase1_packets[0]
    transformer.disagg_transformer_prepare_dispatch(packet)

    assert packet["buffer_sizes"] == sizes
    assert transformer._disagg_data_mgr.data_args[101].data_lens == sizes
    assert transformer._disagg_request_config["i2i_denoise_strength"] == 0.5
    assert phase2_packets[0]["request_config"]["i2i_denoise_strength"] == 0.5
    assert transformer._disagg_request_config["target_shape"] == [512, 512]


def test_static_buffer_capacity_comes_from_startup_config():
    class Buffer:
        def __init__(self, size):
            self.size = size

        def numel(self):
            return self.size

    runner = create_disagg_runner(decentralized=False)
    capacity_config = dict(runner.config)
    capacity_config["task"] = "i2v"

    assert capacity_config["target_video_length"] == 81
    assert capacity_config["target_height"] == 480
    assert capacity_config["target_width"] == 832

    capacity_sizes = _estimate_encoder_buffer_sizes(capacity_config)
    buffers = [Buffer(size) for size in capacity_sizes]

    request_config = dict(capacity_config)
    request_config.update(target_video_length=49, target_height=480, target_width=832)
    validate_disagg_buffer_capacity(buffers, _estimate_encoder_buffer_sizes(request_config), "Phase 1")

    request_config.update(target_video_length=81, target_height=720, target_width=1280)
    with pytest.raises(ValueError, match="startup config"):
        validate_disagg_buffer_capacity(buffers, _estimate_encoder_buffer_sizes(request_config), "Phase 1")


def test_disagg_transfer_failure_is_reported(monkeypatch):
    class Transfer:
        def __init__(self):
            self.statuses = iter((DataPoll.WaitingForInput, DataPoll.Failed))

        def poll(self):
            return next(self.statuses)

    monkeypatch.setattr("lightx2v.disagg.disagg_mixin.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Encoder to Transformer transfer failed"):
        wait_for_disagg_transfer(Transfer(), "Encoder to Transformer transfer")


def test_downstream_request_context_is_restored_without_inheritance():
    runner = create_disagg_runner(mode="transformer")
    runner._disagg_request_config = {
        "data_bootstrap_room": 25,
        "prompt": "upstream",
        "seed": 9,
        "target_video_length": 49,
        "target_shape": [480, 832],
        "target_height": 480,
        "target_width": 832,
    }
    input_info = T2VInputInfo()

    input_info.update(runner._disagg_request_config or {})
    request_config = runner.build_disagg_request_config(input_info)

    assert input_info.prompt == "upstream"
    assert input_info.target_video_length == 49
    assert input_info.target_shape == [480, 832]
    assert request_config["data_bootstrap_room"] == 25

    runner._disagg_request_config = None
    next_config = runner.build_disagg_request_config(
        T2VInputInfo(
            prompt="next",
            seed=10,
            target_video_length=81,
            target_shape=[720, 1280],
        ),
        {"data_bootstrap_room": 26},
    )

    assert next_config["data_bootstrap_room"] == 26
    assert next_config["prompt"] == "next"
    assert next_config["target_video_length"] == 81
    assert next_config["target_shape"] == [720, 1280]


def test_effective_disagg_config_uses_request_context_without_mutating_startup_config():
    runner = create_disagg_runner(mode="transformer")
    request_config = {
        "target_video_length": 49,
        "disagg_config": {
            "sender_engine_rank": 3,
            "receiver_engine_rank": 4,
        },
    }

    config = runner._disagg_effective_config(request_config)
    payload = runner._disagg_build_request_config_snapshot(request_config)

    assert config["target_video_length"] == 49
    assert config["infer_steps"] == 40
    assert config["model_cls"] == "wan2.1"
    assert config["disagg_config"]["sender_engine_rank"] == 3
    assert config["disagg_config"]["receiver_engine_rank"] == 4
    assert runner.config["target_video_length"] == 81
    assert runner.config["disagg_config"]["sender_engine_rank"] == 0
    assert runner.config["disagg_config"]["receiver_engine_rank"] == 3
    assert "infer_steps" not in payload
    assert "model_cls" not in payload


def test_request_context_reaches_every_stage_without_leaking_to_the_next_request():
    encoder = create_disagg_runner(mode="encoder")
    transformer = create_disagg_runner(mode="transformer")
    decoder = create_disagg_runner(mode="decode")

    first_input = T2VInputInfo(
        prompt="first",
        negative_prompt="first negative",
        save_result_path="first.mp4",
        return_result_tensor=True,
        seed=1,
        target_video_length=49,
        target_shape=[480, 832],
    )
    first_config = encoder._disagg_build_request_config_snapshot(
        encoder.build_disagg_request_config(
            first_input,
            {
                "data_bootstrap_room": 101,
                "disagg_phase1_receiver_engine_rank": 3,
            },
        )
    )

    transformer._disagg_request_config = first_config
    transformer_input = T2VInputInfo()
    transformer_input.update(transformer._disagg_request_config or {})
    transformer_config = transformer._disagg_build_request_config_snapshot(transformer.build_disagg_request_config(transformer_input))

    decoder._disagg_request_config = transformer_config
    decoder_input = T2VInputInfo()
    decoder_input.update(decoder._disagg_request_config or {})

    assert decoder_input.prompt == "first"
    assert decoder_input.negative_prompt == "first negative"
    assert decoder_input.save_result_path == "first.mp4"
    assert decoder_input.return_result_tensor is True
    assert decoder_input.seed == 1
    assert decoder_input.target_video_length == 49
    assert decoder_input.target_shape == [480, 832]

    encoder._disagg_request_config = None
    transformer._disagg_request_config = None
    decoder._disagg_request_config = None

    second_config = encoder.build_disagg_request_config(
        T2VInputInfo(
            prompt="second",
            save_result_path="second.mp4",
            seed=2,
            target_video_length=81,
            target_shape=[720, 1280],
        ),
        {
            "data_bootstrap_room": 102,
            "disagg_phase1_receiver_engine_rank": 4,
        },
    )

    assert second_config["data_bootstrap_room"] == 102
    assert second_config["prompt"] == "second"
    assert second_config["save_result_path"] == "second.mp4"
    assert second_config["target_video_length"] == 81
    assert second_config["target_shape"] == [720, 1280]
    assert "negative_prompt" in second_config
    assert second_config["negative_prompt"] == ""
