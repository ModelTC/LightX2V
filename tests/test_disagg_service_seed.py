import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightx2v.disagg.examples import run_service, run_user
from lightx2v.disagg.workload import StageSpec
from lightx2v.models.schedulers.wan.scheduler import WanScheduler


@pytest.fixture
def service_modules(monkeypatch):
    # The CPU tests replace only the unavailable pyverbs transport imports.
    modules = {}
    service_dir = Path(run_service.__file__).parents[1] / "services"
    with monkeypatch.context() as transport_imports:
        for name, class_name in (("rdma_server", "RDMAServer"), ("rdma_client", "RDMAClient")):
            module = ModuleType(f"lightx2v.disagg.{name}")
            setattr(module, class_name, object)
            transport_imports.setitem(sys.modules, module.__name__, module)
        for name in ("controller", "encoder", "transformer", "decoder"):
            spec = importlib.util.spec_from_file_location(f"test_disagg_{name}", service_dir / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[name] = module
    return modules


@pytest.mark.parametrize("role", ["controller", "encoder", "transformer", "decoder"])
@pytest.mark.parametrize("seed_args, expected_seed", [([], 42), (["--seed", "0"], 0), (["--seed", "12"], 12)])
def test_service_cli_separates_startup_config_and_automatic_request(service_modules, monkeypatch, tmp_path, role, seed_args, expected_seed):
    config_path = tmp_path / "deployment.json"
    config_path.write_text(json.dumps({"cpu_offload": True}))
    argv = ["run_service", "--service", role, "--model_path", str(tmp_path), "--config_json", str(config_path), "--prompt", "request", "--image_path", "input.png", *seed_args]
    if role == "controller":
        argv.extend(["--save_result_path", "./output.mp4"])
    monkeypatch.setattr(sys, "argv", argv)
    module = service_modules[role]
    service = Mock()
    service_cls = Mock(return_value=service)
    monkeypatch.setattr(module, f"{role.title()}Service", service_cls)
    monkeypatch.setitem(sys.modules, f"lightx2v.disagg.services.{role}", module)
    seed_all = Mock()
    monkeypatch.setattr(run_service, "seed_all", seed_all)

    run_service.main()

    if role == "controller":
        config, request = service.run.call_args.args
        assert request["seed"] == expected_seed
        assert request["prompt"] == "request"
        assert request["image_path"] == "input.png"
        assert request["save_path"] == "./output.mp4"
    else:
        (config,) = service_cls.call_args.args
        service.run.assert_called_once_with()
    assert not {"seed", "prompt", "negative_prompt", "image_path", "save_path"} & config.keys()
    assert config["cpu_offload"] is True
    seed_all.assert_called_once_with(expected_seed)


@pytest.mark.parametrize("seed", [None, 0, 73])
def test_spawn_command_contains_only_startup_arguments(service_modules, tmp_path, seed):
    config_path = tmp_path / "encoder.json"
    config_path.write_text(json.dumps({"cpu_offload": True}))
    controller = service_modules["controller"].ControllerService.__new__(service_modules["controller"].ControllerService)
    command = controller._build_service_command("encoder", 0, {"model_path": str(tmp_path), "seed": seed}, str(config_path))
    args = run_service._build_parser().parse_args(command[3:])

    assert args.seed == 42
    assert not {"--seed", "--prompt", "--negative_prompt", "--image_path", "--save_result_path"} & set(command)


@pytest.fixture
def controller_runtime(service_modules, monkeypatch):
    module = service_modules["controller"]
    controller = module.ControllerService()
    packets = []
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setenv("DISAGG_AUTO_REQUEST_COUNT", "2")
    monkeypatch.setenv("IS_CENTRALIZED", "0")
    monkeypatch.setenv("ENABLE_MONITOR", "0")
    monkeypatch.setenv("DISAGG_INSTANCE_WARMUP_WAIT_S", "0")
    for name in ("_init_gpu_pool", "_init_request_rdma_buffer", "create_instance", "_run_centralized_ok_server", "_dump_controller_metrics"):
        monkeypatch.setattr(controller, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "_drain_decoder_results_non_block", lambda **kwargs: kwargs["received_rooms"].update(kwargs["expected_rooms"]))
    controller.req_mgr = SimpleNamespace(send=lambda host, port, packet: packets.append(dict(packet)))
    controller.rdma_buffer_request = SimpleNamespace(produce=lambda packet: packets.append(dict(packet)))
    for offset, policy in enumerate((controller.encoder_policy, controller.transformer_policy, controller.decoder_policy)):
        monkeypatch.setattr(policy, "schedule", lambda offset=offset: f"127.0.0.1:{module.REQUEST_POLLING_PORT + offset}")
    return controller, packets


@pytest.mark.parametrize("centralized", [False, True])
@pytest.mark.parametrize("startup_config", [{}, {"seed": None}, {"seed": 0}, {"seed": 73}])
@pytest.mark.parametrize("load_from_user", [False, True])
def test_controller_resolves_each_request_before_dispatch(controller_runtime, monkeypatch, centralized, startup_config, load_from_user):
    controller, packets = controller_runtime
    monkeypatch.setenv("LOAD_FROM_USER", str(int(load_from_user)))
    monkeypatch.setenv("IS_CENTRALIZED", str(int(centralized)))
    requests = [{"save_path": f"./external_{index}.mp4", **seed} for index, seed in enumerate([{}, {"seed": None}, {"seed": 0}, {"seed": 12}, {}])]
    workload = iter([*requests, {"workload_end": True}])
    controller.req_mgr.receive = lambda _: next(workload)
    config = dict(startup_config)
    request_data = {"seed": 73, "prompt": "automatic prompt", "negative_prompt": "", "image_path": "input.png", "save_path": "auto.mp4"}
    original_request = dict(request_data)

    controller.run(config, request_data)

    expected = [42, 42, 0, 12, 42] if load_from_user else [73, 73]
    replicas = 3 if centralized else 1
    assert [packet["seed"] for packet in packets] == [seed for seed in expected for _ in range(replicas)]
    paths = [request["save_path"] for request in requests] if load_from_user else ["auto0.mp4", "auto1.mp4"]
    assert [packet["save_path"] for packet in packets] == [path for path in paths for _ in range(replicas)]
    for packet in packets:
        if load_from_user:
            assert "prompt" not in packet
            assert "image_path" not in packet
        else:
            assert all(packet[key] == value for key, value in request_data.items() if key != "save_path")
    assert config == startup_config
    assert request_data == original_request


@pytest.mark.parametrize("request_data", [{}, {"save_path": None}, {"save_path": ""}])
@pytest.mark.parametrize("load_from_user", [False, True])
def test_controller_rejects_missing_output_before_dispatch(controller_runtime, monkeypatch, request_data, load_from_user):
    controller, packets = controller_runtime
    monkeypatch.setenv("LOAD_FROM_USER", str(int(load_from_user)))
    workload = iter([request_data, {"workload_end": True}])
    controller.req_mgr.receive = lambda _: next(workload)
    metrics = Mock()
    monkeypatch.setattr(controller, "_dump_controller_metrics", metrics)
    if load_from_user:
        controller.run({"save_path": "startup.mp4"}, request_data)
        result = metrics.call_args.args[0][0]
        assert result["ok"] is False
        assert "save_path is required" in result["error"]
    else:
        with pytest.raises(ValueError, match="save_path is required"):
            controller.run({"save_path": "startup.mp4"}, request_data)
    assert packets == []


def test_invalid_external_request_does_not_interrupt_the_batch(controller_runtime, monkeypatch):
    controller, packets = controller_runtime
    monkeypatch.setenv("LOAD_FROM_USER", "1")
    workload = iter([{"save_path": "first.mp4"}, {"save_path": None}, {"save_path": "last.mp4"}, {"workload_end": True}])
    controller.req_mgr.receive = lambda _: next(workload)
    controller.run({})
    assert [(packet["data_bootstrap_room"], packet["save_path"]) for packet in packets] == [(0, "first.mp4"), (2, "last.mp4")]


def test_automatic_output_names_use_resolved_rooms(controller_runtime, monkeypatch):
    controller, packets = controller_runtime
    monkeypatch.setenv("LOAD_FROM_USER", "0")
    controller.run({"data_bootstrap_room": 7}, {"save_path": "outputs/video.test.mp4"})
    assert [(packet["data_bootstrap_room"], packet["save_path"]) for packet in packets] == [(7, "outputs/video.test7.mp4"), (8, "outputs/video.test8.mp4")]


@pytest.mark.parametrize("request_data", [{}, {"save_path": None}, {"save_path": ""}])
def test_decoder_rejects_missing_output_before_computation(service_modules, monkeypatch, request_data):
    module = service_modules["decoder"]
    service = module.DecoderService.__new__(module.DecoderService)
    seed_all = Mock()
    monkeypatch.setattr(module, "seed_all", seed_all)
    with pytest.raises(ValueError, match="save_path is required"):
        service.process({"seed": 42, **request_data})
    seed_all.assert_not_called()


def test_decoder_saves_to_the_explicit_path(service_modules, monkeypatch):
    module = service_modules["decoder"]
    service = module.DecoderService.__new__(module.DecoderService)
    service.logger = Mock()
    latents = torch.zeros(1, 1, 2, 2)
    meta = json.dumps({"latents_shape": [1, 1, 2, 2], "latents_dtype": "torch.float32"}).encode() + b"\x00"
    service._rdma_buffers = {0: [latents, torch.tensor(list(meta), dtype=torch.uint8)]}
    service.data_receiver = {0: object()}
    service.vae_decoder = SimpleNamespace(decode=lambda tensor: tensor)
    monkeypatch.setattr(module, "AI_DEVICE", "cpu")
    monkeypatch.setattr(module, "GET_DTYPE", lambda: torch.float32)
    monkeypatch.setattr(module, "seed_all", lambda _: None)
    monkeypatch.setattr(module, "wan_vae_to_comfy", lambda video: video)
    save_to_video = Mock()
    monkeypatch.setattr(module, "save_to_video", save_to_video)

    assert service.process({"seed": 42, "save_path": "./output.mp4"}) == "./output.mp4"
    save_to_video.assert_called_once()
    assert save_to_video.call_args.args[1] == "./output.mp4"


@pytest.mark.parametrize("seed_args, expected_seed", [([], 42), (["--seed", "0"], 0), (["--seed", "73"], 73)])
def test_user_workload_carries_request_inputs_and_unique_outputs(monkeypatch, tmp_path, seed_args, expected_seed):
    argv = ["run_user", "--prompt", "request prompt", "--negative_prompt", "", "--image_path", "input.png", "--max_requests", "2", *seed_args]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("DISAGG_WORKLOAD_SAVE_PREFIX", str(tmp_path / "output.mp4"))
    stage = StageSpec("test", duration_s=10, user_count=1, spawn_rate=1)
    monkeypatch.setattr(run_user, "load_stage_specs", lambda: [stage])
    monkeypatch.setattr(run_user, "load_base_config", lambda: {"task": "i2v"})
    monkeypatch.setattr(run_user, "DisaggLoadShape", lambda: SimpleNamespace(tick=lambda: (1, 1)))
    monkeypatch.setattr(run_user, "current_stage", lambda _: stage)
    monkeypatch.setattr(run_user.time, "sleep", lambda _: None)
    monkeypatch.setattr(run_user, "send_workload_end_signal", lambda: None)
    packets = []
    monkeypatch.setattr(run_user, "ReqManager", lambda: SimpleNamespace(send=lambda host, port, packet: packets.append(packet)))

    run_user.main()

    assert [packet["save_path"] for packet in packets] == [str(tmp_path / "output_test_0.mp4"), str(tmp_path / "output_test_1.mp4")]
    assert all(packet["seed"] == expected_seed and packet["prompt"] == "request prompt" and packet["image_path"] == "input.png" and packet["negative_prompt"] == "" for packet in packets)


@pytest.mark.parametrize("name", ["encoder", "transformer", "decoder"])
def test_services_seed_when_computation_starts(service_modules, monkeypatch, name):
    module = service_modules[name]
    service_class = getattr(module, f"{name.title()}Service")
    service = service_class.__new__(service_class)
    seeded = []

    class ComputationReached(Exception):
        pass

    def seed_at_computation(seed):
        seeded.append(seed)
        raise ComputationReached

    monkeypatch.setattr(module, "seed_all", seed_at_computation)
    for seed in (0, 73):
        with pytest.raises(ComputationReached):
            service.process({"seed": seed, "save_path": "./output.mp4"})
    assert seeded == [0, 73]


@pytest.mark.parametrize("seed", [0, 73])
def test_encoder_phase1_packet_keeps_the_resolved_seed(service_modules, monkeypatch, seed):
    module = service_modules["encoder"]
    service = module.EncoderService.__new__(module.EncoderService)
    service.logger = SimpleNamespace(info=lambda *args: None)
    service.text_encoder = SimpleNamespace(infer=lambda prompts: [torch.ones(2, 2)])
    service._rdma_buffers = {1: [torch.empty(size, dtype=torch.uint8) for size in (16, 32, 4096)]}
    service.data_sender = {1: SimpleNamespace(send=lambda ptrs: None)}
    service.data_mgr = SimpleNamespace(get_localhost=lambda: "encoder", get_session_id=lambda: "session")
    service._centralized_request_mode = False
    service.sync_comm = False
    packets = []
    seeded = []
    service._produce_phase1_request_with_retry = lambda room, packet: packets.append(packet)
    monkeypatch.setattr(module, "seed_all", seeded.append)
    monkeypatch.setattr(module, "AI_DEVICE", "cpu")
    monkeypatch.setattr(module, "GET_DTYPE", lambda: torch.float32)
    config = {"seed": seed, "task": "t2v", "prompt": "test", "data_bootstrap_room": 1, "text_len": 2, "target_height": 16, "target_width": 16, "target_video_length": 1, "vae_stride": [4, 8, 8]}

    service.process(config)

    assert seeded == [seed]
    assert packets[0]["request_config"]["seed"] == seed
    assert packets[0]["request_config"] is not config


def test_transformer_discards_previous_request_generator(service_modules, monkeypatch):
    module = service_modules["transformer"]
    service = module.TransformerService.__new__(module.TransformerService)
    service.logger = SimpleNamespace(info=lambda *args: None)
    service.scheduler = WanScheduler.__new__(WanScheduler)
    service.scheduler.generator = torch.Generator().manual_seed(999)
    service.rdma_buffer1 = {}
    service.rdma_buffer2 = {}
    service.data_receiver = {}
    service.data_sender = {}
    service._phase2_remote_rooms = set()
    monkeypatch.setattr(module, "seed_all", lambda seed: None)
    monkeypatch.setattr("lightx2v.models.schedulers.wan.scheduler.AI_DEVICE", "cpu")

    latents = []
    for seed in (73, 12, 73):
        config = {"seed": seed, "infer_steps": 1, "sample_shift": 5, "task": "t2v"}
        # Stop at the first hardware boundary, after real per-request scheduler preparation.
        with pytest.raises(RuntimeError, match="phase1 RDMA buffers"):
            service.process(config)
        service.scheduler.prepare_latents(seed, (1, 1, 2, 2))
        latents.append(service.scheduler.latents.clone())

    assert torch.equal(latents[0], latents[2])
    assert not torch.equal(latents[0], latents[1])
