"""CPU coverage of the disk contract and the actual runner's export hooks.

Load the small utility directly and extract runner methods without importing
the CUDA-only model dependency graph. Tensor packing and scheduler updates are
the production implementations, with only transformer/codec work stubbed.
"""

import argparse
import ast
import importlib.util
import json
import math
import os
import sys
from contextlib import nullcontext, suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


export = _load("h3_teacher_export_under_test", "lightx2v/utils/h3_teacher_export.py")
packing = _load("h3_teacher_packing_under_test", "lightx2v/models/networks/minimax_h3/packing.py")


def _runner_type():
    path = ROOT / "lightx2v/models/runners/minimax_h3/minimax_h3_runner.py"
    parsed = ast.parse(path.read_text())
    original = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3Runner")
    names = {"prepare_request", "_split_reference_paths", "_start_teacher_export", "_write_teacher_state", "_unpack_target_latents", "run_segment", "run_main"}
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for method in methods:
        method.decorator_list = [decorator for decorator in method.decorator_list if isinstance(decorator, ast.Name) and decorator.id == "staticmethod"]
    original.body = methods
    original.bases = [ast.Name(id="StubBase", ctx=ast.Load())]
    original.decorator_list = []
    namespace = {
        **vars(export),
        "os": os,
        "torch": torch,
        "suppress": suppress,
        "StubBase": type("StubBase", (), {"prepare_request": lambda self, data: SimpleNamespace(**data)}),
        "patchify_video_latents": packing.patchify_video_latents,
        "unpatchify_video_tokens": packing.unpatchify_video_tokens,
        "unpack_audio_tokens": packing.unpack_audio_tokens,
        "DEFAULT_REFERENCE_IMAGE_RESIZE_MODE": "diffusers",
        "ProfilingContext4DebugL1": lambda *args, **kwargs: nullcontext(),
        "GET_RECORDER_MODE": lambda: False,
        "monitor_cli": SimpleNamespace(lightx2v_run_per_step_dit_duration=None),
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["MiniMaxH3Runner"]


def _native_schedule():
    path = ROOT / "lightx2v/models/schedulers/minimax_h3/scheduler.py"
    parsed = ast.parse(path.read_text())
    scheduler = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3Scheduler")
    method = next(node for node in scheduler.body if isinstance(node, ast.FunctionDef) and node.name == "_step")
    method.decorator_list = []
    schedule = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "_make_schedule")
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method, schedule], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["_step"], namespace["_make_schedule"]


@pytest.fixture
def make_runner(tmp_path):
    def make(enabled=True, latent_only=True, name="export"):
        runner = _runner_type()()
        model_path = tmp_path / "model"
        transformer_path = model_path / "transformer_ref"
        transformer_path.mkdir(parents=True, exist_ok=True)
        (model_path / "decfg_merged.json").write_text('{"complete": true}')
        reference_path = tmp_path / "reference.png"
        reference_path.write_bytes(b"reference fixture")
        directory = tmp_path / name
        runner.config = {"model_path": str(model_path), "model_variant": "ref2av", "fps": 24}
        runner.loaded_transformer_partition = "transformer_ref"
        runner.input_info = SimpleNamespace(
            task="ref2av",
            seed=42,
            prompt="A character walks.",
            image_path=str(reference_path),
            video_path="",
            audio_path="",
            teacher_latents_dir=str(directory) if enabled else None,
            teacher_latents_only=latent_only if enabled else False,
            teacher_sample_id="sample-001",
            save_result_path=None,
            return_result_tensor=False,
            draft_steps=None,
            resample_source_video=None,
        )
        runner.request_height = runner.request_width = 32
        runner.request_num_frames = 124
        runner.condition_video_latents = [torch.ones(1, 24, 1, 2, 2)]
        runner.condition_audio_latents = []
        runner.prepared_references = [SimpleNamespace(kind="image", has_audio=False, num_latent_frames=1, latent_height=2, latent_width=2, num_audio_latents=0)]
        runner.inputs = {"text_encoder_output": {"prompt_embeds": torch.ones(3, 5120, dtype=torch.bfloat16), "text_token_tags": torch.tensor([0, 1, 1])}}
        generator = torch.Generator().manual_seed(42)
        target_video = torch.randn(1, 24, 37, 2, 2, generator=generator)
        target_audio = torch.randn(414, 32, generator=generator)
        scheduler = SimpleNamespace(
            infer_steps=29,
            start_step_index=0,
            draft_steps=None,
            scheduler_type="reference_blend",
            video_shift=12.0,
            audio_shift=3.0,
            num_latent_frames=37,
            latent_height=2,
            latent_width=2,
            num_audio_latents=207,
            num_condition_video_rows=1,
            num_condition_audio_rows=0,
            video_latents=torch.cat([torch.full((1, 96), 7.0), packing.patchify_video_latents(target_video)]),
            audio_latents=target_audio,
        )
        native_step, make_schedule = _native_schedule()
        for kind, shift in (("video", 12.0), ("audio", 3.0)):
            sigmas, timesteps = make_schedule(29, shift, "cpu")
            setattr(scheduler, f"{kind}_sigmas", sigmas)
            setattr(scheduler, f"{kind}_timesteps", timesteps)
        scheduler.step_pre = lambda index: setattr(scheduler, "step_index", index)
        observed = []

        def infer(inputs):
            observed.append((scheduler.video_latents[1:].clone(), scheduler.audio_latents.clone()))
            scheduler.video_noise_pred = torch.full_like(scheduler.video_latents, 0.25)
            scheduler.audio_noise_pred = torch.full_like(scheduler.audio_latents, -0.5)

        def step_post():
            for kind, prefix in (("video", 1), ("audio", 0)):
                latents = getattr(scheduler, f"{kind}_latents")
                latents[prefix:] = native_step(
                    latents[prefix:],
                    getattr(scheduler, f"{kind}_noise_pred")[prefix:],
                    getattr(scheduler, f"{kind}_timesteps")[scheduler.step_index],
                    getattr(scheduler, f"{kind}_sigmas"),
                    scheduler.step_index,
                    scheduler.scheduler_type,
                )

        scheduler.step_post = step_post
        runner.scheduler = scheduler
        runner.model = SimpleNamespace(infer=infer)
        runner.progress_callback = None
        runner.check_stop = lambda: None
        runner._prepare_resampling_source = lambda: None
        runner.init_run = lambda: None
        runner._release_low_memory_vae = lambda: None
        runner.end_run = lambda: None
        runner.decode_calls = []

        def decode(video, audio):
            assert not (directory / "manifest.json").exists()
            runner.decode_calls.append(True)
            return video, audio

        runner.run_vae_decoder = decode
        runner.process_images_after_vae_decoder = lambda: {"video": None, "audio": None}
        return runner, observed, directory

    return make


def test_export_all_inputs_and_clean_endpoint_match_native_schedule(make_runner):
    runner, observed, directory = make_runner()
    result = runner.run_main()
    assert runner.decode_calls == []
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["schema"] == "h3_teacher_trajectory_v1"
    assert manifest["num_steps"] == 29 and manifest["num_states"] == 30
    assert manifest["video_shape"] == [24, 37, 2, 2]
    assert manifest["audio_shape"] == [2, 32, 207]
    assert manifest["source_id"] == "sample-001"
    assert manifest["decfg_merged_identity"][0]["metadata"]["complete"] is True
    assert result["teacher_manifest_path"] == str(directory / "manifest.json")
    for index, (video_rows, audio_rows) in enumerate(observed):
        state = torch.load(directory / manifest["state_paths"][index], weights_only=True)
        expected_video = packing.unpatchify_video_tokens(video_rows, 37, 2, 2)[0]
        assert torch.equal(state["video_latents"], expected_video)
        assert torch.equal(state["audio_latents"], packing.unpack_audio_tokens(audio_rows, 207))
        assert state["video_latents"].dtype == state["audio_latents"].dtype == torch.float32
        assert state["normalized"] and not state["is_clean"]
        assert state["video_sigma"] == manifest["video_sigmas"][index]
    assert len(observed) == 29
    assert torch.equal(runner.scheduler.video_latents[0], torch.full((96,), 7.0))
    clean = torch.load(directory / "state_029.pt", weights_only=True)
    assert clean["is_clean"] and clean["video_sigma"] == clean["audio_sigma"] == 0
    assert torch.equal(clean["video_latents"], torch.load(directory / "video_latents.pt", weights_only=True)["latents"])
    assert torch.equal(clean["audio_latents"], torch.load(directory / "audio_latents.pt", weights_only=True))
    condition = torch.load(directory / "condition.pt", weights_only=True)
    positive = condition["conditioning"]["positive"]
    assert condition["cache_fingerprint"] == manifest["cache_fingerprint"]
    assert condition["prompt"] == manifest["prompt"]
    assert positive["prompt_embeds"].shape == (1, 3, 5120)
    assert positive["target_num_frames"] == 124
    assert positive["references"][0]["path"] == manifest["image_paths"][0]
    assert torch.equal(positive["references"][0]["video_latents"], torch.ones(1, 96))


def test_disabled_export_and_smoke_decode_keep_native_values(make_runner):
    plain, plain_inputs, plain_dir = make_runner(enabled=False, name="plain")
    smoke, smoke_inputs, smoke_dir = make_runner(latent_only=False, name="smoke")
    plain.run_main()
    smoke.run_main()
    assert plain.decode_calls == smoke.decode_calls == [True]
    assert not plain_dir.exists()
    assert (smoke_dir / "manifest.json").is_file()
    assert all(torch.equal(a, b) for left, right in zip(plain_inputs, smoke_inputs) for a, b in zip(left, right))
    assert torch.equal(plain.scheduler.video_latents, smoke.scheduler.video_latents)
    assert torch.equal(plain.scheduler.audio_latents, smoke.scheduler.audio_latents)


def test_decode_failure_does_not_publish_completion(make_runner):
    runner, _, directory = make_runner(latent_only=False)

    def fail(*args):
        raise RuntimeError("decode failed")

    runner.run_vae_decoder = fail
    with pytest.raises(RuntimeError, match="decode failed"):
        runner.run_main()
    assert (directory / "state_029.pt").is_file()
    assert not (directory / "manifest.json").exists()


def test_incomplete_out_of_order_and_completed_directory_rejected(make_runner):
    runner, _, directory = make_runner()
    runner._start_teacher_export()
    with pytest.raises(RuntimeError, match="incomplete"):
        runner._teacher_writer.finish()
    with pytest.raises(ValueError, match="Expected teacher state"):
        runner._write_teacher_state(1)
    assert not (directory / "manifest.json").exists()
    runner.run_main()
    with pytest.raises(FileExistsError, match="already exists"):
        runner._start_teacher_export()


def test_atomic_save_failure_leaves_no_completion_or_partial_state(make_runner, monkeypatch):
    runner, _, directory = make_runner()
    runner._start_teacher_export()

    def fail(value, handle):
        handle.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(export.torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        runner._write_teacher_state(0)
    assert not (directory / "state_000.pt").exists()
    assert not (directory / "manifest.json").exists()
    assert list(directory.glob("*.tmp")) == []


def test_nonzero_rank_does_not_write_and_observes_rank_zero_failure(monkeypatch):
    monkeypatch.setattr(export.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(export.dist, "get_rank", lambda: 1)
    calls = []

    def broadcast(error, src):
        assert src == 0
        error[0] = "OSError: disk full"

    monkeypatch.setattr(export.dist, "broadcast_object_list", broadcast)
    with pytest.raises(RuntimeError, match="disk full"):
        export.run_rank_zero(lambda: calls.append("write"))
    assert calls == []


def test_all_decode_ranks_participate_before_propagating_save_failure(monkeypatch):
    monkeypatch.setattr(export.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(export.dist, "get_world_size", lambda: 2)
    calls = []

    def gather(errors, error):
        assert error is None
        errors[:] = ["OSError: MP4 disk full", None]

    monkeypatch.setattr(export.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="MP4 disk full"):
        export.run_collective_action(lambda: calls.append("decode"))
    assert calls == ["decode"]


def test_latent_only_request_requires_destination_and_skips_output_path(make_runner):
    runner, _, _ = make_runner()
    request = vars(runner.input_info).copy()
    request["save_result_path"] = "unused.mp4"
    prepared = runner.prepare_request(request)
    assert prepared.save_result_path is None
    request["teacher_latents_dir"] = None
    with pytest.raises(ValueError, match="require teacher_latents_dir"):
        runner.prepare_request(request)
    request["teacher_latents_dir"] = "export"
    request["draft_steps"] = 3
    with pytest.raises(ValueError, match="complete schedule"):
        runner.prepare_request(request)


def test_http_cli_and_pipeline_preserve_export_request_fields():
    requested = {"teacher_latents_dir": "/tmp/export", "teacher_latents_only": True, "teacher_sample_id": "sample-1"}
    path = ROOT / "lightx2v/server/schema.py"
    parsed = ast.parse(path.read_text())
    parsed.body = [node for node in parsed.body if not isinstance(node, ast.ImportFrom) or not node.level]
    namespace = {"__name__": "h3_test_schema", "generate_task_id": lambda: "task-1"}
    exec(compile(parsed, str(path), "exec"), namespace)
    request = namespace["VideoTaskRequest"](**requested)
    assert all(getattr(request, key) == value for key, value in requested.items())
    assert set(requested) <= request.model_fields_set
    assert namespace["VideoTaskRequest"]().teacher_latents_only is False

    path = ROOT / "lightx2v/infer.py"
    parser = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "build_parser")
    namespace = {"argparse": argparse, "json": json, "RUNNER_MODULES": ["minimax_h3"]}
    exec(compile(ast.Module(body=[parser], type_ignores=[]), str(path), "exec"), namespace)
    arguments = namespace["build_parser"]().parse_args(
        ["--model_cls", "minimax_h3", "--model_path", "model", "--config_json", "config", "--teacher_latents_dir", "/tmp/export", "--teacher_latents_only", "--teacher_sample_id", "sample-1"]
    )
    assert all(getattr(arguments, key) == value for key, value in requested.items())

    path = ROOT / "lightx2v/pipeline.py"
    pipeline = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "LightX2VPipeline")
    generate = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef) and node.name == "generate")
    generate.decorator_list = []
    namespace = {"logger": SimpleNamespace(info=lambda *args: None)}
    exec(compile(ast.Module(body=[generate], type_ignores=[]), str(path), "exec"), namespace)
    pipeline = SimpleNamespace(task="ref2av", runner=SimpleNamespace(prepare_request=lambda data: data, run_request=lambda data: data))
    result = namespace["generate"](pipeline, **requested)
    assert all(result[key] == value for key, value in requested.items())


def test_model_identity_follows_component_links_and_records_changes(tmp_path):
    model = tmp_path / "model"
    component = tmp_path / "vae"
    model.mkdir()
    component.mkdir()
    (component / "weights.pt").write_bytes(b"weights")
    (model / "vae").symlink_to(component, target_is_directory=True)
    (component / "cycle").symlink_to(model, target_is_directory=True)
    before = export.path_identity(model)
    assert before["files"][0]["path"] == "vae/weights.pt"
    (component / "weights.pt").write_bytes(b"different weights")
    assert export.fingerprint_metadata(before) != export.fingerprint_metadata(export.path_identity(model))
