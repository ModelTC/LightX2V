import importlib
import importlib.util
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lightx2v.models.runners.bagel import sensenova_vision_runner as sensenova
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.hunyuan3d.hunyuan3d_shape_runner import Hunyuan3DShapeRunner
from lightx2v.models.runners.motus import motus_runner as motus
from lightx2v.models.runners.swiftvr import swiftvr_runner as swiftvr
from lightx2v.models.runners.wan import wan_dreamzero_runner as dreamzero
from lightx2v.models.runners.wan.fastwam_runner import FastWAMRunner
from lightx2v.models.runners.wan.wan_lingbot_va_runner import LingbotVARunner
from lightx2v.models.runners.worldmirror import worldmirror_runner as worldmirror
from lightx2v.pipeline import LightX2VPipeline


def make_runner(cls, task, **config):
    runner = object.__new__(cls)
    BaseRunner.__init__(runner, {"task": task, **config})
    runner._gc_frozen = True
    runner.progress_callback = None
    return runner


@pytest.mark.parametrize("request_output", [{}, {"save_result_path": None}, {"save_result_path": ""}, {"save_result_path": "mesh.glb"}])
def test_hunyuan3d_requires_mesh_output(request_output, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = make_runner(Hunyuan3DShapeRunner, "i23d", enable_pbar=False)
    runner.run_input_encoder = lambda: {"image_tensor": torch.zeros(1), "cond": {}, "guidance_scale": 1, "do_classifier_free_guidance": False}
    mesh = Mock()
    runner.vae_decoder = SimpleNamespace(vae=SimpleNamespace(latent_shape=(1,)), decode_mesh=Mock(return_value=mesh))
    runner.scheduler = SimpleNamespace(prepare=Mock(), infer_steps=1, latents=torch.zeros(1))
    runner.model = SimpleNamespace(transformer_infer=SimpleNamespace(fi_moe_autotune=SimpleNamespace(cache_rebuild_needed=lambda: False, session=lambda **kwargs: nullcontext())))
    runner._run_infer_step = Mock()
    runner.end_run = Mock()
    pipeline = object.__new__(LightX2VPipeline)
    pipeline.runner = runner
    pipeline.task = "i23d"

    if not request_output.get("save_result_path"):
        with pytest.raises(ValueError, match="save_result_path must be set"):
            pipeline.generate(image_path="input.png", **request_output)
        mesh.export.assert_not_called()
        return

    result = pipeline.generate(image_path="input.png", **request_output)

    runner._run_infer_step.assert_called_once()
    runner.vae_decoder.decode_mesh.assert_called_once()
    assert result == request_output.get("save_result_path")
    assert mesh.export.call_count == (result is not None)
    runner.end_run.assert_called_once()


@pytest.mark.parametrize("kind", ["image", "video"])
def test_swiftvr_saves_restored_output(kind, tmp_path, monkeypatch):
    runner = make_runner(swiftvr.SwiftVRRunner, "sr", clip_len=4)
    runner.init_device = torch.device("cpu")
    runner.copy_stream = None
    runner.restorer = SimpleNamespace(reset=Mock())
    runner.check_stop = Mock()
    runner.read_image_frame = lambda path: (torch.zeros(1, 3, 8, 8), 8, 8)

    class Reader:
        def __init__(self, path):
            pass

        def __len__(self):
            return 9

        def __getitem__(self, index):
            return np.zeros((8, 8, 3))

        def get_avg_fps(self):
            return 24

        def get_batch(self, indices):
            return torch.zeros(len(indices), 8, 8, 3)

    monkeypatch.setattr(swiftvr, "VideoReader", Reader)

    def restore(frames, chunk, clip_latents, height, width, pad_height, pad_width, stage_marks=None):
        if stage_marks is not None:
            stage_marks.extend([time.perf_counter(), time.perf_counter()])
        return torch.zeros(1, chunk.frame_count, 3, height, width)

    runner.restore_frames = Mock(side_effect=restore)
    image_save = Mock()
    writer = Mock()
    runner.open_video_writer = Mock(return_value=writer)
    copy_frames = Mock(wraps=runner.copy_frames_to_cpu)
    runner.copy_frames_to_cpu = copy_frames
    mux = Mock()
    monkeypatch.setattr(swiftvr, "save_to_image", image_save)
    monkeypatch.setattr(swiftvr, "mux_audio_from_video", mux)
    output_path = str(tmp_path / "output" / ("result.png" if kind == "image" else "result.mp4"))

    input_info = runner.prepare_request({"task": runner.config["task"], f"{kind}_path": "input", "save_result_path": output_path})
    result = runner.run_request(input_info)

    assert result["stats"]["output"] == output_path
    assert result["stats"]["frames"] == (1 if kind == "image" else 9)
    assert result["images" if kind == "image" else "video"] is None
    assert runner.restore_frames.call_count == (1 if kind == "image" else 2)
    assert runner.restorer.reset.call_count == 2
    assert image_save.call_count == (kind == "image")
    assert runner.open_video_writer.call_count == (kind == "video")
    assert mux.call_count == (kind == "video")
    if kind == "video":
        assert writer.append_data.call_count == 9
        writer.close.assert_called_once()
        assert copy_frames.call_count == 2
    assert (tmp_path / "output").is_dir()


@pytest.mark.parametrize("kind", ["image", "video"])
@pytest.mark.parametrize("request_output", [{}, {"save_result_path": None}, {"save_result_path": ""}])
def test_swiftvr_requires_output_path_before_reading_input(kind, request_output, monkeypatch):
    runner = make_runner(swiftvr.SwiftVRRunner, "sr")
    runner.read_image_frame = Mock()
    reader = Mock()
    monkeypatch.setattr(swiftvr, "VideoReader", reader)
    input_info = runner.prepare_request({f"{kind}_path": "input", **request_output})

    with pytest.raises(ValueError, match="requires `save_result_path`"):
        runner.run_request(input_info)

    runner.read_image_frame.assert_not_called()
    reader.assert_not_called()


@pytest.mark.parametrize("source", ["request", "startup"])
def test_swiftvr_video_rejects_tensor_output(source, monkeypatch):
    config = {"return_result_tensor": True} if source == "startup" else {}
    runner = make_runner(swiftvr.SwiftVRRunner, "sr", **config)
    runner.read_image_frame = Mock()
    reader = Mock()
    monkeypatch.setattr(swiftvr, "VideoReader", reader)
    request = {"video_path": "input", "save_result_path": "output"}
    if source == "request":
        request["return_result_tensor"] = True

    with pytest.raises(ValueError, match="return_result_tensor"):
        input_info = runner.prepare_request(request)
        runner.run_request(input_info)

    runner.read_image_frame.assert_not_called()
    reader.assert_not_called()


@pytest.mark.parametrize("request_output", [{}, {"save_result_path": "unused.png"}])
def test_swiftvr_image_returns_tensor_without_saving(request_output, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = make_runner(swiftvr.SwiftVRRunner, "sr", clip_len=4)
    runner.restorer = SimpleNamespace(reset=Mock())
    runner.read_image_frame = Mock(return_value=(torch.zeros(1, 3, 8, 8), 8, 8))
    restored = torch.linspace(0, 1, 3 * 16 * 16, dtype=torch.float16).reshape(1, 1, 3, 16, 16)
    runner.restore_frames = Mock(return_value=restored)
    image_save = Mock()
    monkeypatch.setattr(swiftvr, "save_to_image", image_save)
    input_info = runner.prepare_request({"image_path": "input", "return_result_tensor": True, **request_output})

    result = runner.run_request(input_info)

    torch.testing.assert_close(result["images"], restored[0].permute(0, 2, 3, 1).float())
    assert result["images"].device.type == "cpu"
    assert result["stats"]["output"] is None
    runner.restore_frames.assert_called_once()
    image_save.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("paths", [{}, {"save_result_path": "output"}, {"strict_output_path": "exact"}])
@pytest.mark.parametrize("rank", [0, 1])
def test_worldmirror_saves_only_to_explicit_paths(paths, rank, monkeypatch):
    runner = make_runner(worldmirror.WorldMirrorRunner, "recon", output_path="ignored-json-default", apply_sky_mask=False, apply_edge_mask=False, log_time=False)
    runner.rank = rank
    runner.is_distributed = True
    runner._run_inference = Mock(return_value=({}, torch.zeros(1, 1, 3, 8, 8), 0.1))
    monkeypatch.setattr(worldmirror, "prepare_input", lambda *args, **kwargs: (["input.png"], "scene"))
    monkeypatch.setattr(worldmirror, "compute_adaptive_target_size", lambda paths, size: size)
    save = Mock()
    monkeypatch.setattr(worldmirror, "save_results", save)
    barrier = Mock()
    monkeypatch.setattr(worldmirror.dist, "barrier", barrier)

    input_info = runner.prepare_request({"task": runner.config["task"], "input_path": "input.png", **paths})
    result = runner.run_request(input_info)

    runner._run_inference.assert_called_once()
    assert save.call_count == (bool(paths) and rank == 0)
    assert (result["output_dir"] is not None) == bool(paths)
    if "strict_output_path" in paths:
        assert result["output_dir"] == "exact"
    barrier.assert_called_once()


@pytest.mark.parametrize("runner_cls", [dreamzero.WanDreamZeroRunner, LingbotVARunner])
@pytest.mark.parametrize("output", [None, "video.mp4"])
@pytest.mark.parametrize("return_tensor", [False, True])
def test_action_video_output(runner_cls, output, return_tensor, tmp_path, monkeypatch):
    runner = make_runner(runner_cls, "i2va")
    runner.input_info = runner.prepare_request({"task": runner.config["task"], "save_result_path": str(tmp_path / output) if output else None, "return_result_tensor": return_tensor})
    runner.gen_video = torch.zeros(1)
    runner.pred_action = torch.ones(1, 2, 3)
    save = Mock()
    monkeypatch.setattr(importlib.import_module(runner_cls.__module__), "save_to_video", save)

    if output is None:
        with pytest.raises(ValueError, match="requires save_result_path"):
            runner.process_images_after_vae_decoder()
        save.assert_not_called()
        assert not list(tmp_path.iterdir())
        return

    result = runner.process_images_after_vae_decoder()

    save.assert_called_once()
    assert (tmp_path / "video.actions.npy").exists()
    assert (result["video"] is runner.gen_video) == return_tensor
    if return_tensor:
        assert result["actions"] is runner.pred_action


@pytest.mark.parametrize("output", [None, "video.mp4"])
@pytest.mark.parametrize("action_output", [None, "actions.json"])
def test_motus_requires_video_output(output, action_output, tmp_path, monkeypatch):
    runner = make_runner(motus.MotusRunner, "i2v")
    request = {"image_path": "input.png", "state_path": "state.json", "prompt": "move", "save_result_path": str(tmp_path / output) if output else None}
    if action_output:
        request["save_action_path"] = str(tmp_path / action_output)
    runner._load_state_value = lambda path: [0]
    runner.run_input_encoder = lambda: {"motus_state": torch.zeros(1), "image_encoder_output": None}
    runner.model = SimpleNamespace(prepare_runtime_inputs=lambda inputs, **kwargs: inputs, action_chunk_size=1, action_dim=1, postprocess_actions=lambda: torch.ones(1, 1))
    runner.scheduler = SimpleNamespace(prepare=Mock(), infer_steps=0, video_latents=torch.zeros(1, 1))
    runner.run_vae_decoder = Mock(return_value=torch.zeros(1))
    runner.end_run = Mock()
    monkeypatch.setattr(motus, "wan_vae_to_comfy", lambda video: video)
    save = Mock()
    monkeypatch.setattr(motus, "save_to_video", save)

    input_info = runner.prepare_request({"task": runner.config["task"], **request})
    if output is None:
        with pytest.raises(ValueError, match="requires `save_result_path`"):
            runner.run_request(input_info)
        save.assert_not_called()
        runner.run_vae_decoder.assert_not_called()
        assert not list(tmp_path.iterdir())
        return

    result = runner.run_request(input_info)

    save.assert_called_once()
    expected_action = action_output or "video.actions.json"
    assert [path.name for path in tmp_path.iterdir()] == [expected_action]
    assert torch.equal(result["actions"], torch.ones(1, 1))
    runner.end_run.assert_called_once()


@pytest.mark.parametrize("paths", [{}, {"save_action_path": "actions.npy"}, {"save_result_path": "video.mp4"}])
@pytest.mark.parametrize("return_tensor", [False, True])
def test_fastwam_requires_action_output(paths, return_tensor, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = make_runner(FastWAMRunner, "i2va")
    runner._load_image_pair = lambda: (None, None)
    runner._load_state = lambda: [0] * 8
    actions = np.ones((2, 8), dtype=np.float32)
    runner.policy = SimpleNamespace(predict_action_chunk=Mock(return_value=actions))

    input_info = runner.prepare_request({"task": runner.config["task"], "prompt": "move", "return_result_tensor": return_tensor, **paths})
    if not paths:
        with pytest.raises(ValueError, match="requires `save_action_path` or `save_result_path`"):
            runner.run_request(input_info)
        assert not list(tmp_path.iterdir())
        return

    result = runner.run_request(input_info)

    assert result["actions"] is (actions if return_tensor else None)
    expected_path = paths.get("save_action_path", "video.actions.npy")
    np.testing.assert_array_equal(np.load(tmp_path / expected_path), actions)


@pytest.mark.parametrize("paths", [{}, {"save_result_path": "points.npy"}, {"raw_output_path": "points.npy", "glb_output_path": "scene.glb"}, {"glb_output_path": "scene.glb"}])
def test_sensenova_reconstruction_optional_files(paths, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = make_runner(sensenova.SenseNovaVisionRunner, "omni_vision_task", postprocess_predictions=True)
    info = runner.prepare_request({"omni_vision_subtask": "recon3d", **paths})
    scene = Mock()
    monkeypatch.setattr(sensenova, "load_official_postprocess", lambda source: lambda *args, **kwargs: scene)
    pointmaps = np.zeros((1, 2, 2, 3))

    points, result_scene, raw_path, glb_path = runner._postprocess_recon3d(pointmaps, SimpleNamespace(preprocessed_images=[None]), info, 1)

    assert result_scene is scene
    assert np.array_equal(points, pointmaps)
    has_raw = "save_result_path" in paths or "raw_output_path" in paths
    assert (tmp_path / "points.npy").exists() == has_raw
    assert raw_path == ("points.npy" if has_raw else None)
    assert bool(glb_path) == bool(paths)
    assert scene.export.call_count == bool(paths)
    if not paths:
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("request_data", [{}, {"negative_prompt": ""}, {"negative_prompt": "user negative"}])
def test_dreamzero_uses_request_negative_prompt(request_data):
    runner = make_runner(dreamzero.WanDreamZeroRunner, "i2va", enable_cfg=True)
    runner.input_info = runner.prepare_request({"task": runner.config["task"], "prompt": "move", **request_data})
    consumed = []
    runner.run_text_encoder = lambda info: consumed.append(info.negative_prompt)

    runner._run_input_encoder_local_i2va()

    assert consumed == [request_data.get("negative_prompt", "")]
    assert runner.input_info.prompt == "move"


def test_dreamzero_action_path_still_requires_video(tmp_path, monkeypatch):
    runner = make_runner(dreamzero.WanDreamZeroRunner, "i2va")
    runner.input_info = runner.prepare_request({"task": runner.config["task"], "save_action_path": str(tmp_path / "actions.npy")})
    runner.gen_video = torch.zeros(1)
    runner.pred_action = torch.ones(1, 2)
    save = Mock()
    monkeypatch.setattr(dreamzero, "save_to_video", save)

    with pytest.raises(ValueError, match="requires save_result_path"):
        runner.process_images_after_vae_decoder()

    save.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("action_path", [None, "custom.actions.npy", "absolute"])
def test_dreamzero_action_output_path(action_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = make_runner(dreamzero.WanDreamZeroRunner, "i2va")
    if action_path == "absolute":
        action_path = str(tmp_path / "absolute.actions.npy")
    runner.input_info = runner.prepare_request({"save_result_path": "output/video.mp4", "save_action_path": action_path})
    runner.gen_video = torch.zeros(1)
    runner.pred_action = torch.ones(1, 2)
    save = Mock()
    monkeypatch.setattr(dreamzero, "save_to_video", save)

    runner.process_images_after_vae_decoder()

    save.assert_called_once()
    expected_path = tmp_path / "output" / (action_path or "video.actions.npy")
    np.testing.assert_array_equal(np.load(expected_path), runner.pred_action.numpy())


@pytest.mark.parametrize("output_path", [None, "result.mp4"])
def test_seko_client_downloads_only_saved_results(output_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/server/post_seko_talk_ar.py"
    spec = importlib.util.spec_from_file_location("seko_client", script)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    monkeypatch.setattr(sys, "argv", [str(script)])
    monkeypatch.setattr(client, "submit_task", lambda args: "task1")
    monkeypatch.setattr(client, "wait_task_done", lambda *args: {"status": "completed", "save_result_path": output_path})
    download = Mock()
    monkeypatch.setattr(client, "download_result", download)

    client.main()

    assert download.call_count == (output_path is not None)
