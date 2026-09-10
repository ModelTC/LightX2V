import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).parents[3]


class _Profiler:
    def __init__(self, *_args, **_kwargs):
        pass

    def __call__(self, func):
        return func

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _install_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_runner_module(monkeypatch):
    for package_name in [
        "lightx2v",
        "lightx2v.models",
        "lightx2v.models.audio_encoders",
        "lightx2v.models.audio_encoders.hf",
        "lightx2v.models.input_encoders",
        "lightx2v.models.input_encoders.hf",
        "lightx2v.models.networks",
        "lightx2v.models.networks.minimax_h3",
        "lightx2v.models.runners",
        "lightx2v.models.runners.default_runner",
        "lightx2v.models.schedulers",
        "lightx2v.models.video_encoders",
        "lightx2v.models.video_encoders.hf",
        "lightx2v.models.video_encoders.hf.ltx2",
        "lightx2v.models.video_encoders.hf.ltx2.audio_vae",
        "lightx2v.server",
        "lightx2v.utils",
        "lightx2v_platform",
        "lightx2v_platform.base",
    ]:
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    request_name = "lightx2v.models.runners.request_fields"
    request_spec = importlib.util.spec_from_file_location(request_name, REPO_ROOT / "lightx2v/models/runners/request_fields.py")
    request_module = importlib.util.module_from_spec(request_spec)
    monkeypatch.setitem(sys.modules, request_name, request_module)
    request_spec.loader.exec_module(request_module)

    class DefaultRunner:
        def __init__(self, config):
            self.config = config

        def maybe_empty_cache(self, **_kwargs):
            return False

        def end_run(self):
            pass

    _install_module(monkeypatch, "lightx2v.models.runners.default_runner", DefaultRunner=DefaultRunner)
    _install_module(monkeypatch, "lightx2v.models.audio_encoders.hf.minimax_h3", MiniMaxH3AudioVAE=object)
    _install_module(monkeypatch, "lightx2v.models.input_encoders.hf.minimax_h3", MiniMaxH3Qwen3VLTextEncoder=object)
    _install_module(monkeypatch, "lightx2v.models.networks.minimax_h3.lora", MiniMaxH3LoraAdapter=object)
    _install_module(monkeypatch, "lightx2v.models.networks.minimax_h3.model", MiniMaxH3Model=object)
    _install_module(monkeypatch, "lightx2v.models.schedulers.minimax_h3", MiniMaxH3Scheduler=object)
    _install_module(monkeypatch, "lightx2v.models.video_encoders.hf.minimax_h3", MiniMaxH3VideoVAE=object)

    class Audio:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    _install_module(monkeypatch, "lightx2v.models.video_encoders.hf.ltx2.audio_vae.ops", Audio=Audio)

    class _Metrics:
        def __getattr__(self, _name):
            return None

    _install_module(monkeypatch, "lightx2v.server.metrics", monitor_cli=_Metrics())
    _install_module(monkeypatch, "lightx2v.utils.envs", DTYPE_MAP={"fp32": torch.float32}, GET_RECORDER_MODE=lambda: None)
    _install_module(
        monkeypatch,
        "lightx2v.utils.input_info",
        INPUT_INFO_TYPES={task: object for task in ("t2av", "i2av", "l2av", "fl2av", "ref2av")},
        FL2AVInputInfo=object,
        I2AVInputInfo=object,
        L2AVInputInfo=object,
        Ref2AVInputInfo=object,
        T2AVInputInfo=object,
    )
    _install_module(monkeypatch, "lightx2v.utils.ltx2_media_io", encode_video=lambda **_kwargs: None)
    _install_module(monkeypatch, "lightx2v.utils.profiler", ProfilingContext4DebugL1=_Profiler, ProfilingContext4DebugL2=_Profiler)
    _install_module(monkeypatch, "lightx2v.utils.registry_factory", RUNNER_REGISTER=lambda _name: lambda cls: cls)
    _install_module(monkeypatch, "lightx2v_platform.base.global_var", AI_DEVICE="mps")

    packing_names = {
        "TEXT_TAG": 1,
        "align_num_frames": lambda value: value,
        "prepare_keyframe_image": lambda image, *_args, **_kwargs: image,
        "resolve_canvas_size": lambda width, height: (height, width),
        "unpack_audio_tokens": lambda rows, *_args, **_kwargs: rows,
        "unpatchify_video_tokens": lambda rows, *_args, **_kwargs: rows,
        "validate_t2av_geometry": lambda *_args, **_kwargs: None,
    }
    _install_module(monkeypatch, "lightx2v.models.networks.minimax_h3.packing", **packing_names)
    _install_module(
        monkeypatch,
        "lightx2v.models.networks.minimax_h3.packing_ref2av",
        DEFAULT_REFERENCE_IMAGE_RESIZE_MODE="contain",
        MAX_REFERENCES=12,
        MAX_REFERENCE_AUDIOS=3,
        MAX_REFERENCE_IMAGES=9,
        MAX_REFERENCE_VIDEOS=3,
        REFERENCE_IMAGE_RESIZE_MODES=("contain",),
        MiniMaxH3PreparedReference=object,
        decode_reference_audio=lambda *_args, **_kwargs: None,
        decode_reference_video=lambda *_args, **_kwargs: None,
        prepare_reference_frames=lambda frames, *_args, **_kwargs: frames,
        prepare_reference_image=lambda image, *_args, **_kwargs: image,
        prepare_reference_waveform=lambda waveform, *_args, **_kwargs: waveform,
        resample_reference_frames=lambda frames, *_args, **_kwargs: frames,
        resolve_reference_image_size=lambda width, height, **_kwargs: (height, width),
        trim_reference_num_frames=lambda value: value,
    )

    module_path = REPO_ROOT / "lightx2v/models/runners/minimax_h3/minimax_h3_runner.py"
    spec = importlib.util.spec_from_file_location("minimax_h3_runner_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.torch_device_module = SimpleNamespace(synchronize=lambda: None)
    return module


@pytest.fixture()
def runner_module(monkeypatch):
    return _load_runner_module(monkeypatch)


def _low_memory_config(**overrides):
    config = {
        "task": "t2av",
        "dit_disk_streaming": True,
        "text_encoder_disk_streaming": True,
        "text_encoder_release_block_offload_buffers": True,
        "warmup": False,
        "cpu_offload": True,
    }
    config.update(overrides)
    return config


def _make_runner(runner_module, config):
    runner = object.__new__(runner_module.MiniMaxH3Runner)
    runner.config = config
    return runner


def test_low_memory_load_model_defers_vae(runner_module):
    runner = _make_runner(runner_module, _low_memory_config())
    calls = []
    runner.load_transformer = lambda: calls.append("transformer") or object()
    runner.load_text_encoder = lambda: calls.append("text_encoder") or [object()]
    runner.load_vae = lambda: calls.append("vae") or (object(), object())

    runner.load_model()

    assert calls == ["transformer", "text_encoder"]
    assert runner.video_vae is None
    assert runner.audio_vae is None


def test_non_low_memory_load_model_keeps_eager_vae(runner_module):
    runner = _make_runner(runner_module, _low_memory_config(text_encoder_disk_streaming=False))
    video_vae = object()
    audio_vae = object()
    calls = []
    runner.load_transformer = lambda: calls.append("transformer") or object()
    runner.load_text_encoder = lambda: calls.append("text_encoder") or [object()]
    runner.load_vae = lambda: calls.append("vae") or (video_vae, audio_vae)

    runner.load_model()

    assert calls == ["transformer", "text_encoder", "vae"]
    assert runner.video_vae is video_vae
    assert runner.audio_vae is audio_vae


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"text_encoder_release_block_offload_buffers": False}, "text_encoder_release_block_offload_buffers=true"),
        ({"warmup": True}, "warmup=false"),
    ],
)
def test_low_memory_load_model_rejects_unsupported_first_version_configs(runner_module, override, message):
    runner = _make_runner(runner_module, _low_memory_config(**override))
    runner.load_transformer = lambda: object()
    runner.load_text_encoder = lambda: [object()]
    runner.load_vae = lambda: (object(), object())

    with pytest.raises(ValueError, match=message):
        runner.load_model()


def test_offload_transformer_releases_disk_streaming_buffer(runner_module):
    runner = _make_runner(runner_module, _low_memory_config())
    calls = []
    runner.maybe_empty_cache = lambda **kwargs: calls.append(("empty", kwargs))
    runner.model = SimpleNamespace(
        block_offload=True,
        prepost_resident=False,
        pre_weight=SimpleNamespace(to_cpu=lambda: calls.append("pre_cpu")),
        post_weight=SimpleNamespace(to_cpu=lambda: calls.append("post_cpu")),
        transformer_weights=SimpleNamespace(release_disk_streaming_buffer=lambda: calls.append("release_dit")),
    )

    runner._offload_transformer()

    assert calls == ["pre_cpu", "post_cpu", "release_dit", ("empty", {"force": True, "collect_garbage": True})]


def test_offload_transformer_preserves_regular_block_offload_behavior(runner_module):
    runner = _make_runner(runner_module, _low_memory_config(dit_disk_streaming=False))
    calls = []
    runner.maybe_empty_cache = lambda **kwargs: calls.append(("empty", kwargs))
    runner.model = SimpleNamespace(
        block_offload=True,
        prepost_resident=False,
        pre_weight=SimpleNamespace(to_cpu=lambda: calls.append("pre_cpu")),
        post_weight=SimpleNamespace(to_cpu=lambda: calls.append("post_cpu")),
        transformer_weights=SimpleNamespace(release_disk_streaming_buffer=lambda: calls.append("release_dit")),
    )

    runner._offload_transformer()

    assert calls == ["pre_cpu", "post_cpu", ("empty", {"force": True, "collect_garbage": True})]


def test_run_vae_decoder_lazy_loads_once(runner_module):
    runner = _make_runner(runner_module, _low_memory_config())
    calls = []
    video_vae = SimpleNamespace(
        decode_parallel=False,
        decode=lambda latents: ("video", latents),
    )
    audio_vae = SimpleNamespace(decode=lambda latents: ("audio", latents))
    runner.load_vae = lambda: calls.append("load_vae") or (video_vae, audio_vae)
    runner.video_vae = None
    runner.audio_vae = None
    runner._vae_decode_tile_shapes = {}
    runner.scheduler = SimpleNamespace(
        num_condition_video_rows=0,
        num_condition_audio_rows=0,
        num_latent_frames=1,
        latent_height=1,
        latent_width=1,
        num_audio_latents=1,
    )

    first = runner.run_vae_decoder(torch.tensor([1]), torch.tensor([2]))
    second = runner.run_vae_decoder(torch.tensor([3]), torch.tensor([4]))

    assert calls == ["load_vae"]
    assert first[0][0] == "video"
    assert first[1][0] == "audio"
    assert torch.equal(first[0][1], torch.tensor([1]))
    assert torch.equal(first[1][1], torch.tensor([2]))
    assert second[0][0] == "video"
    assert second[1][0] == "audio"
    assert torch.equal(second[0][1], torch.tensor([3]))
    assert torch.equal(second[1][1], torch.tensor([4]))


def test_run_main_releases_vae_after_processing_result(runner_module):
    runner = _make_runner(runner_module, _low_memory_config())
    calls = []
    runner.maybe_empty_cache = lambda **kwargs: calls.append(("empty", kwargs))
    runner.init_run = lambda: calls.append("init")
    runner.run_segment = lambda _segment: calls.append("dit") or ("video_rows", "audio_rows")
    runner._offload_transformer = lambda: calls.append("offload_dit")
    runner.run_vae_decoder = lambda *_args: calls.append("decode") or ("decoded_video", "decoded_audio")

    def process():
        calls.append(("process", runner.video_vae, runner.audio_vae))
        return "result"

    runner.process_images_after_vae_decoder = process
    runner.end_run = lambda: calls.append("end_run")
    runner.video_vae = object()
    runner.audio_vae = object()

    assert runner.run_main() == "result"

    assert [call if isinstance(call, str) else call[0] for call in calls] == [
        "init",
        "dit",
        "offload_dit",
        "decode",
        "process",
        "empty",
        "end_run",
    ]
    process_call = calls[4]
    assert process_call[1] is not None
    assert process_call[2] is not None
    assert runner.video_vae is None
    assert runner.audio_vae is None
    assert runner.gen_video is None
    assert runner.gen_audio is None
