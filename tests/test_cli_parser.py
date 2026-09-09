import json
from pathlib import Path

import pytest

from lightx2v import infer
from lightx2v.disagg.examples import infer as disagg_infer
from lightx2v.models.runners.bagel.sensenova_vision_runner import SenseNovaVisionRunner
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.flux2.flux2_runner import Flux2Runner
from lightx2v.models.runners.hidream_o1_image.hidream_o1_image_runner import HidreamO1ImageRunner
from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner
from lightx2v.models.runners.wan.wan_audio_runner import Wan22AudioRunner
from lightx2v.models.runners.wan.wan_infinitetalk_runner import InfiniteTalkRunner
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.runners.worldmirror.worldmirror_runner import WorldMirrorRunner
from lightx2v.utils.set_config import build_cli_inputs


class CapturedInputs(Exception):
    def __init__(self, config, request):
        self.config = config
        self.request = request


@pytest.fixture
def cli_inputs(tmp_path, monkeypatch):
    def capture(args):
        raise CapturedInputs(*build_cli_inputs(args))

    def parse(config, options=(), model_cls="wan2.1", task="t2v", entrypoint=infer):
        monkeypatch.setattr(entrypoint, "build_cli_inputs", capture)
        model_path = tmp_path / "model"
        model_path.mkdir(exist_ok=True)
        if model_cls == "sensenova_vision":
            for filename in ("llm_config.json", "vit_config.json"):
                (model_path / filename).write_text("{}")
        config_path = tmp_path / "deployment.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr(
            "sys.argv",
            ["infer", "--model_cls", model_cls, "--task", task, "--model_path", str(model_path), "--config_json", str(config_path), *options],
        )
        with pytest.raises(CapturedInputs) as captured:
            entrypoint.main()
        return captured.value.config, captured.value.request

    return parse


@pytest.mark.parametrize("entrypoint", [infer, disagg_infer])
@pytest.mark.parametrize("options", [(), ("--return_result_tensor",)])
def test_cli_return_result_tensor(cli_inputs, entrypoint, options):
    config, request = cli_inputs({}, options, entrypoint=entrypoint)

    assert ("return_result_tensor" in request) == bool(options)
    assert "return_result_tensor" not in config
    runner = object.__new__(WanRunner)
    BaseRunner.__init__(runner, config)
    assert runner.prepare_request(request).return_result_tensor is bool(options)


def test_static_cli_omissions_resolve_request_content_and_output_specs(cli_inputs):
    defaults = {"target_video_length": 81}
    config, request = cli_inputs(defaults)

    assert request == {"task": "t2v"}
    for field, value in defaults.items():
        assert config[field] == value
    runner = object.__new__(WanRunner)
    BaseRunner.__init__(runner, config)
    input_info = runner.prepare_request({"task": runner.config["task"], **request})
    assert input_info.save_result_path is None
    assert input_info.seed == 42
    assert input_info.prompt == input_info.negative_prompt == ""
    assert input_info.target_video_length == 81


def test_static_cli_preserves_zero_empty_string_and_frame_alias(cli_inputs):
    defaults = {"target_video_length": 81}
    config, request = cli_inputs(defaults, ["--seed", "0", "--negative_prompt", "", "--save_result_path", "", "--num_frames", "49", "--target_shape", "480", "832"])

    assert request == {"task": "t2v", "seed": 0, "negative_prompt": "", "save_result_path": "", "target_video_length": 49, "target_shape": [480, 832]}
    for field, value in defaults.items():
        assert config[field] == value
    assert not {"seed", "prompt", "negative_prompt", "save_result_path"} & config.keys()


@pytest.mark.parametrize("task", ["i2v", "s2v"])
def test_wan22_audio_cli_preserves_audio_request(cli_inputs, task):
    preset = Path(__file__).resolve().parents[1] / "configs/seko_talk/seko_talk_08_5B_base.json"
    config, request = cli_inputs(
        json.loads(preset.read_text()),
        ["--image_path", "portrait.png", "--audio_path", "speech.wav", "--seed", "0"],
        model_cls="wan2.2_audio",
        task=task,
    )
    runner = object.__new__(Wan22AudioRunner)
    BaseRunner.__init__(runner, config)
    input_info = runner.prepare_request(request)

    assert input_info.task == task
    assert input_info.image_path == "portrait.png"
    assert input_info.audio_path == "speech.wav"
    assert input_info.seed == 0
    assert input_info.save_result_path is None
    assert config["vae_stride"] == [4, 16, 16]
    assert config["use_image_encoder"] is False
    assert runner.get_latent_shape_with_lat_hw(4, 4, 17) == [48, 5, 4, 4]


def test_sensenova_subtask_uses_json_until_explicitly_overridden(cli_inputs):
    defaults = {"omni_vision_subtask": "depth"}
    config, request = cli_inputs(defaults, model_cls="sensenova_vision", task="omni_vision_task")
    assert config["omni_vision_subtask"] == "depth"
    assert request == {"task": "omni_vision_task"}

    config, request = cli_inputs(defaults, ["--omni_vision_subtask", "normal"], model_cls="sensenova_vision", task="omni_vision_task")
    assert config["omni_vision_subtask"] == "depth"
    assert request == {"task": "omni_vision_task", "omni_vision_subtask": "normal"}


@pytest.mark.parametrize("defaults", [{}, {"use_compile": True}, {"use_compile": True, "warmup": True}, {"warmup": False}])
def test_warmup_is_only_read_from_system_config(cli_inputs, defaults):
    config, request = cli_inputs(defaults)

    assert config["warmup"] is defaults.get("warmup", False)
    assert request == {"task": "t2v"}


def test_cli_task_is_explicit_and_cannot_be_replaced_by_json(cli_inputs):
    config, request = cli_inputs({"task": "i2v"}, task="t2v")

    assert config["task"] == request["task"] == "t2v"


@pytest.mark.parametrize("option", ["--warmup", "--no-warmup"])
def test_static_cli_does_not_accept_warmup(cli_inputs, capsys, option):
    with pytest.raises(SystemExit) as exc:
        cli_inputs({}, [option])

    assert exc.value.code == 2
    assert f"unrecognized arguments: {option}" in capsys.readouterr().err


@pytest.mark.parametrize(
    "model_cls,runner_cls,task,defaults,options,expected",
    [
        (
            "flux2",
            Flux2Runner,
            "i2i",
            {"inpaint_mask_enabled": True},
            ["--inpaint_blur_sigma", "0.5", "--inpaint_blur_size", "3"],
            {"inpaint_blur_sigma": 0.5, "inpaint_blur_size": 3},
        ),
        (
            "hunyuan_image3",
            HunyuanImage3Runner,
            "t2t",
            {"enable_cfg": False, "bot_task": "auto", "moe_backend": "torch", "text_do_sample": True},
            [
                "--bot_task",
                "think_recaption",
                "--max_new_tokens",
                "128",
                "--system_prompt",
                "Be concise.",
                "--no-text_do_sample",
                "--text_temperature",
                "0.7",
                "--text_top_k",
                "0",
                "--text_top_p",
                "0.9",
            ],
            {"bot_task": "think_recaption", "max_new_tokens": 128, "system_prompt": "Be concise.", "text_do_sample": False, "text_temperature": 0.7, "text_top_k": 0, "text_top_p": 0.9},
        ),
        (
            "hunyuan_image3",
            HunyuanImage3Runner,
            "t2t",
            {"enable_cfg": False, "bot_task": "auto", "moe_backend": "torch", "text_do_sample": False},
            ["--text_do_sample"],
            {"text_do_sample": True},
        ),
        (
            "hunyuan_image3",
            HunyuanImage3Runner,
            "i2i",
            {"moe_backend": "torch", "infer_align_image_size": True},
            ["--no-infer_align_image_size"],
            {"infer_align_image_size": False},
        ),
        (
            "infinitetalk",
            InfiniteTalkRunner,
            "s2v",
            {},
            ["--video_duration", "2.5"],
            {"video_duration": 2.5},
        ),
        (
            "sensenova_vision",
            SenseNovaVisionRunner,
            "omni_vision_task",
            {"omni_vision_subtask": "depth", "postprocess_predictions": True},
            ["--raw_output_path", "raw.npz", "--glb_output_path", "scene.glb", "--no-postprocess_predictions"],
            {"raw_output_path": "raw.npz", "glb_output_path": "scene.glb", "postprocess_predictions": False},
        ),
    ],
)
def test_cli_model_specific_options_reach_runner(cli_inputs, model_cls, runner_cls, task, defaults, options, expected):
    config, request = cli_inputs(defaults, options, model_cls=model_cls, task=task)
    runner = object.__new__(runner_cls)
    BaseRunner.__init__(runner, config)
    input_info = runner.prepare_request(request)

    for field, value in expected.items():
        assert request[field] == value
        assert getattr(input_info, field) == value


@pytest.mark.parametrize(
    "model_cls,runner_cls,task,field",
    [
        ("hidream_o1_image", HidreamO1ImageRunner, "i2i", "keep_original_aspect"),
        ("worldmirror", WorldMirrorRunner, "recon", "save_rendered"),
        ("worldmirror", WorldMirrorRunner, "recon", "render_depth"),
    ],
)
def test_cli_boolean_can_disable_json_default(cli_inputs, model_cls, runner_cls, task, field):
    config, request = cli_inputs({field: True}, [f"--no-{field}"], model_cls=model_cls, task=task)
    runner = object.__new__(runner_cls)
    BaseRunner.__init__(runner, config)
    assert config[field] is True
    assert getattr(runner.prepare_request(request), field) is False
    assert getattr(runner.prepare_request({}), field) is True
