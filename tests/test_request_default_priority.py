import json
from unittest.mock import Mock

import pytest

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.cosmos3.cosmos3_runner import Cosmos3Runner
from lightx2v.models.runners.flux2.flux2_runner import Flux2Runner
from lightx2v.models.runners.longcat_image.longcat_image_runner import LongCatImageRunner
from lightx2v.models.runners.ltx2.ltx2_runner import LTX2Runner
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS, PROMPT_FIELDS, VIDEO_OUTPUT_FIELDS
from lightx2v.models.runners.wan.wan_animate2_runner import WanAnimate2Runner
from lightx2v.models.runners.z_image.z_image_runner import ZImageRunner
from lightx2v.pipeline import LightX2VPipeline
from lightx2v.utils.input_info import UNSET


class PreparedInputRunner(BaseRunner):
    supported_request_fields_by_task = {
        "t2v": COMMON_REQUEST_FIELDS | PROMPT_FIELDS | VIDEO_OUTPUT_FIELDS,
        "t2i": COMMON_REQUEST_FIELDS | PROMPT_FIELDS | {"aspect_ratio", "target_shape"},
        "sr": COMMON_REQUEST_FIELDS | {"image_path", "video_path", "target_shape"},
        "animate": COMMON_REQUEST_FIELDS | {"image_path", "video_path"},
    }

    def __init__(self, config):
        super().__init__(config)
        self.requests = []

    def create_input_info(self, request_data):
        self.requests.append(request_data)
        return super().create_input_info(request_data)

    def run_request(self, input_info):
        return input_info


def make_pipeline(config, model_cls="test_model"):
    pipeline = object.__new__(LightX2VPipeline)
    pipeline.task = config["task"]
    pipeline.model_cls = model_cls
    pipeline.runner = PreparedInputRunner(config)
    return pipeline


@pytest.mark.parametrize(("seed", "expected_seed"), [(UNSET, 42), (None, 42), (0, 0)])
def test_prepare_request_handles_omitted_values_without_changing_explicit_values(seed, expected_seed):
    runner = PreparedInputRunner({"task": "t2i", "target_shape": [512, 768], "enable_cfg": True})
    request_data = {
        "task": "t2i",
        "seed": seed,
        "prompt": None,
        "negative_prompt": "",
        "target_shape": None,
        "aspect_ratio": None,
        "save_result_path": None,
        "return_result_tensor": False,
    }

    input_info = runner.prepare_request(request_data)

    assert input_info.target_shape == [512, 768]
    assert input_info.seed == expected_seed
    assert input_info.prompt == ""
    assert input_info.negative_prompt == ""
    assert runner.requests[0]["negative_prompt"] == ""
    assert input_info.save_result_path is None
    assert input_info.return_result_tensor is False
    assert request_data["seed"] is seed
    assert request_data["target_shape"] is None


def test_shape_request_overrides_json_then_next_request_restores_json():
    runner = PreparedInputRunner({"task": "t2i", "target_shape": [512, 768], "target_height": 720, "target_width": 1280})

    first = runner.prepare_request({"task": runner.config["task"], "target_shape": [480, 832]})
    second = runner.prepare_request({"task": runner.config["task"]})

    assert first.target_shape == [480, 832]
    assert second.target_shape == [512, 768]
    assert runner.config["target_shape"] == [512, 768]


@pytest.mark.parametrize("task", ("i2va", "v2av"))
@pytest.mark.parametrize(
    "request_fields,expected_domain,expected_view",
    [({}, "av", "ego_view"), ({"domain_name": "droid_lerobot", "view_point": "wrist_view"}, "droid_lerobot", "wrist_view")],
)
def test_cosmos_action_file_overrides_config_below_explicit_request(tmp_path, task, request_fields, expected_domain, expected_view):
    config = {"task": task, "action_chunk_size": 16, "raw_action_dim": 29, "domain_name": "agibotworld", "view_point": "concat_view"}
    action_path = tmp_path / "actions.json"
    action_path.write_text(json.dumps({"action_chunk_size": 8, "raw_action_dim": 9, "domain_name": "av", "view_point": "ego_view"}))
    runner = object.__new__(Cosmos3Runner)
    BaseRunner.__init__(runner, config.copy())
    runner.input_info = runner.prepare_request({"action_path": str(action_path), **request_fields})

    runner._prepare_action_context()

    assert runner.input_info.action_chunk_size == 8
    assert runner.input_info.target_video_length == 9
    assert runner._get_action_value("raw_action_dim") == 9
    assert runner._get_action_value("domain_name") == expected_domain
    assert runner._get_action_value("view_point") == expected_view
    assert runner.config == config


@pytest.mark.parametrize("task", ("i2va", "v2av"))
def test_cosmos_action_metadata_is_resolved_again_for_each_request(tmp_path, task):
    config = {"task": task, "action_chunk_size": 16, "raw_action_dim": 29, "domain_name": "agibotworld", "view_point": "concat_view"}
    action_path = tmp_path / "actions.json"
    runner = object.__new__(Cosmos3Runner)
    BaseRunner.__init__(runner, config.copy())

    for spec in (
        {"action_chunk_size": 8, "raw_action_dim": 9, "domain_name": "av", "view_point": "ego_view"},
        {"action_chunk_size": 4, "raw_action_dim": 2, "domain_name": "pusht", "view_point": "third_person_view"},
        {},
    ):
        action_path.write_text(json.dumps(spec))
        runner.input_info = runner.prepare_request({"action_path": str(action_path)})

        runner._prepare_action_context()

        expected = spec or config
        assert runner.input_info.target_video_length == expected["action_chunk_size"] + 1
        for name in ("action_chunk_size", "raw_action_dim", "domain_name", "view_point"):
            assert runner._get_action_value(name) == expected[name]
        assert runner.config == config


def test_explicit_aspect_ratio_can_replace_json_shape():
    runner = PreparedInputRunner({"task": "t2i", "target_shape": [512, 768], "target_height": 720, "target_width": 1280})

    assert runner.prepare_request({"task": runner.config["task"], "aspect_ratio": "1:1"}).target_shape == []
    assert runner.prepare_request({"task": runner.config["task"]}).target_shape == [512, 768]


@pytest.mark.parametrize(
    ("config_shape", "request_shape", "expected_shape", "should_probe"),
    [
        (None, None, [512, 768], True),
        ([640, 960], None, [640, 960], False),
        ([640, 960], [480, 832], [480, 832], False),
        (None, [480, 832], [480, 832], False),
    ],
)
def test_ltx2_source_resolution_only_fills_missing_shape(config_shape, request_shape, expected_shape, should_probe):
    runner = object.__new__(LTX2Runner)
    config = {"task": "v2av", "target_height": 768, "target_width": 1280}
    request = {"video_path": "control.mp4"}
    if config_shape is not None:
        config["target_shape"] = config_shape
    if request_shape is not None:
        request["target_shape"] = request_shape
    BaseRunner.__init__(runner, config)
    probes = []

    def probe(path):
        probes.append(path)
        return 512, 768

    runner._probe_video_hw = probe
    runner._get_ref_downscale_factor = lambda: 1.0
    runner.input_info = runner.prepare_request({"task": runner.config["task"], **request})
    runner._override_target_hw_from_ref_video()

    assert runner.input_info.target_shape == expected_shape
    assert probes == (["control.mp4"] if should_probe else [])
    assert runner.config == config


def test_pipeline_uses_request_content_and_code_defaults():
    pipeline = make_pipeline({"task": "t2v"})

    default = pipeline.generate()
    explicit = pipeline.generate(seed=0, prompt="", save_result_path="explicit.mp4")
    restored = pipeline.generate()

    assert (default.seed, default.prompt, default.save_result_path) == (42, "", None)
    assert (explicit.seed, explicit.prompt, explicit.save_result_path) == (0, "", "explicit.mp4")
    assert (restored.seed, restored.prompt, restored.save_result_path) == (42, "", None)
    assert pipeline.runner.requests[0] == {"task": "t2v"}
    assert pipeline.runner.requests[2] == {"task": "t2v"}


@pytest.mark.parametrize("task", ["t2i", "t2v"])
def test_pipeline_preserves_explicit_none_output_path(task):
    pipeline = make_pipeline({"task": task})

    assert pipeline.generate(save_result_path=None).save_result_path is None
    assert pipeline.generate(save_result_path="explicit.png").save_result_path == "explicit.png"
    assert pipeline.generate().save_result_path is None


@pytest.mark.parametrize("runner_cls", [Flux2Runner, LongCatImageRunner, QwenImageRunner, ZImageRunner])
@pytest.mark.parametrize("output_path", [None, "result.png"])
def test_image_runners_save_only_with_an_output_path(runner_cls, output_path):
    runner = object.__new__(runner_cls)
    BaseRunner.__init__(runner, {"task": "t2i", "model_variant": "klein"})
    runner._gc_frozen = True
    request_data = {} if output_path is None else {"save_result_path": output_path}
    input_info = runner.prepare_request({"task": runner.config["task"], **request_data})
    image = Mock()

    if runner_cls is QwenImageRunner:
        runner._save_images([image], input_info)
    else:
        runner.run_input_encoder = lambda: {}
        runner.set_latent_shape = lambda: None
        runner.run_dit = lambda: (None, None)
        runner.run_vae_decoder = lambda latents: [image]
        runner.end_run = lambda: None
        runner.run_pipeline(input_info)

    if output_path is None:
        image.save.assert_not_called()
    else:
        image.save.assert_called_once_with(output_path)


def test_pipeline_preserves_explicit_default_named_output_for_video():
    pipeline = make_pipeline({"task": "t2v"})

    result = pipeline.generate(save_result_path="lightx2v_gen_result.png")

    assert result.save_result_path == "lightx2v_gen_result.png"


def test_pipeline_sr_does_not_infer_output_paths_from_media():
    pipeline = make_pipeline({"task": "sr"})

    video = pipeline.generate(video_path="input.mp4")
    image = pipeline.generate(video_path="", image_path="input.png")
    restored = pipeline.generate(video_path="input.mp4")

    assert video.save_result_path is None
    assert image.save_result_path is None
    assert restored.save_result_path is None


def test_pipeline_animate2_uses_request_seed_and_code_default():
    pipeline = make_pipeline({"task": "animate", "seed": 71}, model_cls="wan2.2_animate2_distilled")

    assert pipeline.generate().seed == 42
    assert pipeline.generate(seed=None).seed == 42
    assert pipeline.generate(seed=0).seed == 0


@pytest.mark.parametrize(("seed", "expected"), [(None, 42), (123, 123), (0, 0)])
def test_animate2_receives_resolved_request_seed(seed, expected):
    runner = object.__new__(WanAnimate2Runner)
    BaseRunner.__init__(runner, {"task": "animate", "seed": 71})
    assert runner.prepare_request({"task": "animate", "seed": seed, "image_path": "reference.png", "video_path": "driver.mp4"}).seed == expected
