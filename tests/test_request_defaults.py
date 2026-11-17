"""Configured defaults retain startup semantics across model loaders and requests."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import TypeAdapter

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.registry import RUNNER_MODULES, get_model_spec
from lightx2v.server.api.request_parsing import _form_fields
from lightx2v.server.schema import build_task_request
from lightx2v.utils.cli import build_parser, parse_cli_inputs
from lightx2v.utils.request_defaults import get_request_default_arguments, validate_request_defaults
from lightx2v.utils.set_config import build_startup_config, merge_config

ROOT = Path(__file__).resolve().parents[1]
MODEL_TASKS = [(model, task) for model in RUNNER_MODULES for task in get_model_spec(model).tasks]


@pytest.fixture
def model_path(tmp_path):
    metadata = {
        "seed": 11,
        "size": [1024, 1024],
        "num_frames": 81,
        "vae_scale_factors": [8, 32, 32],
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "in_channels": 16,
        "out_channels": 16,
    }
    for subfolder in ("", "transformer", "transformer_ref", "dit-14b", "dit-14b-bidir", "dit-5b", "dit-5b-bidir"):
        folder = tmp_path / subfolder
        folder.mkdir(exist_ok=True)
        (folder / "config.json").write_text(json.dumps(metadata))
    vae = {"block_out_channels": [64, 128, 256, 512], "temperal_downsample": [True, True], "dim_mult": [1, 2, 4, 4]}
    for subfolder in ("vae", "vae-14b", "vae-5b"):
        folder = tmp_path / subfolder
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(vae))
    release = {"scenes": {"castle": {}}, "architecture": {"dit_config": {"out_dim": 16}}}
    (tmp_path / "release-config.json").write_text(json.dumps(release))
    return tmp_path


@pytest.mark.parametrize("model_cls", RUNNER_MODULES)
def test_every_registered_model_loader_resolves_defaults_without_loading_weights(model_path, model_cls):
    task = next(iter(get_model_spec(model_cls).tasks))
    config = {
        "model_cls": model_cls,
        "task": task,
        "model_path": str(model_path),
        "moe_backend": "flashinfer",
        "transformer_model_name": "test",
        "request_defaults": {"seed": 7},
    }
    if model_cls in {"minimax_h3", "taomate_h3"}:
        config["model_variant"] = "fl2av"
    elif model_cls == "cosmos3":
        config["model_variant"] = "omni"
    if model_cls == "minimax_h3_causal":
        config["dit_original_ckpt"] = str(model_path / "transformer_ref")
    if model_cls == "ltx2_5":
        for key in ("dit_original_ckpt", "text_encoder_original_ckpt", "video_vae_original_ckpt", "audio_vae_original_ckpt", "upsampler_original_ckpt", "duration_head_original_ckpt"):
            config[key] = str(model_path)
    source = deepcopy(config)
    resolved = build_startup_config(config, configure_runtime=False)
    assert resolved["request_defaults"]["seed"] == 7
    assert resolved["request_defaults"].keys().isdisjoint(resolved)
    assert config == source
    resolved.lock()
    with pytest.raises(TypeError):
        resolved["request_defaults"]["seed"] = 99


@pytest.mark.parametrize("model_cls,task", MODEL_TASKS)
@pytest.mark.parametrize("dynamic", [{}, {"replace_flag": True, "dancer_stage": "local", "inpaint_mask_enabled": True}])
def test_default_whitelist_is_exposed_by_cli_and_http(model_cls, task, dynamic):
    config = {"model_cls": model_cls, "task": task, "model_variant": "omni"} | dynamic
    spec = get_model_spec(model_cls)
    cli_fields, http_fields = set(), set()
    for supported_task in spec.get_tasks(config):
        cli_fields.update(action.dest for action in build_parser(config, supported_task)._actions)
        http_fields.update(_form_fields(build_task_request(config), supported_task))
    defaults = get_request_default_arguments(config)
    assert defaults.keys() <= cli_fields & http_fields
    assert {"seed", "save_result_path"} <= defaults.keys()
    assert {"task", "return_result_tensor", "policy_image", "policy_state", "reuse", "lora_name"}.isdisjoint(defaults)


@pytest.mark.parametrize("entry", ["cli", "http", "python"])
def test_explicit_overrides_do_not_leak_and_null_preserves_defaults(model_path, entry):
    source = {"request_defaults": {"size": [480, 832], "num_frames": 81, "seed": 7}}
    profile = model_path / "profile.json"
    profile.write_text(json.dumps(source))
    startup = {"model_cls": "wan2.1", "task": "i2v", "model_path": str(model_path), "config_json": str(profile)}
    runner = BaseRunner(build_startup_config(startup, configure_runtime=False))
    original = dict(runner.config)

    def prepare(overrides):
        if entry == "cli":
            argv = ["--model_cls", "wan2.1", "--task", "i2v", "--model_path", str(model_path), "--config_json", str(profile)]
            for name, value in overrides.items():
                if value is not None:
                    argv.extend([f"--{name}", *map(str, value if isinstance(value, list) else [value])])
            _, _, request = parse_cli_inputs(argv)
            assert set(request) == {"task"} | {key for key, value in overrides.items() if value is not None}
        elif entry == "http":
            request = TypeAdapter(build_task_request(runner.config)).validate_python(overrides).model_dump(exclude_unset=True)
            assert request.keys() == overrides.keys()
        else:
            request = {"task": "i2v"} | overrides
        return runner.prepare_request(request)

    first = prepare({"size": [768, 1344], "num_frames": 121, "seed": 0})
    assert (first.size, first.num_frames, first.seed) == ([768, 1344], 121, 0)
    first.size[0] = 16
    for overrides in ({}, {"size": None, "num_frames": None, "seed": None}):
        info = prepare(overrides)
        assert (info.size, info.num_frames, info.seed) == ([480, 832], 81, 7)
    assert runner.config == original
    assert json.loads(profile.read_text()) == source


def test_checkpoint_metadata_defaults_and_frame_normalization(model_path):
    config = build_startup_config(
        {"model_cls": "wan2.1", "task": "i2v", "model_path": str(model_path), "request_defaults": {"num_frames": 82, "size": [480, 832]}},
        configure_runtime=False,
    )
    assert config["request_defaults"]["num_frames"] == 81
    assert config["request_defaults"]["size"] == [480, 832]
    assert "num_frames" not in config and "size" not in config


def test_nested_configured_inputs_are_isolated_between_requests():
    config = merge_config(
        {},
        {
            "model_cls": "hidream_o1_image",
            "task": "i2i",
            "request_defaults": {"layout_bboxes": {"regions": [{"box": [0, 0, 1, 1]}]}},
        },
    )
    config.lock()
    runner = BaseRunner(config)
    first = runner.prepare_request({"task": "i2i"})
    first.layout_bboxes["regions"][0]["box"][0] = 0.5
    first.layout_bboxes["extra"] = True
    second = runner.prepare_request({"task": "i2i"})
    assert second.layout_bboxes == {"regions": [{"box": [0, 0, 1, 1]}]}
    assert config["request_defaults"]["layout_bboxes"] == second.layout_bboxes


def test_seedvr_validates_configured_size_in_the_nested_namespace():
    from lightx2v.models.runners.seedvr.seedvr_runner import SeedVRRunner

    config = merge_config({}, {"model_cls": "seedvr2", "task": "sr", "request_defaults": {"size": [0, 832]}})
    runner = SeedVRRunner.__new__(SeedVRRunner)
    BaseRunner.__init__(runner, config)
    with pytest.raises(ValueError, match="two positive integers"):
        runner.prepare_request({})
    assert runner.prepare_request({"size": [480, 832]}).size == [480, 832]
    assert config["request_defaults"]["size"] == [0, 832]


def test_z_image_scheduler_reads_nested_strength_when_input_has_no_value():
    from lightx2v.models.schedulers.z_image.scheduler import ZImageScheduler

    scheduler = ZImageScheduler.__new__(ZImageScheduler)
    scheduler.config = merge_config({}, {"model_cls": "z_image", "task": "i2i", "request_defaults": {"i2i_denoise_strength": 0.6}})
    assert scheduler._get_i2i_denoise_strength(SimpleNamespace(i2i_denoise_strength=None)) == 0.6
    assert scheduler._get_i2i_denoise_strength(SimpleNamespace(i2i_denoise_strength=0.2)) == 0.2


@pytest.mark.parametrize("defaults", [None, [], "invalid"])
def test_namespace_must_be_an_object(defaults):
    with pytest.raises(ValueError, match="JSON object"):
        validate_request_defaults({"model_cls": "wan2.1", "task": "t2v", "request_defaults": defaults})


@pytest.mark.parametrize("field", ["infer_steps", "fps", "task", "return_result_tensor", "reuse", "lora_name", "unknown"])
def test_non_shared_request_fields_are_rejected(field):
    with pytest.raises(ValueError, match="request_defaults fields"):
        validate_request_defaults({"model_cls": "wan2.1", "request_defaults": {field: 1}})


def test_duplicate_default_sources_are_rejected():
    with pytest.raises(ValueError, match="top level: size"):
        validate_request_defaults({"model_cls": "wan2.1", "task": "t2v", "size": [480, 832], "request_defaults": {"size": [768, 1344]}})


def test_request_defaults_are_not_accepted_as_top_level_startup_fields():
    with pytest.raises(ValueError, match="under request_defaults: size"):
        merge_config({}, {"model_cls": "wan2.1", "task": "t2v", "size": [480, 832]})


def test_fixed_context_and_request_defaults_have_separate_sources():
    metadata = {"size": [480, 832], "num_frames": 81, "seed": 11}
    before = deepcopy(metadata)
    config = merge_config(metadata, {"model_cls": "wan2.1_sf", "task": "t2v", "request_defaults": {"seed": 7}})
    assert config["num_frames"] == 81
    assert config["request_defaults"] == {"size": [480, 832], "seed": 7}
    assert "size" not in config and "seed" not in config
    assert metadata == before
    runner = BaseRunner(config)
    info = runner.prepare_request({})
    assert (info.size, info.num_frames, info.seed) == ([480, 832], 81, 7)
    with pytest.raises(ValueError, match="does not support request fields: num_frames"):
        runner.prepare_request({"num_frames": 121})


def test_model_geometry_and_warmup_read_nested_defaults_without_mutating_config():
    from lightx2v.models.runners.default_runner import DefaultRunner
    from lightx2v.models.runners.lifecycle import RunnerLifecycleMixin

    config = merge_config({}, {"model_cls": "wan2.1", "task": "t2v", "request_defaults": {"size": [480, 832], "num_frames": 81}})
    config.lock()
    runner = DefaultRunner.__new__(DefaultRunner)
    BaseRunner.__init__(runner, config)
    runner._WARMUP_SHAPES = ((480, 480), (480, 832))
    assert runner.get_target_size() == (480, 832)
    assert runner.get_num_frames() == 81
    assert RunnerLifecycleMixin.get_warmup_resolutions(runner) == ((480, 480), (480, 832))
    runner.input_info = runner.prepare_request({"size": [768, 1344], "num_frames": 121})
    assert runner.get_target_size() == (768, 1344)
    assert runner.get_num_frames() == 121
    assert config["request_defaults"] == {"size": [480, 832], "num_frames": 81}
    assert "size" not in config and "num_frames" not in config


def test_invalid_task_reports_a_configuration_error_before_loading_weights():
    with pytest.raises(ValueError, match="does not support task"):
        validate_request_defaults({"model_cls": "wan2.1", "task": "invalid", "request_defaults": {"size": [480, 832]}})


def test_defaults_namespace_is_not_a_post_request_field():
    with pytest.raises(ValueError, match="Extra inputs"):
        TypeAdapter(build_task_request({"model_cls": "wan2.1", "task": "t2v"})).validate_python({"request_defaults": {"size": [480, 832]}})


@pytest.mark.parametrize(
    "model,task,options,defaults",
    [
        ("wan2.1", "i2v", {"self_attn_1_type": "nbhd_attn"}, {"num_frames": 81}),
        ("wan2.1_sf", "t2v", {}, {"num_frames": 81}),
        ("wan2.2_s2v", "s2v", {}, {"size": [480, 832]}),
        ("qwen_image", "i2i", {}, {"resolution": 1024}),
        ("qwen_image", "i2i", {"layered": True}, {"i2i_denoise_strength": 0.5}),
        ("flux2", "i2i", {}, {"inpaint_blur_size": 11}),
        ("wan_dancer", "s2v", {"dancer_stage": "global"}, {"video_path": "input.mp4"}),
    ],
)
def test_fixed_or_disabled_fields_stay_startup_only(model, task, options, defaults):
    with pytest.raises(ValueError, match="request_defaults fields"):
        validate_request_defaults({"model_cls": model, "task": task, "request_defaults": defaults} | options)


def test_defaults_stay_nested_and_preserve_null_false_and_empty_values():
    raw = {"model_cls": "seedvr2", "task": "sr", "request_defaults": {"size": [], "match_target_size": False, "seed": None}}
    before = deepcopy(raw)
    defaults = validate_request_defaults(raw)
    assert defaults == raw["request_defaults"]
    config = merge_config({}, raw)
    assert config["request_defaults"]["size"] == []
    assert config["request_defaults"]["match_target_size"] is False
    assert config["request_defaults"]["seed"] is None
    assert {"size", "match_target_size", "seed"}.isdisjoint(config)
    assert raw == before
    assert BaseRunner(config).prepare_request({}).seed == 42


def test_ltx25_configured_frames_remain_a_fallback_and_do_not_disable_duration_head():
    from lightx2v.models.runners.ltx2.ltx25_runner import LTX25Runner

    config = merge_config({}, {"model_cls": "ltx2_5", "fps": 24, "request_defaults": {"num_frames": 81}})
    runner = LTX25Runner.__new__(LTX25Runner)
    BaseRunner.__init__(runner, config)
    predictor = Mock(return_value=41)
    runner.duration_predictor = SimpleNamespace(predict=predictor)
    runner.input_info = runner.prepare_request({"task": "t2av"})
    assert runner.input_info.num_frames is None
    assert runner._resolve_num_frames({"v_context_p": None, "a_context_p": None}) == 41
    runner.input_info = runner.prepare_request({"task": "t2av", "num_frames": 17})
    assert runner._resolve_num_frames({}) == 17
    assert predictor.call_count == 1
    config["auto_duration"] = False
    runner.input_info = runner.prepare_request({"task": "t2av"})
    assert runner._resolve_num_frames({}) == 81


def test_ltx_control_video_keeps_default_size_separate_from_an_explicit_size():
    from lightx2v.models.runners.ltx2.ltx2_runner import LTX2Runner

    config = merge_config({}, {"model_cls": "ltx2", "task": "v2av", "vae_scale_factors": [8, 32, 32], "request_defaults": {"size": [512, 768], "num_frames": 81}})
    runner = LTX2Runner.__new__(LTX2Runner)
    BaseRunner.__init__(runner, config)
    assert runner.prepare_request({}).size == []
    assert runner.prepare_request({"size": [768, 1344]}).size == [768, 1344]
    assert config["request_defaults"]["size"] == [512, 768]
    assert "size" not in config


def test_cosmos_action_file_still_precedes_configured_metadata(tmp_path):
    from lightx2v.models.runners.cosmos3.cosmos3_runner import Cosmos3Runner

    path = tmp_path / "actions.json"
    path.write_text(json.dumps({"domain_name": "libero"}))
    config = merge_config({}, {"model_cls": "cosmos3", "model_variant": "omni", "action_gen": True, "request_defaults": {"domain_name": "droid_lerobot", "view_point": "concat_view"}})
    runner = Cosmos3Runner.__new__(Cosmos3Runner)
    BaseRunner.__init__(runner, config)
    runner.input_info = runner.prepare_request({"task": "i2va", "action_path": str(path)})
    assert runner._get_action_value("domain_name") == "libero"
    assert runner._get_action_value("view_point") == "concat_view"
    runner.input_info = runner.prepare_request({"task": "i2va", "action_path": str(path), "domain_name": "camera_pose"})
    assert runner._get_action_value("domain_name") == "camera_pose"


def test_hunyuan_image_size_default_and_automatic_reference_size():
    from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner

    config = merge_config({}, {"model_cls": "hunyuan_image3", "request_defaults": {"size": [1024, 1024], "align_image_size": True}})
    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    BaseRunner.__init__(runner, config)
    reference = [[SimpleNamespace(i=SimpleNamespace(image_height=640, image_width=896))]]
    for task in ("i2i", "ti2i"):
        assert runner._resolve_image_size(runner.prepare_request({"task": task}), reference) == (1024, 1024)
        assert runner._resolve_image_size(runner.prepare_request({"task": task, "size": []}), reference) == (640, 896)
        assert runner._resolve_image_size(runner.prepare_request({"task": task, "size": [768, 1344]}), reference) == (768, 1344)


@pytest.mark.parametrize("default_seed", [None, 0, 7])
def test_neopp_keeps_session_rng_unless_a_seed_is_configured_or_submitted(default_seed):
    from lightx2v.models.runners.neopp.neopp_runner import NeoppRunner

    config = merge_config({}, {"model_cls": "neopp", "request_defaults": {"seed": default_seed}})
    runner = NeoppRunner.__new__(NeoppRunner)
    BaseRunner.__init__(runner, config)
    assert runner.prepare_request({}).seed == default_seed
    assert runner.prepare_request({"seed": 13}).seed == 13
    assert runner.prepare_request({"seed": None}).seed == default_seed


def test_disagg_workload_preserves_nested_defaults_during_stage_overrides():
    from lightx2v.disagg.workload import StageSpec, build_payload

    base = {"model_cls": "wan2.2_moe", "request_defaults": {"size": [480, 832], "seed": 7}}
    stage = StageSpec("test", 1, 1, 1, config_variants=[{"request_defaults": {"size": [768, 1344]}}])
    payload = build_payload(base, stage, 0)
    assert payload["request_defaults"]["size"] == [768, 1344]
    assert payload["request_defaults"]["seed"] == 7
    assert "size" not in payload and "seed" not in payload
    assert base["request_defaults"]["size"] == [480, 832]


def test_disagg_snapshots_preserve_nested_startup_defaults():
    from lightx2v.disagg.disagg_mixin import DisaggMixin

    runtime = DisaggMixin()
    runtime.config = merge_config({}, {"model_cls": "wan2.1", "task": "t2v", "request_defaults": {"size": [480, 832], "num_frames": 81}})
    runtime._disagg_bootstrap_room = 0
    effective = runtime._disagg_effective_config({"request_defaults": {"num_frames": 121}})
    assert effective["request_defaults"] == {"size": [480, 832], "num_frames": 121}
    assert "size" not in effective and "num_frames" not in effective
    snapshot = runtime._disagg_build_request_config_snapshot({"num_frames": 161})
    assert snapshot["num_frames"] == 161
    assert snapshot["size"] == [480, 832]
    assert runtime.config["request_defaults"]["num_frames"] == 81
    assert "size" not in runtime.config and "num_frames" not in runtime.config
    runtime._disagg_request_config = {"request_defaults": {"seed": 7}}
    assert runtime.resolve_request_seed({}) == 7


def test_all_repository_configs_are_well_formed_and_have_one_default_source():
    paths = list((ROOT / "configs").rglob("*.json"))
    migrated = 0
    for path in paths:
        source = json.loads(path.read_text())
        if not isinstance(source, dict) or "request_defaults" not in source:
            continue
        migrated += 1
        before = deepcopy(source)
        assert isinstance(source["request_defaults"], dict), path
        assert source["request_defaults"].keys().isdisjoint(source), path
        if source.get("model_cls"):
            validate_request_defaults(source)
        assert source == before, path
    assert migrated
    nullable = json.loads((ROOT / "configs/edenlive/wan21_14b_few_step.json").read_text())
    assert nullable["request_defaults"]["num_frames"] is None
    fixed = json.loads((ROOT / "configs/attentions/wan_i2v_svg.json").read_text())
    assert "num_frames" in fixed and "num_frames" not in fixed["request_defaults"]
