import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from PIL import Image

from lightx2v.models.networks.bagel.sensenova_tasks import (
    CAMERA_POSE_PROMPT,
    MODE_PROFILES,
    get_omni_vision_task_spec,
    normalize_omni_vision_subtask,
    resolve_mode,
    resolve_prompt,
)
from lightx2v.models.networks.bagel.sensenova_transforms import build_sensenova_transforms
from lightx2v.models.networks.bagel.sensenova_vision_model import SenseNovaVisionModel
from lightx2v.models.runners.bagel.sensenova_postprocess import resolve_pose_string
from lightx2v.models.runners.bagel.sensenova_vision_runner import SenseNovaVisionRunner
from lightx2v.models.video_encoders.hf.bagel.sensenova_vae import SenseNovaVisionVae
from lightx2v.utils.input_info import SenseNovaVisionInputInfo, init_empty_input_info
from lightx2v.utils.set_config import set_config, validate_model_task_args

LIGHTX2V_ROOT = Path(__file__).resolve().parents[1]
SENSENOVA_SOURCE_ROOT = LIGHTX2V_ROOT.parent / "sensenova-vision-v2"
SENSENOVA_MODEL_ROOT = LIGHTX2V_ROOT.parent / "SenseNova-Vision"
MODEL_PATH = SENSENOVA_MODEL_ROOT / "models/SenseNova-Vision-7B-MoT"


def test_sensenova_mode_profiles_match_official_defaults():
    dense = MODE_PROFILES["dense_perception"]
    assert dense == {
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.0, 1.0],
        "timestep_shift": 4.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "text_channel",
    }
    assert MODE_PROFILES["caption_generate"]["max_think_token_n"] == 8192
    assert MODE_PROFILES["dense_OCR"]["max_think_token_n"] == 20000
    assert resolve_mode("camera_pose") == "understanding"
    assert resolve_mode("raw_query", "think_edit") == "think_edit"

    subtask, spec = get_omni_vision_task_spec("binary_seg")
    assert subtask == "binary_segmentation"
    assert spec.mode == "dense_perception"
    assert normalize_omni_vision_subtask("object-detection") == "object_detection"


def test_sensenova_task_prompts_and_input_info_are_registered():
    assert "relative depth" in resolve_prompt("depth", "").lower()
    assert "<quat>" in CAMERA_POSE_PROMPT
    assert resolve_prompt("binary_seg", "cat") == ("Can you segment the image based on the following categories: <p>cat</p>? Please output the binary segmentation masks.")
    info = init_empty_input_info("omni_vision_task")
    info.omni_vision_subtask = "recon3d"
    assert isinstance(info, SenseNovaVisionInputInfo)
    assert info.postprocess_predictions is None


def test_sensenova_model_task_validation_lives_in_set_config():
    validate_model_task_args(Namespace(task="omni_vision_task", model_cls="sensenova_vision"))
    validate_model_task_args(Namespace(task="omni_vision_task", model_cls="sensenova_vision", omni_vision_subtask="depth"))

    invalid_args = (
        Namespace(task="omni_vision_task", model_cls="bagel", omni_vision_subtask="depth"),
        Namespace(task="omni_vision_task", model_cls="sensenova_vision", omni_vision_subtask=None),
        Namespace(task="t2i", model_cls="sensenova_vision", omni_vision_subtask=None),
        Namespace(task="t2i", model_cls="bagel", omni_vision_subtask="depth"),
    )
    for args in invalid_args:
        with pytest.raises(ValueError):
            validate_model_task_args(args)

    infer_source = (LIGHTX2V_ROOT / "lightx2v/infer.py").read_text()
    assert "--task omni_vision_task requires --model_cls sensenova_vision" not in infer_source


def test_sensenova_interleaves_images_at_official_placeholders():
    images = [Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))]
    items = SenseNovaVisionRunner._build_interleaved_inputs(
        "prefix<image>middle<image>suffix",
        images,
    )
    assert items[0] == "prefix"
    assert items[1] is images[0]
    assert items[2] == "middle"
    assert items[3] is images[1]
    assert items[4] == "suffix"


def test_sensenova_transforms_have_official_shapes_and_normalization():
    image = Image.new("RGB", (1234, 777), (255, 128, 0))
    transforms = build_sensenova_transforms()
    vae_image = transforms["vae"].resize_transform(image)
    vit_image = transforms["vit"].resize_transform(image)
    assert vae_image.size == (1024, 640)
    assert vit_image.size == (980, 616)
    tensor = transforms["vae"](image)
    assert tensor.shape == (3, 640, 1024)
    assert tensor[0, 0, 0].item() == pytest.approx(1.0)
    assert tensor[2, 0, 0].item() == pytest.approx(-1.0)


def test_sensenova_multi_output_batch_collapse_preserves_total_tokens():
    generation_input = {
        "packed_seqlens": torch.tensor([11, 13, 17], dtype=torch.int32),
        "key_values_lens": torch.tensor([101, 0, 0], dtype=torch.int32),
    }
    SenseNovaVisionModel._collapse_generation_batch(generation_input)
    assert generation_input["packed_seqlens"].tolist() == [41]
    assert generation_input["key_values_lens"].tolist() == [101]


def test_sensenova_raw_vae_output_detaches_before_numpy_conversion():
    class FakeAutoencoder:
        @staticmethod
        def decode(latent):
            return latent[:, :3]

    vae = SenseNovaVisionVae.__new__(SenseNovaVisionVae)
    vae.device = torch.device("cpu")
    vae.vae_model = FakeAutoencoder()
    latent = torch.randn(4, 16, requires_grad=True)
    output = vae.decode(
        latent,
        {
            "packed_seqlens": torch.tensor([6]),
            "image_shapes": [(8, 8)],
            "latent_downsample": 8,
            "latent_patch_size": 2,
            "latent_channel": 16,
            "output_raw_tensor": True,
        },
    )
    assert output[0].shape == (2, 2, 3)


def test_sensenova_pose_parser_matches_coordinate_token_scaling():
    text = "<frame><quat>[0,0,0,1000]</quat><offset>[1000,0,-500]</offset><scale>200</scale></frame>"
    pose = resolve_pose_string(text)
    assert pose["rotation"] == [[0.0, 0.0, 0.0, 1.0]]
    assert pose["translation"] == [[2.0, 0.0, -1.0]]


@pytest.mark.skipif(not MODEL_PATH.is_dir(), reason="local SenseNova-Vision checkpoint not available")
def test_sensenova_config_assembles_bagel_structure_instead_of_root_metadata(monkeypatch):
    monkeypatch.setenv("SENSENOVA_SOURCE_PATH", str(SENSENOVA_SOURCE_ROOT))
    config = set_config(
        Namespace(
            model_cls="sensenova_vision",
            task="omni_vision_task",
            model_path=str(MODEL_PATH),
            config_json=str(LIGHTX2V_ROOT / "configs/sensenova_vision/sensenova_vision.json"),
            seed=42,
        )
    )
    root_metadata = json.loads((MODEL_PATH / "config.json").read_text())
    assert root_metadata["base_model"] == "BAGEL-7B-MoT"
    assert "base_model" not in config
    assert config["llm_config"]["num_hidden_layers"] == 28
    assert config["vit_config"]["num_hidden_layers"] == 27
    assert config["vae_config"] == {"z_channels": 16, "downsample": 8}
    assert config["enable_vision_context"] is True
    assert config["llm_config_update"]["qk_norm"] is True

    assert config["sensenova_source_path"] == str(SENSENOVA_SOURCE_ROOT)


def test_sensenova_launch_script_exposes_official_example_command():
    script = (LIGHTX2V_ROOT / "scripts/sensenova_vision/run_sensenova_vision.sh").read_text()
    assert 'if [[ "${TASK}" == "example" ]]' in script
    assert "examples/sensenova_vision/example_visualize.py" in script
    assert "--model_cls sensenova_vision" in script
    assert '--example "${EXAMPLE_ID:-all}"' in script

    assert "--task omni_vision_task" in script
    assert "--omni_vision_subtask" in script
    infer_source = (LIGHTX2V_ROOT / "lightx2v/infer.py").read_text()
    assert '"omni_vision_task"' in infer_source
    assert '"raw_query"' not in infer_source
    for option in ("--sensenova_mode", "--raw_output_path", "--glb_output_path", "--postprocess_predictions", "--sensenova_source_path"):
        assert option not in infer_source


def test_sensenova_server_launch_script_defaults_to_official_parity():
    script = (LIGHTX2V_ROOT / "scripts/sensenova_vision/start_sensenova_vision_server.sh").read_text()
    assert 'LIGHTX2V_CACHE_DIR="${LIGHTX2V_CACHE_DIR:-${lightx2v_path}/save_results/sensenova_vision_server_cache}"' in script
    assert 'OFFICIAL_PARITY="${SENSENOVA_OFFICIAL_PARITY:-true}"' in script
    assert 'export PYTHONHASHSEED="0"' in script
    assert 'export DTYPE="BF16"' in script
    assert 'export SENSITIVE_LAYER_DTYPE="None"' in script
    assert "--official-parity)" in script
    assert "--no-official-parity)" in script
    assert "--model-path)" in script
    assert "--source-path)" in script
    assert "--config-json)" in script
    assert "--cache-dir)" in script
    assert "export LIGHTX2V_CACHE_DIR" in script

    assert "--task omni_vision_task" in script
    assert '--sensenova_source_path "${SENSENOVA_SOURCE_PATH}"' not in script


def test_sensenova_split_example_scripts_select_exact_task():
    task_names = {
        "01": "understanding",
        "02": "binary_segmentation",
        "03": "depth",
        "04": "normal",
        "05": "gcg_segmentation",
        "06": "object_detection",
        "07": "recon3d",
        "08": "panoptic_segmentation",
        "09": "interactive_segmentation",
        "10": "vgd_segmentation",
        "11": "camera_pose",
        "12": "point_detection",
        "13": "keypoint",
        "14": "ocr",
    }
    script_dir = LIGHTX2V_ROOT / "scripts/sensenova_vision"
    for example_id, task_name in task_names.items():
        script_path = script_dir / f"run_sensenova_vision_example_{example_id}_{task_name}.sh"
        script = script_path.read_text()
        assert script.startswith("#!/usr/bin/env bash")
        assert f'run_sensenova_vision_example.sh" {example_id}' in script
        assert '"$@"' in script

    example_source = (LIGHTX2V_ROOT / "examples/sensenova_vision/example_visualize.py").read_text()
    for example_id in task_names:
        assert f'if selected("{example_id}"):' in example_source

    common_script = (script_dir / "run_sensenova_vision_example.sh").read_text()
    assert "--gpus)" in common_script
    assert 'export CUDA_VISIBLE_DEVICES="${2}"' in common_script
    assert "comma-separated GPU index list" in common_script
