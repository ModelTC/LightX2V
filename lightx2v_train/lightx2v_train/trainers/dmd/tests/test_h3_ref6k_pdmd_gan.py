"""CPU contracts for CM-only student initialization and balanced Ref6k GAN data."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from lightx2v_train.trainers.dmd.student_warmstart import load_student_adapter
from safetensors.torch import save_file
from torch import nn

TRAIN_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_ROOT = TRAIN_ROOT / "scripts"
with patch.object(sys, "path", [str(SCRIPT_ROOT), *sys.path]):
    spec = importlib.util.spec_from_file_location("_ref6k_preflight_test", SCRIPT_ROOT / "minimax_h3_ref6k_pdmd_gan_preflight.py")
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)


@pytest.fixture
def config(tmp_path, monkeypatch):
    values = {
        "H3_MODEL_PATH": str(tmp_path / "base"),
        "H3_STUDENT_INIT": str(tmp_path / "cm" / "train" / "checkpoint-000001500" / "student_ema"),
        "H3_REF6K_CACHE": str(tmp_path / "ref6k" / "metadata.jsonl"),
        "H3_REF6K_OUTPUT": str(tmp_path / "pdmd" / "train"),
        "H3_REF6K_MAX_ITERS": "3000",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    runtime = preflight.leaf("lightx2v_train/runtime/config.py", "_ref6k_test_config")
    return runtime.load_config(TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_pdmd8_gan_from_cm1500_ref6k_fsdp32.yaml")


class AdapterLayer(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.base_layer = nn.Linear(3, 4)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(3, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, 4, bias=False)})
        self.r = {"default": rank}
        self.scaling = {"default": 1.0}


def adapter_model():
    model = nn.Module()
    model.to_q = AdapterLayer(128)
    for name in ("adaln_proj", "norm_out"):
        module = nn.Module()
        module.linear = AdapterLayer(16)
        model.add_module(name, module)
    return model


@pytest.fixture
def cm_export(config):
    export = Path(config["training"]["student"]["checkpoint_path"])
    export.mkdir(parents=True)
    model = adapter_model()
    weights = {name.replace(".default.weight", ".weight"): torch.full_like(value, (index + 1) / 100) for index, (name, value) in enumerate(model.named_parameters()) if ".lora_" in name}
    save_file(weights, export / "pytorch_lora_weights.safetensors")
    lora = config["training"]["student"]["lora"]
    (export / "adapter_config.json").write_text(
        json.dumps({"r": lora["rank"], "lora_alpha": lora["alpha"], "rank_pattern": lora["rank_pattern"], "alpha_pattern": lora["alpha_pattern"], "target_modules": lora["target_modules"]})
    )
    sidecar = {
        "schema_version": 1,
        "enabled": False,
        "export_role": "student",
        "model_name": "minimax_h3_ref2av",
        "inference_requires_preset": False,
        "export_semantics": "learned_lora_only",
        "score_training_adapter": {"enabled": True, "roles": ["fake", "teacher"]},
    }
    (export / "training_adapter.json").write_text(json.dumps(sidecar))
    marker = {"version": 1, "iteration": 1500, "world_size": 32}
    (export.parent / "turbo.complete.json").write_text(json.dumps(marker))
    for rank in range(32):
        state = {**marker, "rank": rank, "fake_updates": 0, "consumed_microbatches": 1500, "recipe": copy.deepcopy(config)}
        state["recipe"]["training"]["method"] = "turbo"
        (export.parent / f"turbo.rank{rank:05d}.json").write_text(json.dumps(state))
    return export, weights


def test_recipe_accepts_student_only_source_and_rejects_fake_transfer(config):
    source, _ = preflight.validate_recipe(config)
    assert source.name == "student_ema"
    config["training"]["fake"]["checkpoint_path"] = str(source.parent)
    with pytest.raises(ValueError, match="Fresh fake"):
        preflight.validate_recipe(config)


@pytest.mark.parametrize("field,value", [("learning_rate", 2e-5), ("checkpoint_path", "/tmp/checkpoint-000001500")])
def test_recipe_rejects_wrong_lr_or_implicit_ema(config, field, value):
    student = config["training"]["student"]
    (student["optimizer"] if field == "learning_rate" else student)[field] = value
    with pytest.raises(ValueError, match="LR|explicitly"):
        preflight.validate_recipe(config)


def test_real_student_loader_preserves_base_and_fresh_optimizer(config, cm_export):
    export, expected = cm_export
    source, helper = preflight.validate_recipe(config)
    info = preflight.validate_source(config, source, helper)
    assert info["weights_only"] == ["student"] and info["new_score_uses_decfg"] is False
    assert info["cm_score_preset_roles"] == ["fake", "teacher"]
    assert not (export.parent / "dist_state").exists()
    model = adapter_model()
    before = {name: value.clone() for name, value in model.named_parameters() if ".lora_" not in name}
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    load_student_adapter(model, export, strict=True, expected_preset={"enabled": False}, model_name="minimax_h3_ref2av")
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, expected[name.replace(".default.weight", ".weight")] if ".lora_" in name else before[name], rtol=0, atol=0)
    assert not optimizer.state


def test_source_requires_every_cm_rank_and_correct_target(config, cm_export):
    export, _ = cm_export
    source, helper = preflight.validate_recipe(config)
    rank_path = export.parent / "turbo.rank00031.json"
    state = json.loads(rank_path.read_text())
    state["recipe"]["model"]["name"] = "minimax_h3_t2av"
    rank_path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="recipes differ"):
        preflight.validate_source(config, source, helper)
    rank_path.unlink()
    with pytest.raises(FileNotFoundError):
        preflight.validate_source(config, source, helper)


def test_source_rejects_student_decfg_dependency(config, cm_export):
    export, _ = cm_export
    sidecar = export / "training_adapter.json"
    record = json.loads(sidecar.read_text())
    record["inference_requires_preset"] = True
    sidecar.write_text(json.dumps(record))
    source, helper = preflight.validate_recipe(config)
    with pytest.raises(RuntimeError, match="without DeCFG dependency"):
        preflight.validate_source(config, source, helper)


def test_source_rejects_missing_output_head(config, cm_export):
    export, weights = cm_export
    save_file({name: value for name, value in weights.items() if not name.startswith("norm_out")}, export / "pytorch_lora_weights.safetensors")
    source, helper = preflight.validate_recipe(config)
    with pytest.raises(ValueError, match="norm_out.linear"):
        preflight.validate_source(config, source, helper)


def test_all_roles_require_original_base_without_decfg(config):
    base = Path(config["model"]["pretrained_model_name_or_path"])
    for name in ("transformer_ref", "text_encoder", "vae", "audio_vae"):
        (base / name).mkdir(parents=True)
        (base / name / "config.json").write_text("{}")
    preflight.validate_models(config)
    config["model"]["fake"]["training_adapter"]["enabled"] = True
    with pytest.raises(ValueError, match="Disable online DeCFG"):
        preflight.validate_models(config)
    config["model"]["fake"]["training_adapter"]["enabled"] = False
    (base / "decfg_merged.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Premerged DeCFG"):
        preflight.validate_models(config)


@pytest.fixture
def dataset(config):
    manifest = Path(config["data"]["train"]["data_path"])
    manifest.parent.mkdir(parents=True)
    payload = manifest.parent / "tensor.pt"
    payload.write_bytes(b"placeholder")
    rows = []
    for count in range(1, 7):
        for index in range(1000):
            condition = manifest.parent / f"condition-{count}-{index}.pt"
            condition.hardlink_to(payload)
            rows.append(
                {
                    "source_id": f"{count}-{index}",
                    "task": "ref2av",
                    "caption": "Video.",
                    "reference_image_count": count,
                    "reference_video_count": 0,
                    "reference_audio_count": 0,
                    "target_num_frames": 124,
                    "target_height": 768,
                    "target_width": 1344,
                    "fps": 24,
                    "vae_normalized": True,
                    "video_latent_shape": [24, 37, 48, 84],
                    "audio_latent_shape": [2, 32, 207],
                    "cache_fingerprint": "condition-identity",
                    "latent_fingerprint": "target-identity",
                    "condition_path": str(condition),
                    "video_latent_path": str(payload),
                    "audio_latent_path": str(payload),
                    "packed_sequence_tokens_124": 40000,
                    "ref6k_source": "selfgen_single_image" if count == 1 else "omni_teacher_base",
                }
            )
    receipt = {
        "schema": "h3_ref6k_v1",
        "schema_version": 1,
        "teacher_weights": "base",
        "completed_count": 6000,
        "input_total_rows": 6000,
        "failed_count": 0,
        "preprocess_fingerprint": "complete",
        "reference_image_counts": {str(count): 1000 for count in range(1, 7)},
        "source_counts": {"omni_teacher_base": 5000, "selfgen_single_image": 1000},
        "teacher_model_path": config["model"]["pretrained_model_name_or_path"],
    }

    def publish():
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        receipt["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        manifest.with_suffix(".complete.json").write_text(json.dumps(receipt))

    publish()
    return rows, receipt, publish


def test_real_gan_manifest_reader_validates_balanced_6k(config, dataset):
    result = preflight.validate_data(config)
    assert result["rows"] == 6000
    assert result["reference_image_counts"] == dict.fromkeys(range(1, 7), 1000)


@pytest.mark.parametrize("change,match", [("count", "1000 rows"), ("receipt", "completion receipt"), ("source", "5000 Omni")])
def test_ref6k_rejects_wrong_counts_or_teacher_provenance(config, dataset, change, match):
    rows, receipt, publish = dataset
    if change == "count":
        rows[0]["reference_image_count"] = 2
    elif change == "receipt":
        receipt["teacher_weights"] = "decfg"
    else:
        rows[0]["ref6k_source"] = "omni_teacher_base"
    publish()
    with pytest.raises(ValueError, match=match):
        preflight.validate_data(config)
