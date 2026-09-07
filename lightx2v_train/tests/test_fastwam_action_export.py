import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from torch import nn


@pytest.fixture
def exporter():
    path = Path(__file__).parents[1] / "tools/export_fastwam_action_dmd.py"
    spec = importlib.util.spec_from_file_location("action_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method,weights,expected", [
    ("fastwam_action_dmd", "auto", "student"),
    ("fastwam_action_consistency", "auto", "ema"),
    ("fastwam_action_consistency", "student", "student"),
])
def test_export_selects_weights_and_merges_lora(exporter, tmp_path, monkeypatch, method, weights, expected):
    role = {"train_type": "lora", "optimizer": {}, "lora": {
        "rank": 2, "alpha": 2, "target_modules": ["q"], "modules_to_save": [],
    }}
    config = {"training": {"method": method, "student": role, "fake": role,
                           "action_dmd": {}, "action_consistency": {}}}
    checkpoint = tmp_path / "checkpoint-000030000"
    checkpoint.mkdir()
    config_path = checkpoint / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    parsed = exporter.parse_training_config(config)

    def expert():
        model = nn.Module()
        model.q = nn.Linear(3, 3, bias=False)
        nn.init.zeros_(model.q.weight)
        return model

    wrapped = exporter.configure_action_role(expert(), parsed.student)
    state = get_peft_model_state_dict(wrapped)
    for name, value in (("student", 1.0), ("ema", 2.0)):
        torch.save({key: torch.full_like(tensor, value) for key, tensor in state.items()},
                   checkpoint / f"{name}_action.pt")

    module = SimpleNamespace(action_expert=expert(), mot=SimpleNamespace(mixtures={}))
    module.save_checkpoint = lambda path, step: torch.save(
        {"action": module.action_expert.state_dict(), "step": step}, path)
    model = SimpleNamespace(load_components=lambda: None, unwrap_module=lambda: module)
    monkeypatch.setattr(exporter, "build_model", lambda config: model)
    output = tmp_path / "merged.pt"
    adapters = tmp_path / "lora"
    monkeypatch.setattr(exporter, "parse_args", lambda: SimpleNamespace(
        config=str(config_path), checkpoint=str(checkpoint), output=str(output),
        weights=weights, lora_output=str(adapters)))
    exporter.main()

    merged = torch.load(output, weights_only=True)
    value = 2.0 if expected == "ema" else 1.0
    torch.testing.assert_close(merged["action"]["q.weight"], torch.full((3, 3), 2 * value ** 2))
    assert merged["step"] == 30000
    assert module.mot.mixtures["action"] is module.action_expert
    saved = load_file(adapters / "action/adapter_model.safetensors")
    for tensor in saved.values():
        torch.testing.assert_close(tensor, torch.full_like(tensor, value))


def test_dmd_rejects_ema(exporter):
    role = {"train_type": "full", "optimizer": {}}
    parsed = exporter.parse_training_config({"training": {
        "student": role, "fake": role, "action_dmd": {},
    }})
    with pytest.raises(ValueError, match="only available for consistency"):
        exporter.resolve_weights(parsed, "ema")
