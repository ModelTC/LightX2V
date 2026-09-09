import pytest

from lightx2v.disagg.utils import load_wan_transformer


@pytest.mark.parametrize(
    ("model_cls", "model_config", "model_count"),
    [
        ("wan2.1", {}, 1),
        ("wan2.2_moe", {"boundary_step_index": 2}, 2),
    ],
)
def test_disagg_loader_uses_distill_model_class(monkeypatch, model_cls, model_config, model_count):
    selected_methods = []

    class RecordingModel:
        def __init__(self, **kwargs):
            self.model_type = kwargs.get("model_type")

    def select_model_class(distill_method):
        selected_methods.append(distill_method)
        return RecordingModel

    monkeypatch.setattr("lightx2v.models.runners.wan.wan_runner.get_wan_model_class", select_model_class)

    config = {
        "model_cls": model_cls,
        "model_path": "/path/to/model",
        "distill_method": "dmd2",
        "cpu_offload": True,
        **model_config,
    }
    model = load_wan_transformer(config)
    models = model.model if model_cls == "wan2.2_moe" else [model]

    assert selected_methods == ["dmd2"]
    assert len(models) == model_count
    assert all(isinstance(item, RecordingModel) for item in models)


@pytest.mark.parametrize("branch,index", [("high_noise_model", 0), ("low_noise_model", 1)])
def test_disagg_lora_can_target_one_branch(monkeypatch, branch, index):
    monkeypatch.setattr("lightx2v.models.networks.wan.model.WanModel", lambda **kwargs: kwargs)
    config = {
        "model_cls": "wan2.2_moe",
        "model_path": "/path/to/model",
        "cpu_offload": True,
        "boundary": 0.9,
        "lora_dynamic_apply": True,
        "lora_configs": [{"name": branch, "path": "/path/to/adapter.safetensors", "strength": 0.0}],
    }

    models = load_wan_transformer(config).model

    assert models[index]["lora_path"] == "/path/to/adapter.safetensors"
    assert models[index]["lora_strength"] == 0.0
    assert "lora_path" not in models[1 - index]
    assert "lora_strength" not in models[1 - index]
