from types import SimpleNamespace

import pytest

from lightx2v.models.networks.wan.edenlive_model import EdenLiveARModel, EdenLiveModel
from lightx2v.models.networks.wan.model import WanModel


def make_model():
    events = []
    model = EdenLiveARModel.__new__(EdenLiveARModel)
    model.config = {"skip_cache_update_suffix": True}
    model.cpu_offload = False
    model.offload_granularity = "model"
    model.scheduler = SimpleNamespace(cache_update=True, infer_condition=False, noise_pred=object())
    model.pre_weight = object()
    model.transformer_weights = object()
    pre = object()
    model.pre_infer = SimpleNamespace(infer=lambda weights, inputs: events.append(("pre", weights, inputs)) or pre)
    model.transformer_infer = SimpleNamespace(infer=lambda weights, value: events.append(("transformer", weights, value)))
    model.to_cuda = lambda: events.append("cuda")
    return model, pre, events


def test_commit_only_skips_prediction_and_preserves_last_noise(monkeypatch):
    model, pre, events = make_model()
    monkeypatch.setattr(EdenLiveModel, "infer", lambda self, inputs: pytest.fail("prediction path must be skipped"))
    previous = model.scheduler.noise_pred
    model.infer("inputs")
    assert events == [("pre", model.pre_weight, "inputs"), ("transformer", model.transformer_weights, pre)]
    assert model.scheduler.infer_condition is True and model.scheduler.noise_pred is previous


@pytest.mark.parametrize("disabled", ["flag", "denoising", "cpu_offload", "enable_cfg", "seq_parallel", "tensor_parallel", "lazy_load", "clean_cuda_cache"])
def test_prediction_or_unsupported_mode_uses_parent(monkeypatch, disabled):
    model, _, events = make_model()
    if disabled == "flag":
        model.config["skip_cache_update_suffix"] = False
    elif disabled == "denoising":
        model.scheduler.cache_update = False
    elif disabled == "cpu_offload":
        model.cpu_offload = True
    else:
        model.config[disabled] = True
    monkeypatch.setattr(EdenLiveModel, "infer", lambda self, inputs: events.append(("parent", inputs)))
    model.infer("inputs")
    assert events == (["cuda"] if disabled == "cpu_offload" else []) + [("parent", "inputs")]


@pytest.mark.parametrize("trim", [False, True])
def test_only_final_layer_builds_shared_commit_kv(monkeypatch, trim):
    import lightx2v.models.networks.wan.edenlive_model as module

    class Factory:
        @classmethod
        def from_modules(cls, modules, packed=False):
            return (modules, packed)

    monkeypatch.setattr(module, "ConvRotProjectionGroup", Factory)
    monkeypatch.setattr(WanModel, "_apply_weights", lambda self, weights: None)
    model = EdenLiveModel.__new__(EdenLiveModel)
    model.config = {"dit_quant_scheme": "int8-convrot", "share_qkv_activation_quant": True, "pack_qkv": True, "skip_cache_update_suffix": trim}
    phases = [SimpleNamespace(self_attn_q=object(), self_attn_k=object(), self_attn_v=object()) for _ in range(2)]
    model.transformer_weights = SimpleNamespace(blocks=[SimpleNamespace(compute_phases=[phase, SimpleNamespace(cross_attn_k=object(), cross_attn_v=object())]) for phase in phases])
    model._apply_weights()
    assert phases[0].self_attn_cache_kv_projection_group is None
    expected = ((phases[1].self_attn_k, phases[1].self_attn_v), False) if trim else None
    assert phases[1].self_attn_cache_kv_projection_group == expected
