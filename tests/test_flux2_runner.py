import pytest
import torch

from lightx2v.models.networks.flux2.model import Flux2DevTransformerModel, Flux2KleinTransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.flux2.flux2_runner import Flux2Runner
from lightx2v.models.schedulers.flux2.feature_caching.scheduler import Flux2DevSchedulerCaching, Flux2SchedulerCaching
from lightx2v.models.schedulers.flux2.scheduler import Flux2DevScheduler, Flux2Scheduler
from lightx2v.utils.registry_factory import RUNNER_REGISTER


def test_flux2_runner_registration():
    assert RUNNER_REGISTER["flux2"] is Flux2Runner
    assert "flux2_klein" not in RUNNER_REGISTER
    assert "flux2_dev" not in RUNNER_REGISTER


@pytest.mark.parametrize(
    ("model_variant", "transformer_class", "scheduler_class", "caching_scheduler_class"),
    [
        ("klein", Flux2KleinTransformerModel, Flux2Scheduler, Flux2SchedulerCaching),
        ("dev", Flux2DevTransformerModel, Flux2DevScheduler, Flux2DevSchedulerCaching),
    ],
)
def test_flux2_variant_components(monkeypatch, model_variant, transformer_class, scheduler_class, caching_scheduler_class):
    monkeypatch.setattr(DefaultRunner, "__init__", lambda self, config: setattr(self, "config", config))

    runner = Flux2Runner({"model_variant": model_variant})

    assert runner.transformer_class is transformer_class
    assert runner.scheduler_class is scheduler_class
    assert runner.caching_scheduler_class is caching_scheduler_class


@pytest.mark.parametrize("model_variant", [None, "unknown"])
def test_flux2_model_variant_is_required(model_variant):
    with pytest.raises(ValueError, match="Unsupported Flux2 model_variant"):
        Flux2Runner({"model_variant": model_variant})


class RecordingTextEncoder:
    def __init__(self):
        self.prompts = []

    def infer(self, prompts):
        self.prompts.append(prompts)
        return [torch.zeros(2, 4)], None


@pytest.mark.parametrize(
    ("model_variant", "enable_cfg", "sample_guide_scale", "uses_negative_prompt"),
    [
        ("klein", True, 4.0, True),
        ("klein", False, 4.0, False),
        ("klein", False, 1.0, False),
        ("klein", True, 0.5, True),
        ("dev", True, 4.0, False),
    ],
)
def test_flux2_text_encoder_matches_cfg_execution(
    monkeypatch,
    model_variant,
    enable_cfg,
    sample_guide_scale,
    uses_negative_prompt,
):
    monkeypatch.setattr("lightx2v.models.runners.flux2.flux2_runner.AI_DEVICE", "cpu")
    text_encoder = RecordingTextEncoder()
    runner = object.__new__(Flux2Runner)
    runner.model_variant = model_variant
    runner.config = {
        "enable_cfg": enable_cfg,
        "sample_guide_scale": sample_guide_scale,
    }
    runner.text_encoders = [text_encoder]

    output = runner.run_text_encoder("prompt")

    assert ("negative_prompt_embeds" in output) is uses_negative_prompt
    expected_prompts = [["prompt"], [""]] if uses_negative_prompt else [["prompt"]]
    assert text_encoder.prompts == expected_prompts
