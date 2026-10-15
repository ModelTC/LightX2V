"""Lightweight model catalog; discovery must not import CUDA or serving engines."""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class ModelSpec:
    name: str
    architecture: str
    family: str
    learner_registration: str
    rollout_registration: str
    validator: str | None = None
    preflight: str | None = None


_MODELS: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec):
    """Register a family adapter explicitly, without importing its GPU code."""
    if spec.name in _MODELS:
        raise ValueError(f"Duplicate RL model adapter: {spec.name}")
    _MODELS[spec.name] = spec


def get_model_spec(architecture: str, adapter: str | None = None) -> ModelSpec:
    # Different generations may share a Diffusers pipeline class (e.g. Wan).
    # Explicit adapter identities keep future dual-expert implementations separate.
    candidates = [spec for spec in _MODELS.values() if spec.architecture == architecture
                  and (adapter is None or spec.name == adapter)]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Unsupported or ambiguous RL architecture/adapter: {architecture!r}/{adapter!r}; "
        f"set model.rl_adapter to a matching registered adapter: {', '.join(sorted(_MODELS))}"
    )


def resolve_hook(target: str):
    module, name = target.rsplit(".", 1)
    return getattr(import_module(module), name)


def validate_model_recipe(config, fewstep):
    model = config.actor_rollout_ref.model
    spec = get_model_spec(model.architecture, model.get("rl_adapter"))
    if config.actor_rollout_ref.model.external_lib != spec.learner_registration:
        raise ValueError(f"Model {spec.architecture} requires external_lib={spec.learner_registration}")
    if config.actor_rollout_ref.rollout.external_lib != spec.rollout_registration:
        raise ValueError(f"Model {spec.architecture} requires rollout.external_lib={spec.rollout_registration}")
    if spec.validator:
        resolve_hook(spec.validator)(config, fewstep)


register_model(ModelSpec(
    name="wan2_1", architecture="WanPipeline", family="wan",
    learner_registration="lightx2v_rl.models.wan.register",
    rollout_registration="lightx2v_rl.models.wan.rollout_registration",
))
register_model(ModelSpec(
    name="qwen_image21", architecture="QwenImage21Pipeline", family="qwen_image",
    learner_registration="lightx2v_rl.models.qwen_image.register",
    rollout_registration="lightx2v_rl.models.qwen_image.rollout_registration",
    validator="lightx2v_rl.models.qwen_image.validation.validate_recipe",
    preflight="lightx2v_rl.models.qwen_image.preflight.check",
))
