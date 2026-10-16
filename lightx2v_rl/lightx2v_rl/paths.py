"""Resolve portable recipe paths at the runtime boundary, independent of cwd."""

from pathlib import Path

from omegaconf import OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def repository_path(value, root=None):
    if value is None or not isinstance(value, (str, Path)):
        return value
    value = str(value)
    if not value or "://" in value:
        return value
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else Path(root or REPOSITORY_ROOT) / path).resolve())


def resolve_runtime_paths(config, *, root=None):
    """Return a copy; saved source recipes retain their relative path values.

    Ray workers may start from another working directory. Only local filesystem
    fields are resolved; package imports and service URLs are never rewritten.
    """
    result = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    scalar_keys = (
        "actor_rollout_ref.model.path",
        "actor_rollout_ref.model.tokenizer_path",
        "actor_rollout_ref.model.config_path",
        "actor_rollout_ref.model.lora_adapter_path",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path",
        "reward.custom_reward_function.path",
        "data.custom_cls.path",
        "data.sampler.class_path",
        "trainer.default_local_dir",
        "trainer.rollout_data_dir",
        "trainer.validation_data_dir",
        "trainer.metric_logging.raw_dir",
    )
    for key in scalar_keys:
        value = OmegaConf.select(result, key)
        if value:
            OmegaConf.update(result, key, repository_path(value, root))
    for key in ("data.train_files", "data.val_files"):
        value = OmegaConf.select(result, key)
        if value is not None:
            resolved = repository_path(value, root) if isinstance(value, str) else [repository_path(item, root) for item in value]
            OmegaConf.update(result, key, resolved)
    return result
