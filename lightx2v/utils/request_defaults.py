"""Resolve configured request defaults without making them explicit requests."""

from collections.abc import Mapping
from copy import copy

from lightx2v.models.runners.registry import RUNNER_MODULES, get_model_spec

FIXED_FRAME_ATTENTION_TYPES = frozenset({"svg_attn", "radial_attn", "nbhd_attn", "nbhd_attn_flashinfer", "general_sparse_attn"})


def get_request_default_arguments(config):
    """Return serializable, shared CLI/HTTP defaults supported by this deployment."""
    model_cls = config.get("model_cls")
    if model_cls not in RUNNER_MODULES:
        raise ValueError("A registered model_cls is required to validate request_defaults")
    spec = get_model_spec(model_cls)
    task = config.get("task")
    if task is not None and task not in spec.tasks:
        raise ValueError(f"{model_cls} does not support task {task!r}")
    try:
        tasks = spec.get_tasks(config)
    except (KeyError, ValueError):
        # Configuration loaders and internal tools may resolve defaults before
        # the caller selects a task or supplies checkpoint-specific variants.
        tasks = (task,) if task is not None else tuple(spec.tasks)
    arguments = {}
    for task in tasks:
        for name, argument in spec.get_arguments(config, task).items():
            if name == "task" or not (argument.cli and argument.server):
                continue
            previous = arguments.get(name)
            if previous is not None and previous.type != argument.type:
                argument = copy(argument)
                argument.type = previous.type | argument.type
            arguments[name] = argument

    if RUNNER_MODULES[model_cls].runner.startswith("wan.") and model_cls != "tbdub" and config.get("self_attn_1_type") in FIXED_FRAME_ATTENTION_TYPES:
        arguments.pop("num_frames", None)
    if model_cls == "qwen_image" and config.get("layered", False):
        arguments.pop("i2i_denoise_strength", None)
    if model_cls == "seko_talk_ar" and config.get("prompt_travel"):
        arguments.pop("prompt", None)
    return arguments


def validate_request_defaults(config, *, arguments=None):
    """Validate the namespace and return its values without expanding the config."""
    defaults = config.get("request_defaults", {})
    if not isinstance(defaults, Mapping):
        raise ValueError("request_defaults must be a JSON object")
    conflicts = defaults.keys() & config.keys()
    if conflicts:
        raise ValueError(f"Request defaults must not also appear at the top level: {', '.join(sorted(conflicts))}")
    if not defaults:
        return {}
    from lightx2v.utils.request_spec import create_request_model

    arguments = get_request_default_arguments(config) if arguments is None else arguments
    unsupported = defaults.keys() - arguments.keys()
    if unsupported:
        raise ValueError(f"{config['model_cls']} does not support request_defaults fields: {', '.join(sorted(unsupported))}")
    return create_request_model("RequestDefaults", arguments).model_validate(defaults).model_dump(exclude_unset=True)


_MISSING = object()


def get_input_value(config, name, default=_MISSING):
    """Read a fixed/explicit input or a nested default without changing config.

    Startup configs keep request fields exclusively in request_defaults. A
    dispatched request may also carry explicit inputs at its top level.
    """
    if name in config:
        return config[name]
    defaults = config.get("request_defaults", {})
    if name in defaults:
        return defaults[name]
    if default is _MISSING:
        raise KeyError(name)
    return default


def set_input_value(config, name, value):
    """Normalize an input in its own namespace; never flatten request defaults."""
    if name in config:
        config[name] = value
    elif name in config.get("request_defaults", {}) or name in get_request_default_arguments(config):
        defaults = dict(config.get("request_defaults", {}))
        defaults[name] = value
        config["request_defaults"] = defaults
    else:
        config[name] = value
