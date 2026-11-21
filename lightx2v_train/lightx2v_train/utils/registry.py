import importlib
from collections.abc import MutableMapping

from lightx2v_train.utils.utils import is_cache_build


def _qualified_name(target):
    return f"{target.__module__}.{target.__qualname__}"


class Register(MutableMapping):
    def __init__(self, *args, builtin_modules=None, **kwargs):
        self._dict = {}
        self._builtin_modules = builtin_modules or {}
        self.update(dict(*args, **kwargs))

    def __call__(self, target_or_name):
        if callable(target_or_name):
            return self.register(target_or_name)

        def decorate(target):
            return self.register(target, key=target_or_name)

        return decorate

    def register(self, target, key=None):
        if not callable(target):
            raise TypeError(f"{target!r} must be callable.")

        if key is None:
            key = target.__name__

        builtin_module = self._builtin_modules.get(key)
        if builtin_module is not None and target.__module__ != builtin_module:
            raise ValueError(f"{key!r} belongs to built-in module {builtin_module!r}; cannot register {_qualified_name(target)}.")
        if key in self._dict:
            previous = self._dict[key]
            raise ValueError(f"{key!r} is already registered by {_qualified_name(previous)}; cannot register {_qualified_name(target)}.")
        self._dict[key] = target
        return target

    def __setitem__(self, key, value):
        self.register(value, key)

    def __getitem__(self, key):
        return self._dict[key]

    def __delitem__(self, key):
        del self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def merge(self, other_register):
        self.update(other_register)


_MODEL_MODULES = {
    "flux2_dev": "lightx2v_train.model_zoo.flux2.flux2_dev",
    "flux2_dev_edit": "lightx2v_train.model_zoo.flux2.flux2_dev_edit",
    "flux2_klein": "lightx2v_train.model_zoo.flux2.flux2_klein",
    "flux2_klein_edit": "lightx2v_train.model_zoo.flux2.flux2_klein_edit",
    "lingbot_video": "lightx2v_train.model_zoo.wan.lingbot_video",
    "longcat_image": "lightx2v_train.model_zoo.longcat_image.longcat_image",
    "longcat_image_edit": "lightx2v_train.model_zoo.longcat_image.longcat_image_edit",
    "minimax_h3_t2av": "lightx2v_train.model_zoo.minimax_h3.minimax_h3_t2av",
    "qwen_image": "lightx2v_train.model_zoo.qwen_image.qwen_image",
    "qwen_image_2_1": "lightx2v_train.model_zoo.qwen_image.qwen_image_2_1",
    "qwen_image_edit": "lightx2v_train.model_zoo.qwen_image.qwen_image_edit",
    "wan_t2v": "lightx2v_train.model_zoo.wan.wan_t2v",
    "wan_t2v_ar": "lightx2v_train.model_zoo.wan.wan_t2v",
    "wan_t2v_14b": "lightx2v_train.model_zoo.wan.wan_t2v",
    "wan_t2v_14b_ar": "lightx2v_train.model_zoo.wan.wan_t2v",
}

_TRAINER_MODULES = {
    "video_cfg_distill": "lightx2v_train.trainers.cfg_distill.trainer",
    "video_cfg_trajectory_distill": "lightx2v_train.trainers.cfg_trajectory_distill.trainer",
    "autoregressive_dmd": "lightx2v_train.trainers.dmd.autoregressive_dmd",
    "consistency": "lightx2v_train.trainers.consistency.trainer",
    "dmd": "lightx2v_train.trainers.dmd.trainer",
    "flow_matching": "lightx2v_train.trainers.flow_matching",
    "phased_dmd": "lightx2v_train.trainers.phased_dmd.trainer",
    "sgmd": "lightx2v_train.trainers.sgmd",
    "teacher_forcing": "lightx2v_train.trainers.teacher_forcing",
    "cache_build": "lightx2v_train.trainers.cache_build",
}

_INFERENCER_MODULES = {
    "image_infer": "lightx2v_train.infer.image",
    "lingbot_video_t2v_infer": "lightx2v_train.infer.video",
    "wan_t2v_infer": "lightx2v_train.infer.video",
    "wan_t2v_14b_infer": "lightx2v_train.infer.video",
    "wan_t2v_dual_infer": "lightx2v_train.infer.video",
    "wan_t2v_ar_infer": "lightx2v_train.infer.video",
    "wan_t2v_14b_ar_infer": "lightx2v_train.infer.video",
}

_SAMPLE_PROCESSOR_MODULES = {
    "flux2_dev": "lightx2v_train.model_zoo.flux2.data_process",
    "flux2_dev_edit": "lightx2v_train.model_zoo.flux2.data_process",
    "flux2_klein": "lightx2v_train.model_zoo.flux2.data_process",
    "flux2_klein_edit": "lightx2v_train.model_zoo.flux2.data_process",
    "longcat_image": "lightx2v_train.model_zoo.longcat_image.data_process",
    "longcat_image_edit": "lightx2v_train.model_zoo.longcat_image.data_process",
    "minimax_h3_t2av": "lightx2v_train.model_zoo.minimax_h3.data_process",
    "qwen_image": "lightx2v_train.model_zoo.qwen_image.data_process",
    "qwen_image_edit": "lightx2v_train.model_zoo.qwen_image.data_process",
    "qwen_image_2_1": "lightx2v_train.model_zoo.qwen_image.data_process",
}

_DATA_MODULES = {
    "image_dataset": "lightx2v_train.data.image_dataset",
    "cache_dataset": "lightx2v_train.data.cache_dataset",
    "prompt_dataset": "lightx2v_train.data.video_dataset",
    "video_dataset": "lightx2v_train.data.video_dataset",
}

MODEL_REGISTER = Register(builtin_modules=_MODEL_MODULES)
TRAINER_REGISTER = Register(builtin_modules=_TRAINER_MODULES)
INFERENCER_REGISTER = Register(builtin_modules=_INFERENCER_MODULES)
DATA_REGISTER = Register(builtin_modules=_DATA_MODULES)
SAMPLE_PROCESSOR_REGISTER = Register(builtin_modules=_SAMPLE_PROCESSOR_MODULES)


def _load_builtin(name, register, module_map):
    if name in register:
        return
    module_name = module_map.get(name)
    if module_name is not None:
        importlib.import_module(module_name)


def build_model(config):
    name = config["model"]["name"]
    _load_builtin(name, MODEL_REGISTER, _MODEL_MODULES)
    if name not in MODEL_REGISTER:
        available = ", ".join(sorted(MODEL_REGISTER))
        raise ValueError(f"Unknown model {name!r}. Available models: {available}")
    return MODEL_REGISTER[name](config)


def build_trainer(config):
    name = "cache_build" if is_cache_build(config) else config["training"]["method"]
    _load_builtin(name, TRAINER_REGISTER, _TRAINER_MODULES)
    if name not in TRAINER_REGISTER:
        available = ", ".join(sorted(TRAINER_REGISTER))
        raise ValueError(f"Unknown trainer {name!r}. Available trainers: {available}")
    return TRAINER_REGISTER[name](config)


def build_inferencer(config):
    name = config["inference"]["method"]
    _load_builtin(name, INFERENCER_REGISTER, _INFERENCER_MODULES)
    if name not in INFERENCER_REGISTER:
        available = ", ".join(sorted(INFERENCER_REGISTER))
        raise ValueError(f"Unknown inferencer {name!r}. Available inferencers: {available}")
    return INFERENCER_REGISTER[name](config)


def build_sample_processor(config):
    processor_config = config.get("data", {}).get("processor", {})
    name = processor_config.get("name", config["model"]["name"])
    _load_builtin(name, SAMPLE_PROCESSOR_REGISTER, _SAMPLE_PROCESSOR_MODULES)
    if name not in SAMPLE_PROCESSOR_REGISTER:
        if "name" not in processor_config:
            return None
        available = ", ".join(sorted(SAMPLE_PROCESSOR_REGISTER))
        raise ValueError(f"Unknown sample processor {name!r}. Available processors: {available}")
    return SAMPLE_PROCESSOR_REGISTER[name](config)


def build_data(config, train_or_val, sample_processor=None):
    data_config = config.get("data", {})
    if train_or_val not in data_config:
        available_splits = ", ".join(repr(k) for k in sorted(data_config))
        raise ValueError(f"config['data'] has no key {train_or_val!r}. Available keys: {available_splits}")
    split_config = dict(data_config[train_or_val])

    if is_cache_build(config):
        split_config.update(
            prompt_dropout_rate=0.0,
            dataset_repeat=1,
            shuffle=False,
            drop_last=False,
            decode_retries=1,
        )
        # Validation cache building also needs to partition samples across ranks.
        split_config["distributed_cache_build"] = True
        split_config.pop("max_samples", None)

    data_name = split_config.get("name", "image_dataset")
    if train_or_val == "train" and data_name in {"prompt_dataset", "cache_dataset"}:
        generation_shapes = config.get("training", {}).get("dmd", {}).get("generation_shapes")
        if generation_shapes is not None:
            split_config["generation_shapes"] = generation_shapes

    _load_builtin(data_name, DATA_REGISTER, _DATA_MODULES)
    if data_name not in DATA_REGISTER:
        available_names = ", ".join(sorted(DATA_REGISTER))
        raise ValueError(f"Unknown data {data_name!r}. Available data: {available_names}")

    kwargs = {}
    if data_name in {"cache_dataset", "image_dataset", "video_dataset"}:
        kwargs = {
            "sample_processor": sample_processor,
            "unconditional_prompt": getattr(sample_processor, "unconditional_prompt", " "),
        }

    return DATA_REGISTER[data_name](split_config, train_or_val=train_or_val, **kwargs)
