import importlib
from collections.abc import MutableMapping


class Register(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)

    def __call__(self, target_or_name):
        if callable(target_or_name):
            return self.register(target_or_name)
        else:
            return lambda x: self.register(x, key=target_or_name)

    def register(self, target, key=None):
        if not callable(target):
            raise Exception(f"Error: {target} must be callable!")

        if key is None:
            key = target.__name__

        if key in self._dict:
            raise Exception(f"{key} already exists.")

        self[key] = target
        return target

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __delitem__(self, key):
        del self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def get(self, key, default=None):
        return self._dict.get(key, default)

    def merge(self, other_register):
        for key, value in other_register.items():
            if key in self._dict:
                raise Exception(f"{key} already exists in target register.")
            self[key] = value


MODEL_REGISTER = Register()
TRAINER_REGISTER = Register()
INFERENCER_REGISTER = Register()
DATA_REGISTER = Register()
SAMPLE_PROCESSOR_REGISTER = Register()


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
    "qwen_image_edit": "lightx2v_train.model_zoo.qwen_image.qwen_image_edit",
    "wan_t2v": "lightx2v_train.model_zoo.wan.wan_t2v",
    "wan_t2v_ar": "lightx2v_train.model_zoo.wan.wan_t2v",
    "wan_t2v_14b": "lightx2v_train.model_zoo.wan.wan_t2v",
    "wan_t2v_14b_ar": "lightx2v_train.model_zoo.wan.wan_t2v",
}

_TRAINER_MODULES = {
    "autoregressive_dmd": "lightx2v_train.trainers.dmd.autoregressive_dmd",
    "consistency": "lightx2v_train.trainers.consistency.trainer",
    "dmd": "lightx2v_train.trainers.dmd.trainer",
    "flow_matching": "lightx2v_train.trainers.flow_matching",
    "phased_dmd": "lightx2v_train.trainers.phased_dmd.trainer",
    "sgmd": "lightx2v_train.trainers.sgmd",
    "teacher_forcing": "lightx2v_train.trainers.teacher_forcing",
    "training_cache": "lightx2v_train.trainers.training_cache",
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
}


def _ensure_registered(name, register, module_map):
    if name in register:
        return
    module_name = module_map.get(name)
    if module_name is not None:
        importlib.import_module(module_name)


def _ensure_data_registered(data_name):
    if data_name in DATA_REGISTER:
        return
    if data_name == "image_dataset":
        import lightx2v_train.data.image_dataset  # noqa: F401
    elif data_name == "training_cache_dataset":
        import lightx2v_train.data.training_cache_dataset  # noqa: F401
    elif data_name in {"prompt_dataset", "video_dataset"}:
        import lightx2v_train.data.video_dataset  # noqa: F401


def build_model(config):
    name = config["model"]["name"]
    _ensure_registered(name, MODEL_REGISTER, _MODEL_MODULES)
    if name not in MODEL_REGISTER:
        available = ", ".join(sorted(MODEL_REGISTER.keys()))
        raise ValueError(f"Unknown model {name!r}. Available models: {available}")
    return MODEL_REGISTER[name](config)


def build_trainer(config):
    name = "training_cache" if "training_cache" in config else config["training"]["method"]
    _ensure_registered(name, TRAINER_REGISTER, _TRAINER_MODULES)
    if name not in TRAINER_REGISTER:
        available = ", ".join(sorted(TRAINER_REGISTER.keys()))
        raise ValueError(f"Unknown trainer {name!r}. Available trainers: {available}")
    return TRAINER_REGISTER[name](config)


def build_inferencer(config):
    name = config["inference"]["method"]
    _ensure_registered(name, INFERENCER_REGISTER, _INFERENCER_MODULES)
    if name not in INFERENCER_REGISTER:
        available = ", ".join(sorted(INFERENCER_REGISTER.keys()))
        raise ValueError(f"Unknown inferencer {name!r}. Available inferencers: {available}")
    return INFERENCER_REGISTER[name](config)


def build_sample_processor(config):
    processor_config = config.get("data", {}).get("processor", {})
    name = processor_config.get("name", config["model"]["name"])
    _ensure_registered(name, SAMPLE_PROCESSOR_REGISTER, _SAMPLE_PROCESSOR_MODULES)
    if name not in SAMPLE_PROCESSOR_REGISTER:
        if "name" not in processor_config:
            return None
        available = ", ".join(sorted(SAMPLE_PROCESSOR_REGISTER.keys()))
        raise ValueError(f"Unknown sample processor {name!r}. Available processors: {available}")
    return SAMPLE_PROCESSOR_REGISTER[name](config)


def build_data(config, train_or_val, sample_processor=None):
    from lightx2v_train.data.training_cache import training_cache_info

    data_config = config.get("data", {})
    if train_or_val not in data_config:
        available_splits = ", ".join(repr(k) for k in sorted(data_config.keys()))
        raise ValueError(f"config['data'] has no key {train_or_val!r}. Available keys: {available_splits}")
    data_config_split = dict(data_config[train_or_val])
    if "training_cache" in config:
        data_config_split.update(
            prompt_dropout_rate=0.0,
            dataset_repeat=1,
            shuffle=False,
            drop_last=False,
            decode_retries=1,
        )
        data_config_split.pop("max_samples", None)
    data_name = data_config_split.get("name", "image_dataset")
    use_training_cache = data_config.get("use_training_cache", False) if train_or_val == "train" and "training_cache" not in config else False
    if use_training_cache:
        data_name = "training_cache_dataset"
    if train_or_val == "train" and data_name == "prompt_dataset":
        image_sizes = config.get("training", {}).get("dmd", {}).get("image_sizes")
        if image_sizes is not None:
            data_config_split["image_sizes"] = image_sizes
    _ensure_data_registered(data_name)
    if data_name not in DATA_REGISTER:
        available_names = ", ".join(sorted(DATA_REGISTER.keys()))
        raise ValueError(f"Unknown data {data_name!r}. Available data: {available_names}")
    kwargs = (
        {
            "unconditional_prompt": getattr(sample_processor, "unconditional_prompt", " "),
            "expected_cache_info": training_cache_info(config) if use_training_cache else None,
        }
        if data_name == "training_cache_dataset"
        else {
            "sample_processor": sample_processor,
            "unconditional_prompt": getattr(sample_processor, "unconditional_prompt", " "),
        }
        if data_name in {"image_dataset", "video_dataset"}
        else {}
    )
    return DATA_REGISTER[data_name](data_config_split, train_or_val=train_or_val, **kwargs)
