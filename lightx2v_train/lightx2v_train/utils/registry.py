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


_MODEL_MODULES = {
    "flux2_dev": "lightx2v_train.model_zoo.flux2_dev",
    "flux2_klein": "lightx2v_train.model_zoo.flux2_klein",
    "lingbot_video": "lightx2v_train.model_zoo.lingbot_video",
    "longcat_image": "lightx2v_train.model_zoo.longcat_image",
    "ltx_t2av": "lightx2v_train.model_zoo.ltx_t2av",
    "ltx_t2av_ar": "lightx2v_train.model_zoo.ltx_t2av",
    "qwen_image": "lightx2v_train.model_zoo.qwen_image",
    "qwen_image_edit": "lightx2v_train.model_zoo.qwen_image_edit",
    "wan_t2v": "lightx2v_train.model_zoo.wan_t2v",
    "wan_t2v_ar": "lightx2v_train.model_zoo.wan_t2v",
    "wan_t2v_14b": "lightx2v_train.model_zoo.wan_t2v",
    "wan_t2v_14b_ar": "lightx2v_train.model_zoo.wan_t2v",
    "wan_fastwam": "lightx2v_train.model_zoo.wan_fastwam",
    "wan_ti2v_5b": "lightx2v_train.model_zoo.wan_ti2v_5b",
    "wan_ti2v_5b_ar": "lightx2v_train.model_zoo.wan_ti2v_5b",
}

_TRAINER_MODULES = {
    "dmd": "lightx2v_train.trainers.dmd.trainer",
    "dopsd": "lightx2v_train.trainers.dopsd",
    "fastwam": "lightx2v_train.trainers.fastwam",
    "flow": "lightx2v_train.trainers.flow",
    "lingbot_video_dmd": "lightx2v_train.trainers.dmd.video_trainer",
    "ltx_t2av_ar_dmd": "lightx2v_train.trainers.dmd.ltx_trainer",
    "ltx_t2av_dmd": "lightx2v_train.trainers.dmd.ltx_trainer",
    "ltx_t2av_flow": "lightx2v_train.trainers.flow",
    "ltx_t2av_teacher_forcing": "lightx2v_train.trainers.tf",
    "teacher_forcing": "lightx2v_train.trainers.tf",
    "video_ar_dmd": "lightx2v_train.trainers.dmd.video_ar_trainer",
    "video_dmd": "lightx2v_train.trainers.dmd.video_trainer",
    "video_phased_dmd": "lightx2v_train.trainers.phased_dmd.trainer",
    "video_sgmd": "lightx2v_train.trainers.sgmd",
}

_INFERENCER_MODULES = {
    "image_infer": "lightx2v_train.infer.image",
    "image_native_infer": "lightx2v_train.infer.image_native",
    "lingbot_video_t2v_infer": "lightx2v_train.infer.video",
    "wan_t2v_infer": "lightx2v_train.infer.video",
    "wan_t2v_14b_infer": "lightx2v_train.infer.video",
    "wan_t2v_dual_infer": "lightx2v_train.infer.video",
    "wan_t2v_ar_infer": "lightx2v_train.infer.video",
    "wan_t2v_14b_ar_infer": "lightx2v_train.infer.video",
    "wan_ti2v_5b_infer": "lightx2v_train.infer.video",
    "wan_ti2v_5b_ar_infer": "lightx2v_train.infer.video",
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
    elif data_name == "libero_fastwam_dataset":
        import lightx2v_train.data.libero.dataset  # noqa: F401
    elif data_name in {"latent_dataset", "prompt_dataset", "video_dataset"}:
        import lightx2v_train.data.video_dataset  # noqa: F401


def build_model(config):
    name = config["model"]["name"]
    _ensure_registered(name, MODEL_REGISTER, _MODEL_MODULES)
    if name not in MODEL_REGISTER:
        available = ", ".join(sorted(MODEL_REGISTER.keys()))
        raise ValueError(f"Unknown model {name!r}. Available models: {available}")
    return MODEL_REGISTER[name](config)


def build_trainer(config):
    name = config["training"]["method"]
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


def build_data(config, train_or_val):
    data_config = config.get("data", {})
    if train_or_val not in data_config:
        available_splits = ", ".join(repr(k) for k in sorted(data_config.keys()))
        raise ValueError(f"config['data'] has no key {train_or_val!r}. Available keys: {available_splits}")
    data_config_split = data_config[train_or_val]
    data_name = data_config_split.get("name", "image_dataset")
    _ensure_data_registered(data_name)
    if data_name not in DATA_REGISTER:
        available_names = ", ".join(sorted(DATA_REGISTER.keys()))
        raise ValueError(f"Unknown data {data_name!r}. Available data: {available_names}")
    return DATA_REGISTER[data_name](data_config_split, train_or_val=train_or_val)
