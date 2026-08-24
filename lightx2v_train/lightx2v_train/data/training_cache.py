import hashlib
import json

CACHE_SCHEMA_VERSION = 2

_DMD_METHODS = {"dmd", "autoregressive_dmd", "phased_dmd", "sgmd"}
_RUNTIME_MODEL_KEYS = {
    "cache_condition_encoder_cpu_offload",
    "cache_encoder_cpu_offload",
    "load_condition_encoder",
    "load_text_encoder",
    "load_transformer",
    "load_vae",
}

_RUNTIME_DATA_KEYS = {
    "data_path",
    "dataset_repeat",
    "drop_last",
    "max_samples",
    "num_workers",
    "persistent_workers",
    "pin_memory",
    "prefetch_factor",
    "preserve_records",
    "shuffle",
}


def preserve_cache_dtype(key):
    return isinstance(key, str) and (key.endswith("_ids") or key.endswith("_mask"))


def training_cache_info(config):
    model = config["model"]
    data = config.get("data", {})
    train_data = data.get("train", {})
    training = config["training"]
    method = training["method"]
    objective_key = "dmd" if method in _DMD_METHODS else method
    signature_data = {
        "model": {key: value for key, value in model.items() if key not in _RUNTIME_MODEL_KEYS},
        "data": {
            "processor": data.get("processor", {}),
            "train": {key: value for key, value in train_data.items() if key not in _RUNTIME_DATA_KEYS},
        },
        "training": {
            "method": method,
            "objective": training.get(objective_key, {}),
            "teacher": training.get("teacher", {}),
        },
        "target_latent_mode": "mode",
    }
    serialized = json.dumps(signature_data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_name": model["name"],
        "model_path": model["pretrained_model_name_or_path"],
        "training_method": method,
        "signature": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }
