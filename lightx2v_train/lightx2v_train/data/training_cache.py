import hashlib
import json

CACHE_SCHEMA_VERSION = 2

_RUNTIME_DATA_KEYS = {
    "data_path",
    "name",
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
    signature_data = {
        "model": model,
        "data": {
            "processor": data.get("processor", {}),
            "train": {key: value for key, value in train_data.items() if key not in _RUNTIME_DATA_KEYS},
        },
        "training": {
            "method": method,
            "objective": training.get(method, {}),
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
