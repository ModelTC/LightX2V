import importlib

from lightx2v_train.utils.registry import Register

CONSISTENCY_OBJECTIVE_REGISTER = Register()

_OBJECTIVE_MODULES = {
    "cm": "lightx2v_train.trainers.consistency.cm",
    "mean_flow": "lightx2v_train.trainers.consistency.mean_flow",
    "meanflow": "lightx2v_train.trainers.consistency.mean_flow",
    "pcm": "lightx2v_train.trainers.consistency.pcm",
    "scm": "lightx2v_train.trainers.consistency.scm",
    "tcm": "lightx2v_train.trainers.consistency.tcm",
}


def build_consistency_objective(config, path):
    consistency_config = config["training"].get("consistency", {})
    if not isinstance(consistency_config, dict):
        raise ValueError("training.consistency must be a mapping.")
    algorithm = str(consistency_config.get("algorithm", "cm")).lower()
    if algorithm not in CONSISTENCY_OBJECTIVE_REGISTER:
        module_name = _OBJECTIVE_MODULES.get(algorithm)
        if module_name is not None:
            importlib.import_module(module_name)
    if algorithm not in CONSISTENCY_OBJECTIVE_REGISTER:
        available = ", ".join(sorted(CONSISTENCY_OBJECTIVE_REGISTER.keys()))
        raise ValueError(f"Unknown consistency algorithm {algorithm!r}. Available algorithms: {available}")
    return CONSISTENCY_OBJECTIVE_REGISTER[algorithm](config, path)
