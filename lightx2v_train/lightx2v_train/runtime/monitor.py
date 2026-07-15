import atexit
from copy import deepcopy

from loguru import logger

from lightx2v_train.runtime.distributed import is_main_process


def build_monitor(config):
    swanlab_config = config.get("logging", {}).get("swanlab", {})
    if not swanlab_config.get("enable", False):
        return NoopMonitor()
    return SwanLabMonitor(config, swanlab_config)


class NoopMonitor:
    def log_metrics(self, metrics, step=None):
        return

    def finish(self):
        return


class SwanLabMonitor:
    def __init__(self, config, swanlab_config):
        self._swanlab = None
        self._enabled = is_main_process()
        if not self._enabled:
            return

        try:
            import swanlab
        except ImportError as exc:
            raise RuntimeError("SwanLab is enabled but not installed. Install 'swanlab' or set logging.swanlab.enable=false.") from exc

        api_key = swanlab_config.get("api_key")
        if not api_key:
            raise ValueError("logging.swanlab.api_key must be set when logging.swanlab.enable=true.")
        swanlab.login(api_key=api_key)

        init_kwargs = {}
        if swanlab_config.get("project") is not None:
            init_kwargs["project"] = swanlab_config["project"]
        if swanlab_config.get("name") is not None:
            init_kwargs["experiment_name"] = swanlab_config["name"]
        init_kwargs["config"] = self._config_without_secrets(config)

        self._swanlab = swanlab
        self._swanlab.init(**init_kwargs)
        atexit.register(self.finish)
        logger.info("[monitor] SwanLab enabled")

    @staticmethod
    def _config_without_secrets(config):
        safe_config = deepcopy(config)
        safe_config.get("logging", {}).get("swanlab", {}).pop("api_key", None)
        return safe_config

    def log_metrics(self, metrics, step=None):
        if not self._enabled:
            return
        values = {}
        for key, value in metrics.items():
            if hasattr(value, "item"):
                value = value.item()
            values[key] = value
        if values:
            self._swanlab.log(values, step=step)

    def finish(self):
        if not self._enabled:
            return
        self._swanlab.finish()
        self._enabled = False
