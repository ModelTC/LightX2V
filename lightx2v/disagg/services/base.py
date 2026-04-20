import logging
from abc import ABC

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseService(ABC):
    def __init__(self):
        """
        Base initialization for all services.
        """
        self.logger = logger
        self.logger.info(f"Initializing {self.__class__.__name__}")

    def _sync_runtime_config(self, config):
        current_config = getattr(self, "config", None)
        if current_config is None:
            self.config = dict(config)
            return self.config

        if current_config is not config:
            current_config.clear()
            current_config.update(config)

        self.config = current_config
        return self.config
