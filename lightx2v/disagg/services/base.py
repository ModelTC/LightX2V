import logging
import torch
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseService(ABC):
    def __init__(self, config):
        """
        Base initialization for all services.
        Args:
            config: A dictionary or object containing configuration parameters.
        """
        self.config = config
        self.logger = logger
        self.logger.info(f"Initializing {self.__class__.__name__} with config: {config}")
        
    @abstractmethod
    def load_models(self):
        """
        Abstract method to load necessary models.
        """
        pass
