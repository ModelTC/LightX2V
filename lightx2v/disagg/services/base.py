import logging
from abc import ABC, abstractmethod

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
