from abc import ABC
import sys

from loguru import logger as loguru_logger


loguru_logger.remove()
loguru_logger.add(
    sys.stderr,
    level="INFO",
    format="[{level}] {time:DD MMM YYYY HH:mm:ss} | {name}:{function}:{line} - {message}",
)


class _LoguruLoggerAdapter:
    def __init__(self, logger):
        self._logger = logger

    @staticmethod
    def _format_message(message, args):
        if not args:
            return message
        try:
            return message % args
        except Exception:
            return f"{message} {' '.join(map(str, args))}"

    def debug(self, message, *args, **kwargs):
        self._logger.opt(depth=1).debug(self._format_message(message, args))

    def info(self, message, *args, **kwargs):
        self._logger.opt(depth=1).info(self._format_message(message, args))

    def warning(self, message, *args, **kwargs):
        self._logger.opt(depth=1).warning(self._format_message(message, args))

    def error(self, message, *args, **kwargs):
        self._logger.opt(depth=1).error(self._format_message(message, args))

    def critical(self, message, *args, **kwargs):
        self._logger.opt(depth=1).critical(self._format_message(message, args))

    def exception(self, message, *args, **kwargs):
        self._logger.opt(depth=1, exception=True).error(self._format_message(message, args))

    def log(self, level, message, *args, **kwargs):
        self._logger.opt(depth=1).log(level, self._format_message(message, args))

    def bind(self, **kwargs):
        return _LoguruLoggerAdapter(self._logger.bind(**kwargs))

    def opt(self, *args, **kwargs):
        return self._logger.opt(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._logger, item)


logger = _LoguruLoggerAdapter(loguru_logger)


class BaseService(ABC):
    def __init__(self):
        """
        Base initialization for all services.
        """
        self.logger = logger
        self.logger.info("Initializing %s", self.__class__.__name__)

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
