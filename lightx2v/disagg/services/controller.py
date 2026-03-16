from collections import deque
from threading import Lock
from typing import Any, Deque
import time

from lightx2v.disagg.conn import REQUEST_POLLING_PORT, ReqManager
from lightx2v.disagg.services.base import BaseService


class ControllerService(BaseService):
	def __init__(self):
		super().__init__()
		self.request_queue: Deque[Any] = deque()
		self._lock = Lock()
		self.req_mgr = ReqManager()

	def add_request(self, config):
		"""Add request config into internal request queue and dispatch it to services."""
		if config is None:
			raise ValueError("config cannot be None")

		with self._lock:
			self.request_queue.append(config)

		bootstrap_addr = config.get("data_bootstrap_addr", "127.0.0.1")
		self.req_mgr.send(bootstrap_addr, REQUEST_POLLING_PORT + 0, config)
		self.req_mgr.send(bootstrap_addr, REQUEST_POLLING_PORT + 1, config)
		self.req_mgr.send(bootstrap_addr, REQUEST_POLLING_PORT + 2, config)
		self.logger.info("Request added to controller queue and dispatched to services")

		time.sleep(10)  # Sleep briefly to allow services to process the request