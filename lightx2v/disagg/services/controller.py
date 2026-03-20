import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any, Deque

from lightx2v.disagg.conn import REQUEST_POLLING_PORT, ReqManager
from lightx2v.disagg.scheduler.round_robin import RoundRobinPolicy
from lightx2v.disagg.services.base import BaseService


class ControllerService(BaseService):
    def __init__(self):
        super().__init__()
        self.request_queue: Deque[Any] = deque()
        self.encoder_policy = RoundRobinPolicy()
        self.transformer_policy = RoundRobinPolicy()
        self.decoder_policy = RoundRobinPolicy()
        self._lock = Lock()
        self.req_mgr = ReqManager()

    def add_instance(self, instance_type: str, instance_address: str):
        """Add instance address to the matching scheduling policy by type."""
        if not instance_address:
            raise ValueError("instance_address cannot be empty")

        if instance_type == "encoder":
            self.encoder_policy.add_instance(instance_address)
        elif instance_type == "transformer":
            self.transformer_policy.add_instance(instance_address)
        elif instance_type == "decoder":
            self.decoder_policy.add_instance(instance_address)
        else:
            raise ValueError("instance_type must be one of: encoder, transformer, decoder")

    def remove_instance(self, instance_type: str, instance_address: str):
        """Remove instance address from the matching scheduling policy by type."""
        if not instance_address:
            raise ValueError("instance_address cannot be empty")

        if instance_type == "encoder":
            self.encoder_policy.remove_instance(instance_address)
        elif instance_type == "transformer":
            self.transformer_policy.remove_instance(instance_address)
        elif instance_type == "decoder":
            self.decoder_policy.remove_instance(instance_address)
        else:
            raise ValueError("instance_type must be one of: encoder, transformer, decoder")

    def send_request(self, config):
        """Dispatch request config to services."""
        if config is None:
            raise ValueError("config cannot be None")

        encoder_addr = self.encoder_policy.schedule()
        transformer_addr = self.transformer_policy.schedule()
        decoder_addr = self.decoder_policy.schedule()

        encoder_ip, encoder_port_str = encoder_addr.rsplit(":", 1)
        transformer_ip, transformer_port_str = transformer_addr.rsplit(":", 1)
        decoder_ip, decoder_port_str = decoder_addr.rsplit(":", 1)

        self.req_mgr.send(encoder_ip, int(encoder_port_str), config)
        self.req_mgr.send(transformer_ip, int(transformer_port_str), config)
        self.req_mgr.send(decoder_ip, int(decoder_port_str), config)
        self.logger.info("Request added to controller queue and dispatched to services")

    def run(self, config):
        """Initialize instances from config and submit request multiple times."""
        if config is None:
            raise ValueError("config cannot be None")

        bootstrap_addr = config.get("data_bootstrap_addr", "127.0.0.1")
        encoder_engine_rank = config.get("encoder_engine_rank", 0)
        transformer_engine_rank = config.get("transformer_engine_rank", 1)
        decoder_engine_rank = config.get("decoder_engine_rank", 2)

        self.encoder_policy = RoundRobinPolicy()
        self.transformer_policy = RoundRobinPolicy()
        self.decoder_policy = RoundRobinPolicy()

        self.add_instance("encoder", f"{bootstrap_addr}:{REQUEST_POLLING_PORT + encoder_engine_rank}")
        self.add_instance(
            "transformer",
            f"{bootstrap_addr}:{REQUEST_POLLING_PORT + transformer_engine_rank}",
        )
        self.add_instance("decoder", f"{bootstrap_addr}:{REQUEST_POLLING_PORT + decoder_engine_rank}")

        base_save_path = config.get("save_path")

        for i in range(2):
            request_config = dict(config)
            request_config["data_bootstrap_room"] = i
            if base_save_path:
                save_path = Path(base_save_path)
                request_config["save_path"] = str(save_path.with_name(f"{save_path.stem}{i + 1}{save_path.suffix}"))
            # TODO: use queue to receive request from client and dispatch, currently we just send the same request multiple times for testing
            with self._lock:
                self.request_queue.append(request_config)
                current_request = self.request_queue.popleft()
            self.send_request(current_request)

            time.sleep(2)  # Sleep briefly to allow services to process the request
