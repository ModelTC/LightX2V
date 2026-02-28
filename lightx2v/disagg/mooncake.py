import json
import logging
import os
import uuid
import socket
import struct
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

@dataclass
class MooncakeTransferEngineConfig:
    local_hostname: str
    metadata_server: str
    protocol: str
    device_name: str

    @staticmethod
    def from_file(file_path: str) -> "MooncakeTransferEngineConfig":
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeTransferEngineConfig(
            local_hostname=config.get("local_hostname", None),
            metadata_server=config.get("metadata_server"),
            protocol=config.get("protocol", "rdma"),
            device_name=config.get("device_name", ""),
        )

    @staticmethod
    def load_from_env() -> "MooncakeTransferEngineConfig":
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH", "/root/zht/LightX2V/configs/mooncake_config.json")
        if config_file_path is None:
            raise ValueError("The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeTransferEngineConfig.from_file(config_file_path)


class MooncakeTransferEngine:
    def __init__(self):
        self.engine = None
        try:
            from mooncake.engine import TransferEngine
            self.engine = TransferEngine()
        except ImportError as e:
            logger.warning(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
                "to run with MooncakeTransferEngine."
            )
            # We allow continuing without engine for non-transfer operations or testing structure
        
        try:
            self.config = MooncakeTransferEngineConfig.load_from_env()
            logger.info("Mooncake Configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Mooncake config: {e}")
            raise

        # session_suffix = "_" + str(uuid.uuid4())
        self.initialize(
            self.config.local_hostname,
            self.config.metadata_server,
            self.config.protocol,
            self.config.device_name,
        )
        # session_suffix = ":" + self.engine.get_rpc_port()
        # self.session_id = self.config.local_hostname + session_suffix
        self.session_id = f"{self.config.local_hostname}:{self.engine.get_rpc_port()}"

    def register(self, ptr, length):
        if self.engine:
            ret = self.engine.register_memory(ptr, length)
            if ret != 0:
                logger.error("Mooncake memory registration failed.")
                raise RuntimeError("Mooncake memory registration failed.")

    def deregister(self, ptr):
        if self.engine:
            ret = self.engine.unregister_memory(ptr)
            if ret != 0:
                logger.error("Mooncake memory deregistration failed.")
                raise RuntimeError("Mooncake memory deregistration failed.")

    def initialize(
        self,
        local_hostname: str,
        metadata_server: str,
        protocol: str,
        device_name: str,
    ) -> None:
        """Initialize the mooncake instance."""
        if self.engine:
            self.engine.initialize(local_hostname, metadata_server, protocol, device_name)

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously transfer data to the specified address."""
        if self.engine:
            ret = self.engine.transfer_sync_write(
                session_id, buffer, peer_buffer_address, length
            )
            if ret < 0:
                logger.error("Transfer Return Error")
                raise Exception("Transfer Return Error")
            return ret
        return -1

    def get_localhost(self):
        return self.config.local_hostname

    def get_session_id(self):
        return self.session_id
