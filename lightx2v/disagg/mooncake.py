import json
import logging
import os
import random
import socket
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _detect_non_loopback_ipv4() -> str | None:
    # Use a UDP connect trick to discover the outbound interface IP without sending traffic.
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    try:
        host_ip = socket.gethostbyname(socket.gethostname())
        if host_ip and not host_ip.startswith("127."):
            return host_ip
    except Exception:
        pass

    return None


def _collect_local_ipv4_addresses() -> list[str]:
    candidates: list[str] = []

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            address = info[4][0]
            if address and not address.startswith("127.") and address not in candidates:
                candidates.append(address)
    except Exception:
        pass

    detected = _detect_non_loopback_ipv4()
    if detected is not None and detected not in candidates:
        candidates.append(detected)

    return candidates


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
        cfg = MooncakeTransferEngineConfig.from_file(config_file_path)
        local_ipv4s = _collect_local_ipv4_addresses()

        env_metadata_server = os.getenv("MOONCAKE_METADATA_SERVER", "").strip()
        if env_metadata_server:
            cfg.metadata_server = env_metadata_server

        env_protocol = os.getenv("MOONCAKE_PROTOCOL", "").strip()
        if env_protocol:
            cfg.protocol = env_protocol

        env_device_name = os.getenv("MOONCAKE_DEVICE_NAME", "").strip()
        if env_device_name:
            cfg.device_name = env_device_name

        # Keep session IDs and metadata endpoints stable on single-node runs.
        # localhost may resolve to IPv6 on some hosts while peers use IPv4.
        force_ipv4 = os.getenv("MOONCAKE_FORCE_IPV4_LOOPBACK", "1") not in ("0", "false", "False")
        env_host = os.getenv("MOONCAKE_LOCAL_HOSTNAME", "").strip()
        if env_host:
            if env_host in ("localhost", "::1", "127.0.0.1") or env_host in local_ipv4s:
                cfg.local_hostname = env_host
            else:
                detected = _detect_non_loopback_ipv4()
                if detected is not None:
                    logger.warning(
                        "Ignoring MOONCAKE_LOCAL_HOSTNAME=%s because it does not match this host (local_ipv4s=%s); using %s",
                        env_host,
                        local_ipv4s,
                        detected,
                    )
                    cfg.local_hostname = detected
                elif force_ipv4:
                    cfg.local_hostname = "127.0.0.1"
        elif force_ipv4 and cfg.local_hostname in ("localhost", "::1", "127.0.0.1"):
            detected = _detect_non_loopback_ipv4()
            if detected is not None:
                cfg.local_hostname = detected
            else:
                cfg.local_hostname = "127.0.0.1"
        elif cfg.local_hostname not in local_ipv4s and cfg.local_hostname not in ("localhost", "::1", "127.0.0.1"):
            detected = _detect_non_loopback_ipv4()
            if detected is not None:
                logger.warning(
                    "Auto-correcting Mooncake local_hostname from %s to %s on this host (local_ipv4s=%s)",
                    cfg.local_hostname,
                    detected,
                    local_ipv4s,
                )
                cfg.local_hostname = detected
        return cfg


class MooncakeTransferEngine:
    def __init__(self):
        self.engine = None
        try:
            from mooncake.engine import TransferEngine

            self.engine = TransferEngine()
        except ImportError as e:
            logger.warning(
                "Please install mooncake by following the instructions at https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/getting_started/build.md to run with MooncakeTransferEngine."
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
                raise RuntimeError("Mooncake memory registration failed.")

    def deregister(self, ptr):
        if self.engine:
            ret = self.engine.unregister_memory(ptr)
            if ret != 0:
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

    def transfer_sync(self, session_id: str, buffer: int, peer_buffer_address: int, length: int) -> int:
        """Synchronously transfer data to the specified address."""
        if self.engine:
            if os.getenv("NETWORK_LATENCY"):
                latency_prob_raw = os.getenv("NETWORK_LATENCY_PROB", "0.02")
                latency_sec_raw = os.getenv("NETWORK_LATENCY_SEC", "5")
                try:
                    latency_prob = float(latency_prob_raw)
                except ValueError:
                    latency_prob = 0.02
                # Accept either ratio (0.02) or percentage (2 / 5).
                if latency_prob > 1.0:
                    latency_prob = latency_prob / 100.0
                latency_prob = max(0.0, min(1.0, latency_prob))

                try:
                    latency_sec = float(latency_sec_raw)
                except ValueError:
                    latency_sec = 5.0
                latency_sec = max(0.0, latency_sec)

                if random.random() < latency_prob:
                    logger.warning(
                        "Simulated network latency: sleeping %.3fs before transfer_sync_write (prob=%.4f)",
                        latency_sec,
                        latency_prob,
                    )
                    time.sleep(latency_sec)

            retry_count = int(os.getenv("MOONCAKE_TRANSFER_RETRY", "5"))
            retry_backoff_s = float(os.getenv("MOONCAKE_TRANSFER_RETRY_BACKOFF_S", "0.05"))
            for attempt in range(retry_count + 1):
                ret = self.engine.transfer_sync_write(session_id, buffer, peer_buffer_address, length)
                if ret >= 0:
                    return ret

                logger.warning(
                    "Transfer Return Error attempt=%s/%s session=%s src=0x%x dst=0x%x len=%s",
                    attempt + 1,
                    retry_count + 1,
                    session_id,
                    int(buffer),
                    int(peer_buffer_address),
                    int(length),
                )
                if attempt < retry_count:
                    time.sleep(retry_backoff_s)

            logger.error("Transfer Return Error after retries")
            return -1
        return -1

    def get_localhost(self):
        return self.config.local_hostname

    def get_session_id(self):
        return self.session_id
