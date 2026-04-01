import time
from pathlib import Path
from threading import Event, Lock, Thread

from lightx2v.disagg.conn import MONITOR_POLLING_PORT, REQUEST_POLLING_PORT, ReqManager
from lightx2v.disagg.monitor import Monitor
from lightx2v.disagg.rdma_buffer import RDMABuffer
from lightx2v.disagg.rdma_server import RDMAServer
from lightx2v.disagg.scheduler.round_robin import RoundRobinPolicy
from lightx2v.disagg.services.base import BaseService


class ControllerService(BaseService):
    def __init__(self):
        super().__init__()
        self.rdma_buffer_request: RDMABuffer | None = None
        self.rdma_buffer_phase1: RDMABuffer | None = None
        self.rdma_buffer_phase2: RDMABuffer | None = None
        self.encoder_policy = RoundRobinPolicy()
        self.transformer_policy = RoundRobinPolicy()
        self.decoder_policy = RoundRobinPolicy()
        self._lock = Lock()
        self.req_mgr = ReqManager()
        self.monitor = Monitor(nodes=[])
        self._rdma_server_request: RDMAServer | None = None
        self._rdma_server_phase1: RDMAServer | None = None
        self._rdma_server_phase2: RDMAServer | None = None
        self._rdma_handshake_thread_request: Thread | None = None
        self._rdma_handshake_thread_phase1: Thread | None = None
        self._rdma_handshake_thread_phase2: Thread | None = None

    def _init_request_rdma_buffer(self, bootstrap_addr: str, config: dict):
        slots = int(config.get("rdma_buffer_slots", "128"))
        slot_size = int(config.get("rdma_buffer_slot_size", "4096"))
        handshake_port = int(config.get("rdma_request_handshake_port", "5566"))
        phase1_slots = slots
        phase1_slot_size = slot_size
        phase1_handshake_port = int(config.get("rdma_phase1_handshake_port", "5567"))
        phase2_slots = slots
        phase2_slot_size = slot_size
        phase2_handshake_port = int(config.get("rdma_phase2_handshake_port", "5568"))

        # Normalize RDMA request-buffer parameters so downstream services consume the same values.
        config["rdma_request_host"] = bootstrap_addr
        config["rdma_buffer_slots"] = slots
        config["rdma_buffer_slot_size"] = slot_size
        config["rdma_request_handshake_port"] = handshake_port
        config["rdma_phase1_host"] = bootstrap_addr
        config["rdma_phase1_handshake_port"] = phase1_handshake_port
        config["rdma_phase2_host"] = bootstrap_addr
        config["rdma_phase2_handshake_port"] = phase2_handshake_port

        need_bytes = 16 + slots * slot_size
        self._rdma_server_request = RDMAServer(buffer_size=need_bytes)
        self.rdma_buffer_request = RDMABuffer(
            role="server",
            buffer_size=slots,
            slot_size=slot_size,
            rdma_server=self._rdma_server_request,
        )

        self._rdma_handshake_thread_request = Thread(
            target=self._rdma_server_request.handshake,
            kwargs={"host": bootstrap_addr, "port": handshake_port},
            name="controller-rdma-handshake",
            daemon=True,
        )
        self._rdma_handshake_thread_request.start()

        need_bytes_phase1 = 16 + phase1_slots * phase1_slot_size
        self._rdma_server_phase1 = RDMAServer(buffer_size=need_bytes_phase1)
        self.rdma_buffer_phase1 = RDMABuffer(
            role="server",
            buffer_size=phase1_slots,
            slot_size=phase1_slot_size,
            rdma_server=self._rdma_server_phase1,
        )
        self._rdma_handshake_thread_phase1 = Thread(
            target=self._rdma_server_phase1.handshake,
            kwargs={"host": bootstrap_addr, "port": phase1_handshake_port},
            name="controller-rdma-handshake-phase1",
            daemon=True,
        )
        self._rdma_handshake_thread_phase1.start()

        need_bytes_phase2 = 16 + phase2_slots * phase2_slot_size
        self._rdma_server_phase2 = RDMAServer(buffer_size=need_bytes_phase2)
        self.rdma_buffer_phase2 = RDMABuffer(
            role="server",
            buffer_size=phase2_slots,
            slot_size=phase2_slot_size,
            rdma_server=self._rdma_server_phase2,
        )
        self._rdma_handshake_thread_phase2 = Thread(
            target=self._rdma_server_phase2.handshake,
            kwargs={"host": bootstrap_addr, "port": phase2_handshake_port},
            name="controller-rdma-handshake-phase2",
            daemon=True,
        )
        self._rdma_handshake_thread_phase2.start()
        self.logger.info(
            "Initialized RDMA buffers: request=(%s,%s,%s) phase1=(%s,%s,%s) phase2=(%s,%s,%s)",
            slots,
            slot_size,
            need_bytes,
            phase1_slots,
            phase1_slot_size,
            need_bytes_phase1,
            phase2_slots,
            phase2_slot_size,
            need_bytes_phase2,
        )

    def serve_rdma_dispatch_only(self, config: dict) -> None:
        """Expose request + phase1 + phase2 RDMA meta rings, then block.

        For Qwen/Wan HTTP encoder + pull-based transformer/decoder workers that do not
        use the encoder ``request`` ring; rings must stay up for handshake.
        """
        if config is None:
            raise ValueError("config cannot be None")
        dc = config.get("disagg_config", {})
        bootstrap_addr = config.get("data_bootstrap_addr", dc.get("bootstrap_addr", "127.0.0.1"))
        self._init_request_rdma_buffer(bootstrap_addr, config)
        self.logger.info("RDMA dispatch rings ready on %s (Ctrl+C to exit).", bootstrap_addr)
        try:
            while True:
                time.sleep(3600.0)
        except KeyboardInterrupt:
            self.logger.info("Controller serve_rdma_dispatch_only interrupted, exiting.")

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

        if self.rdma_buffer_request is None:
            raise RuntimeError("RDMA request buffer is not initialized")
        self.rdma_buffer_request.produce(config)
        self.logger.info("Request enqueued to encoder request RDMA buffer")

    def run(self, config):
        """Initialize instances, send requests, wait for decoder save_path callbacks, then exit."""
        if config is None:
            raise ValueError("config cannot be None")

        bootstrap_addr = config.get("data_bootstrap_addr", "127.0.0.1")
        encoder_engine_rank = config.get("encoder_engine_rank", 0)
        transformer_engine_rank = config.get("transformer_engine_rank", 1)
        decoder_engine_rank = config.get("decoder_engine_rank", 2)
        request_count = int(config.get("request_count", 2))
        result_port = int(config.get("controller_result_port", REQUEST_POLLING_PORT - 1))

        self.encoder_policy = RoundRobinPolicy()
        self.transformer_policy = RoundRobinPolicy()
        self.decoder_policy = RoundRobinPolicy()

        self._init_request_rdma_buffer(bootstrap_addr, config)

        self.add_instance("encoder", f"{bootstrap_addr}:{REQUEST_POLLING_PORT + encoder_engine_rank}")
        self.add_instance(
            "transformer",
            f"{bootstrap_addr}:{REQUEST_POLLING_PORT + transformer_engine_rank}",
        )
        self.add_instance("decoder", f"{bootstrap_addr}:{REQUEST_POLLING_PORT + decoder_engine_rank}")

        monitor_nodes = [
            f"tcp://{bootstrap_addr}:{MONITOR_POLLING_PORT + encoder_engine_rank}",
            f"tcp://{bootstrap_addr}:{MONITOR_POLLING_PORT + transformer_engine_rank}",
            f"tcp://{bootstrap_addr}:{MONITOR_POLLING_PORT + decoder_engine_rank}",
        ]
        self.monitor.nodes = monitor_nodes

        monitor_stop_event = Event()

        def _monitor_callback(results):
            for item in results:
                self.logger.info("monitor: %s", item)

        monitor_thread = Thread(
            target=self.monitor.run_forever,
            kwargs={
                "interval_seconds": 5.0,
                "callback": _monitor_callback,
                "stop_event": monitor_stop_event,
            },
            name="controller-monitor",
            daemon=True,
        )
        # monitor_thread.start()

        base_save_path = config.get("save_path")
        expected_rooms: set[int] = set()
        received_rooms: set[int] = set()
        received_results: list[dict] = []
        try:
            for i in range(request_count):
                request_config = dict(config)
                request_config["data_bootstrap_room"] = i
                request_config["controller_result_host"] = bootstrap_addr
                request_config["controller_result_port"] = result_port
                if base_save_path:
                    save_path = Path(base_save_path)
                    request_config["save_path"] = str(save_path.with_name(f"{save_path.stem}{i + 1}{save_path.suffix}"))
                # TODO: use queue to receive request from client and dispatch, currently we just send the same request multiple times for testing
                with self._lock:
                    current_request = request_config
                self.send_request(current_request)

                expected_rooms.add(i)

            self.logger.info(
                "Waiting for decoder results: expected=%s on port=%s",
                sorted(expected_rooms),
                result_port,
            )
            while len(received_rooms) < len(expected_rooms):
                result = self.req_mgr.receive(result_port)
                if not isinstance(result, dict):
                    self.logger.warning("Ignored non-dict decoder result: %s", result)
                    continue
                room = result.get("data_bootstrap_room")
                if room is None:
                    self.logger.warning("Ignored decoder result without data_bootstrap_room: %s", result)
                    continue
                room = int(room)
                if room not in expected_rooms:
                    self.logger.warning("Ignored decoder result for unexpected room=%s: %s", room, result)
                    continue
                if room in received_rooms:
                    self.logger.info("Duplicate decoder result for room=%s ignored", room)
                    continue

                received_rooms.add(room)
                received_results.append(result)

                if result.get("ok", False):
                    self.logger.info(
                        "Decoder result received room=%s save_path=%s (%s/%s)",
                        room,
                        result.get("save_path"),
                        len(received_rooms),
                        len(expected_rooms),
                    )
                else:
                    self.logger.error(
                        "Decoder result failed room=%s error=%s (%s/%s)",
                        room,
                        result.get("error"),
                        len(received_rooms),
                        len(expected_rooms),
                    )

            self.logger.info("All decoder results received. Controller exiting.")
        finally:
            pass
            # monitor_stop_event.set()
            # monitor_thread.join(timeout=1.0)