import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

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
        self._instance_lock = Lock()
        self._free_gpus: set[int] = set()
        self._managed_instances: dict[str, dict[str, Any]] = {}
        self.started_instances: list[tuple[str, str]] = []
        self._runtime_config: dict[str, Any] | None = None
        self._bootstrap_addr: str = "127.0.0.1"
        self._gpu_reuse_block_until: dict[int, float] = {}
        self._gpu_reuse_grace_seconds: float = 5.0
        self._shutting_down: bool = False

    def _is_tcp_port_open(self, host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex((host, port)) == 0

    def _wait_for_tcp_port_state(self, host: str, port: int, should_be_open: bool, timeout_seconds: float) -> bool:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            is_open = self._is_tcp_port_open(host, port)
            if is_open == should_be_open:
                return True
            time.sleep(0.1)
        return self._is_tcp_port_open(host, port) == should_be_open

    def _to_plain(self, value: Any) -> Any:
        """Recursively convert config containers (e.g. LockableDict) to built-in Python types."""
        if isinstance(value, Mapping):
            return {k: self._to_plain(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_plain(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._to_plain(v) for v in value)
        if isinstance(value, set):
            return {self._to_plain(v) for v in value}
        return value

    def _resolve_service_config_json(self, config_json: str, instance_type: str) -> str:
        config_path = Path(config_json)
        if config_path.is_file():
            if config_path.name.endswith("_controller.json"):
                candidate = config_path.with_name(config_path.name.replace("_controller.json", f"_{instance_type}.json"))
                if candidate.is_file():
                    return str(candidate)
            if config_path.name.endswith("_distill_controller.json"):
                candidate = config_path.with_name(config_path.name.replace("_distill_controller.json", f"_distill_{instance_type}.json"))
                if candidate.is_file():
                    return str(candidate)
        return config_json

    def _monitor_node_from_instance_address(self, instance_address: str) -> str:
        host, port_str = instance_address.rsplit(":", 1)
        rank = int(port_str) - REQUEST_POLLING_PORT
        return f"tcp://{host}:{MONITOR_POLLING_PORT + rank}"

    def _instance_address_from_monitor_node(self, monitor_node: str) -> str:
        host_port = monitor_node
        if host_port.startswith("tcp://"):
            host_port = host_port[len("tcp://") :]
        host, port_str = host_port.rsplit(":", 1)
        rank = int(port_str) - MONITOR_POLLING_PORT
        return f"{host}:{REQUEST_POLLING_PORT + rank}"

    def _init_gpu_pool(self, config: dict):
        disagg_cfg = config.get("disagg_config") if isinstance(config.get("disagg_config"), dict) else {}
        total_ranks = int(config.get("ranks", disagg_cfg.get("ranks", 8)))
        if total_ranks <= 0:
            raise ValueError("ranks must be positive")

        self._free_gpus = set(range(total_ranks))

    def create_instance(self, instance_type: str) -> str:
        """Create one service instance on an idle GPU and add it to scheduling pool."""
        if instance_type not in {"encoder", "transformer", "decoder"}:
            raise ValueError("instance_type must be one of: encoder, transformer, decoder")
        if self._runtime_config is None:
            raise RuntimeError("controller runtime config is not initialized")

        with self._instance_lock:
            if not self._free_gpus:
                raise RuntimeError("no idle GPU available")

            now = time.time()
            gpu_id: int | None = None
            for candidate_gpu in sorted(self._free_gpus):
                if now < self._gpu_reuse_block_until.get(candidate_gpu, 0.0):
                    continue

                monitor_port = MONITOR_POLLING_PORT + candidate_gpu
                if self._is_tcp_port_open(self._bootstrap_addr, monitor_port):
                    self.logger.warning(
                        "Skip gpu=%s for %s creation because monitor port %s is still in use",
                        candidate_gpu,
                        instance_type,
                        monitor_port,
                    )
                    continue

                gpu_id = candidate_gpu
                break

            if gpu_id is None:
                raise RuntimeError(f"no idle GPU available for {instance_type}: all candidates cooling down or port is in use")

            instance_cfg = self._to_plain(self._runtime_config)
            instance_cfg["disagg_mode"] = instance_type
            if instance_type == "encoder":
                instance_cfg["encoder_engine_rank"] = gpu_id
            elif instance_type == "transformer":
                instance_cfg["transformer_engine_rank"] = gpu_id
            else:
                instance_cfg["decoder_engine_rank"] = gpu_id

            model_path = instance_cfg.get("model_path")
            config_json = instance_cfg.get("config_json")
            if not model_path or not config_json:
                raise RuntimeError("model_path and config_json are required to launch service subprocess")
            service_config_json = self._resolve_service_config_json(str(config_json), instance_type)

            cmd = [
                sys.executable,
                "-m",
                "lightx2v.disagg.examples.run_service",
                "--service",
                instance_type,
                "--engine_rank",
                str(gpu_id),
                "--model_cls",
                str(instance_cfg.get("model_cls", "wan2.1")),
                "--task",
                str(instance_cfg.get("task", "t2v")),
                "--model_path",
                str(model_path),
                "--config_json",
                service_config_json,
                "--seed",
                str(instance_cfg.get("seed", 42)),
                "--prompt",
                str(instance_cfg.get("prompt", "")),
                "--negative_prompt",
                str(instance_cfg.get("negative_prompt", "")),
                "--save_result_path",
                str(instance_cfg.get("save_path", "")),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            process = subprocess.Popen(cmd, env=env)

            monitor_port = MONITOR_POLLING_PORT + gpu_id
            if not self._wait_for_tcp_port_state(self._bootstrap_addr, monitor_port, should_be_open=True, timeout_seconds=8.0):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                raise RuntimeError(f"service {instance_type} on gpu={gpu_id} failed to expose monitor port {monitor_port}")

            instance_address = f"{self._bootstrap_addr}:{REQUEST_POLLING_PORT + gpu_id}"
            self._free_gpus.remove(gpu_id)
            self.add_instance(instance_type, instance_address)
            monitor_node = f"tcp://{self._bootstrap_addr}:{MONITOR_POLLING_PORT + gpu_id}"
            if monitor_node not in self.monitor.nodes:
                self.monitor.nodes.append(monitor_node)
            self._managed_instances[instance_address] = {
                "instance_type": instance_type,
                "gpu_id": gpu_id,
                "process": process,
            }
            self.started_instances.append((instance_type, instance_address))
            self.logger.info(
                "Created %s instance on gpu=%s pid=%s address=%s",
                instance_type,
                gpu_id,
                process.pid,
                instance_address,
            )
            return instance_address

    def reclaim_instance(self, instance_type: str, instance_address: str | None = None) -> str:
        """Reclaim one managed instance and return its GPU back to idle pool."""
        if instance_type not in {"encoder", "transformer", "decoder"}:
            raise ValueError("instance_type must be one of: encoder, transformer, decoder")

        with self._instance_lock:
            target_address = instance_address
            if target_address is None:
                candidates = [addr for addr, meta in self._managed_instances.items() if meta.get("instance_type") == instance_type]
                if not candidates:
                    raise RuntimeError(f"no managed {instance_type} instance to reclaim")
                target_address = candidates[-1]

            meta = self._managed_instances.get(target_address)
            if meta is None:
                raise RuntimeError(f"instance not managed by controller: {target_address}")
            if meta.get("instance_type") != instance_type:
                raise RuntimeError(f"instance type mismatch for {target_address}: expected={instance_type} got={meta.get('instance_type')}")

            process = meta.get("process")
            gpu_id = int(meta.get("gpu_id"))

            self.remove_instance(instance_type, target_address)
            monitor_node = self._monitor_node_from_instance_address(target_address)
            if monitor_node in self.monitor.nodes:
                self.monitor.nodes.remove(monitor_node)

            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)

            monitor_port = MONITOR_POLLING_PORT + gpu_id
            if not self._wait_for_tcp_port_state(self._bootstrap_addr, monitor_port, should_be_open=False, timeout_seconds=5.0):
                self.logger.warning(
                    "Monitor port still open after reclaim: service=%s gpu=%s port=%s",
                    instance_type,
                    gpu_id,
                    monitor_port,
                )

            self._free_gpus.add(gpu_id)
            self._gpu_reuse_block_until[gpu_id] = time.time() + self._gpu_reuse_grace_seconds
            self._managed_instances.pop(target_address, None)
            if (instance_type, target_address) in self.started_instances:
                self.started_instances.remove((instance_type, target_address))
            self.logger.info(
                "Reclaimed %s instance from gpu=%s address=%s",
                instance_type,
                gpu_id,
                target_address,
            )
            return target_address

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

    def _build_latency_summary(self, result: dict[str, Any], controller_recv_ts: float) -> dict[str, float] | None:
        request_metrics = result.get("request_metrics")
        if not isinstance(request_metrics, dict):
            return None

        stages = request_metrics.get("stages")
        if not isinstance(stages, dict):
            return None

        def _as_float(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _stage(name: str) -> dict[str, Any]:
            stage_metrics = stages.get(name)
            return stage_metrics if isinstance(stage_metrics, dict) else {}

        controller_send_ts = _as_float(request_metrics.get("controller_send_ts"))
        encoder = _stage("encoder")
        transformer = _stage("transformer")
        decoder = _stage("decoder")

        encoder_recv_ts = _as_float(encoder.get("request_received_ts"))
        encoder_compute_start_ts = _as_float(encoder.get("compute_start_ts"))
        encoder_compute_end_ts = _as_float(encoder.get("compute_end_ts"))
        encoder_output_enqueued_ts = _as_float(encoder.get("output_enqueued_ts"))

        transformer_recv_ts = _as_float(transformer.get("request_received_ts"))
        transformer_compute_start_ts = _as_float(transformer.get("compute_start_ts"))
        transformer_compute_end_ts = _as_float(transformer.get("compute_end_ts"))
        transformer_output_enqueued_ts = _as_float(transformer.get("output_enqueued_ts"))

        decoder_recv_ts = _as_float(decoder.get("request_received_ts"))
        decoder_compute_start_ts = _as_float(decoder.get("compute_start_ts"))
        decoder_compute_end_ts = _as_float(decoder.get("compute_end_ts"))
        decoder_output_enqueued_ts = _as_float(decoder.get("output_enqueued_ts"))

        required_values = [
            controller_send_ts,
            encoder_recv_ts,
            encoder_compute_start_ts,
            encoder_compute_end_ts,
            encoder_output_enqueued_ts,
            transformer_recv_ts,
            transformer_compute_start_ts,
            transformer_compute_end_ts,
            transformer_output_enqueued_ts,
            decoder_recv_ts,
            decoder_compute_start_ts,
            decoder_compute_end_ts,
            decoder_output_enqueued_ts,
        ]
        if any(value is None for value in required_values):
            return None

        summary: dict[str, float] = {
            "controller_to_encoder_comm_delay_s": encoder_recv_ts - controller_send_ts,
            "encoder_scheduling_delay_s": encoder_compute_start_ts - encoder_recv_ts,
            "encoder_compute_delay_s": encoder_compute_end_ts - encoder_compute_start_ts,
            "encoder_communication_delay_s": transformer_recv_ts - encoder_output_enqueued_ts,
            "transformer_scheduling_delay_s": transformer_compute_start_ts - transformer_recv_ts,
            "transformer_compute_delay_s": transformer_compute_end_ts - transformer_compute_start_ts,
            "transformer_communication_delay_s": decoder_recv_ts - transformer_output_enqueued_ts,
            "decoder_scheduling_delay_s": decoder_compute_start_ts - decoder_recv_ts,
            "decoder_compute_delay_s": decoder_compute_end_ts - decoder_compute_start_ts,
            "decoder_communication_delay_s": controller_recv_ts - decoder_output_enqueued_ts,
            "end_to_end_delay_s": controller_recv_ts - controller_send_ts,
        }
        summary["sum_of_components_s"] = sum(value for key, value in summary.items() if key != "end_to_end_delay_s" and key != "sum_of_components_s")
        return summary

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
        """Initialize controller buffers, stream request configs from workload, then wait for all callbacks."""
        if config is None:
            raise ValueError("config cannot be None")

        self._shutting_down = False

        bootstrap_addr = config.get("data_bootstrap_addr", "127.0.0.1")
        request_ingress_port = int(config.get("controller_request_port", os.getenv("DISAGG_CONTROLLER_REQUEST_PORT", REQUEST_POLLING_PORT - 2)))
        result_port = int(config.get("controller_result_port", REQUEST_POLLING_PORT - 1))
        self._bootstrap_addr = str(bootstrap_addr)
        self._runtime_config = self._to_plain(config)
        self._init_gpu_pool(config)

        self.encoder_policy = RoundRobinPolicy()
        self.transformer_policy = RoundRobinPolicy()
        self.decoder_policy = RoundRobinPolicy()

        self._init_request_rdma_buffer(bootstrap_addr, config)
        
        time.sleep(5.0)

        for instance_type in ("encoder", "transformer", "decoder"):
            address = self.create_instance(instance_type)
        for _ in range(5):
            self.create_instance("transformer")

        monitor_stop_event = Event()
        scale_out_threshold = 80.0
        scale_in_threshold = 20.0
        scale_cooldown_seconds = 30.0
        last_scale_ts: dict[str, float] = {
            "encoder": 0.0,
            "transformer": 0.0,
            "decoder": 0.0,
        }

        def _monitor_callback(results):
            return
            if self._shutting_down:
                return

            service_metrics: dict[str, list[dict[str, Any]]] = {
                "encoder": [],
                "transformer": [],
                "decoder": [],
            }

            for item in results:
                # self.logger.info("monitor: %s", item)
                if not isinstance(item, dict):
                    continue

                service_type = str(item.get("service_type", ""))
                if service_type not in {"encoder", "transformer", "decoder"}:
                    continue
                
                # if service_type not in {"transformer"}:
                #     continue

                if item.get("status") != "ok":
                    continue

                try:
                    gpu_utilization = float(item.get("gpu_utilization", 0.0))
                except (TypeError, ValueError):
                    continue

                monitor_address = str(item.get("address", ""))
                if not monitor_address:
                    continue

                queue_total_pending = item.get("queue_total_pending", None)
                try:
                    queue_total_pending_int = int(queue_total_pending) if queue_total_pending is not None else -1
                except (TypeError, ValueError):
                    queue_total_pending_int = -1

                all_queues_empty = bool(item.get("all_queues_empty", False))

                service_metrics[service_type].append(
                    {
                        "gpu_utilization": gpu_utilization,
                        "monitor_address": monitor_address,
                        "queue_total_pending": queue_total_pending_int,
                        "all_queues_empty": all_queues_empty,
                    }
                )

            for service_type, metrics in service_metrics.items():
                if not metrics:
                    continue

                now = time.time()
                avg_gpu_utilization = sum(float(metric["gpu_utilization"]) for metric in metrics) / len(metrics)

                if avg_gpu_utilization > scale_out_threshold and now - last_scale_ts[service_type] >= scale_cooldown_seconds:
                    try:
                        new_address = self.create_instance(service_type)
                        last_scale_ts[service_type] = now
                        self.logger.info(
                            "Auto-scale out triggered: service=%s avg_gpu_utilization=%.2f new_instance=%s",
                            service_type,
                            avg_gpu_utilization,
                            new_address,
                        )
                    except Exception as exc:
                        pass
                        # self.logger.warning(
                        #     "Auto-scale out skipped for service=%s avg_gpu_utilization=%.2f reason=%s",
                        #     service_type,
                        #     avg_gpu_utilization,
                        #     exc,
                        # )

                low_metric = min(metrics, key=lambda metric: float(metric["gpu_utilization"]))
                low_utilization = float(low_metric["gpu_utilization"])
                low_monitor_address = str(low_metric["monitor_address"])
                with self._instance_lock:
                    service_instance_count = sum(1 for meta in self._managed_instances.values() if meta.get("instance_type") == service_type)

                queues_empty_for_service = bool(low_metric.get("all_queues_empty", False)) and int(low_metric.get("queue_total_pending", -1)) == 0

                if low_utilization < scale_in_threshold and service_instance_count > 1 and queues_empty_for_service and now - last_scale_ts[service_type] >= scale_cooldown_seconds:
                    try:
                        target_instance_address = self._instance_address_from_monitor_node(low_monitor_address)
                        self.reclaim_instance(service_type, target_instance_address)
                        last_scale_ts[service_type] = now
                        self.logger.info(
                            "Auto-scale in triggered: service=%s low_gpu_utilization=%.2f reclaimed_instance=%s",
                            service_type,
                            low_utilization,
                            target_instance_address,
                        )
                    except Exception as exc:
                        self.logger.warning(
                            "Auto-scale in skipped for service=%s low_gpu_utilization=%.2f reason=%s",
                            service_type,
                            low_utilization,
                            exc,
                        )

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
        monitor_thread.start()
        
        time.sleep(5.0)

        base_save_path = config.get("save_path")
        expected_rooms: set[int] = set()
        received_rooms: set[int] = set()
        received_results: list[dict] = []
        next_room = 0
        batch_request_start_ts: float | None = None

        def _handle_decoder_result(result: Any):
            if not isinstance(result, dict):
                self.logger.warning("Ignored non-dict decoder result: %s", result)
                return
            room = result.get("data_bootstrap_room")
            if room is None:
                self.logger.warning("Ignored decoder result without data_bootstrap_room: %s", result)
                return
            room = int(room)
            if room not in expected_rooms:
                self.logger.warning("Ignored decoder result for unexpected room=%s: %s", room, result)
                return
            if room in received_rooms:
                self.logger.info("Duplicate decoder result for room=%s ignored", room)
                return

            controller_recv_ts = time.time()
            latency_summary = self._build_latency_summary(result, controller_recv_ts)
            if latency_summary is not None:
                result["latency_summary"] = latency_summary
                self.logger.info("Latency summary room=%s metrics=%s", room, latency_summary)

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

        def _drain_decoder_results_non_block():
            while True:
                result = self.req_mgr.receive_non_block(result_port)
                if result is None:
                    break
                _handle_decoder_result(result)

        try:
            self.logger.info("Waiting workload configs on port=%s", request_ingress_port)
            while True:
                workload_config = self.req_mgr.receive(request_ingress_port)
                if not isinstance(workload_config, dict):
                    self.logger.warning("Ignored invalid workload config packet: %s", workload_config)
                    continue

                if workload_config.get("workload_end") or workload_config.get("end") or workload_config.get("stop"):
                    self.logger.info("Received workload end signal, stop accepting new configs.")
                    break

                request_config = dict(config)
                request_config.update(self._to_plain(workload_config))

                room = request_config.get("data_bootstrap_room", next_room)
                try:
                    room = int(room)
                except (TypeError, ValueError):
                    room = next_room
                if room in expected_rooms:
                    while next_room in expected_rooms:
                        next_room += 1
                    room = next_room
                next_room = max(next_room, room + 1)

                request_config["data_bootstrap_room"] = room
                request_config["controller_result_host"] = bootstrap_addr
                request_config["controller_result_port"] = result_port

                metrics = request_config.get("request_metrics")
                if not isinstance(metrics, dict):
                    metrics = {}
                metrics["request_id"] = int(metrics.get("request_id", room))
                metrics["controller_send_ts"] = time.time()
                if not isinstance(metrics.get("stages"), dict):
                    metrics["stages"] = {}
                request_config["request_metrics"] = metrics

                if base_save_path and not request_config.get("save_path"):
                    save_path = Path(base_save_path)
                    request_config["save_path"] = str(save_path.with_name(f"{save_path.stem}{room}{save_path.suffix}"))

                with self._lock:
                    current_request = request_config

                if batch_request_start_ts is None:
                    batch_request_start_ts = time.time()

                self.send_request(current_request)
                self.logger.info(
                    "Dispatched request room=%s save_path=%s",
                    room,
                    request_config.get("save_path"),
                )
                expected_rooms.add(room)

                _drain_decoder_results_non_block()

            self.logger.info(
                "Waiting for decoder results: expected=%s on port=%s",
                sorted(expected_rooms),
                result_port,
            )
            while len(received_rooms) < len(expected_rooms):
                result = self.req_mgr.receive(result_port)
                _handle_decoder_result(result)

            self.logger.info("All decoder results received. Controller exiting.")
            if batch_request_start_ts is None:
                batch_request_start_ts = time.time()
            batch_total_time_s = time.time() - batch_request_start_ts
            self.logger.info(
                "Batch total elapsed time: requests=%s completed=%s total_time_s=%.3f",
                len(expected_rooms),
                len(received_rooms),
                batch_total_time_s,
            )
        finally:
            self._shutting_down = True
            monitor_stop_event.set()
            monitor_thread.join(timeout=2.0)

            for instance_type, address in reversed(list(self.started_instances)):
                try:
                    self.reclaim_instance(instance_type, address)
                except Exception:
                    self.logger.exception("Failed to reclaim %s instance address=%s", instance_type, address)
