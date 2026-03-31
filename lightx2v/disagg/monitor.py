import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import zmq

logger = logging.getLogger(__name__)


@dataclass
class ReporterConfig:
    service_type: str
    gpu_id: int
    bind_address: str


class Reporter:
    def __init__(self, service_type: str, gpu_id: int, bind_address: str):
        self.config = ReporterConfig(
            service_type=service_type,
            gpu_id=gpu_id,
            bind_address=bind_address,
        )
        self._context = zmq.Context.instance()
        self._stop_event = threading.Event()
        self._metrics_lock = threading.Lock()
        self._extra_metrics_provider: Optional[Callable[[], Dict[str, Any]]] = None

    def set_extra_metrics_provider(self, provider: Optional[Callable[[], Dict[str, Any]]]):
        with self._metrics_lock:
            self._extra_metrics_provider = provider

    def _query_gpu_metrics(self) -> Dict[str, Any]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            str(self.config.gpu_id),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        if not out:
            raise RuntimeError("nvidia-smi returned empty output")

        util_str, mem_used_str, mem_total_str = [x.strip() for x in out.split(",")]
        return {
            "gpu_utilization": float(util_str),
            "gpu_memory_used_mb": float(mem_used_str),
            "gpu_memory_total_mb": float(mem_total_str),
        }

    def get_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "service_type": self.config.service_type,
            "gpu_id": self.config.gpu_id,
            "timestamp": time.time(),
        }
        try:
            metrics.update(self._query_gpu_metrics())
            metrics["status"] = "ok"

            with self._metrics_lock:
                provider = self._extra_metrics_provider
            if provider is not None:
                extra_metrics = provider()
                if isinstance(extra_metrics, dict):
                    metrics.update(extra_metrics)
        except Exception as exc:
            metrics["status"] = "error"
            metrics["error"] = str(exc)
        return metrics

    def serve_forever(self):
        socket = self._context.socket(zmq.REP)
        socket.linger = 0
        socket.bind(self.config.bind_address)
        logger.info("Reporter started: service=%s gpu=%s bind=%s", self.config.service_type, self.config.gpu_id, self.config.bind_address)

        try:
            while not self._stop_event.is_set():
                if socket.poll(timeout=500) == 0:
                    continue
                try:
                    req = socket.recv_json()
                except Exception:
                    socket.send_json({"status": "error", "error": "invalid request"})
                    continue

                cmd = req.get("cmd", "metrics") if isinstance(req, dict) else "metrics"
                if cmd == "metrics":
                    socket.send_json(self.get_metrics())
                else:
                    socket.send_json({"status": "error", "error": f"unsupported cmd: {cmd}"})
        finally:
            socket.close()

    def stop(self):
        self._stop_event.set()


class Monitor:
    def __init__(self, nodes: List[str], request_timeout_ms: int = 1000):
        self.nodes = nodes
        self.request_timeout_ms = request_timeout_ms
        self._context = zmq.Context.instance()

    def _poll_one(self, address: str) -> Dict[str, Any]:
        socket = self._context.socket(zmq.REQ)
        socket.linger = 0
        socket.rcvtimeo = self.request_timeout_ms
        socket.sndtimeo = self.request_timeout_ms
        socket.connect(address)
        try:
            socket.send_json({"cmd": "metrics"})
            resp = socket.recv_json()
            if not isinstance(resp, dict):
                return {
                    "status": "error",
                    "address": address,
                    "error": "invalid response type",
                }
            result = dict(resp)
            result["address"] = address
            return result
        except Exception as exc:
            return {
                "status": "error",
                "address": address,
                "error": str(exc),
            }
        finally:
            socket.close()

    def poll_once(self) -> List[Dict[str, Any]]:
        return [self._poll_one(address) for address in self.nodes]

    def run_forever(
        self,
        interval_seconds: float = 5.0,
        callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            results = self.poll_once()
            if callback is not None:
                callback(results)
            time.sleep(interval_seconds)
