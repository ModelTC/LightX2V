import ipaddress
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import shlex
import signal
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import zmq

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
        self._graceful_reclaim_timeout_seconds: float = float(os.getenv("DISAGG_RECLAIM_GRACEFUL_TIMEOUT_SECONDS", "30.0"))
        self._force_kill_wait_seconds: float = float(os.getenv("DISAGG_RECLAIM_FORCE_KILL_WAIT_SECONDS", "1.0"))
        self._instance_start_timeout_seconds: float = float(os.getenv("DISAGG_INSTANCE_START_TIMEOUT_SECONDS", "90.0"))
        self._sidecar_start_timeout_seconds: float = float(os.getenv("DISAGG_SIDECAR_START_TIMEOUT_SECONDS", "15.0"))
        self._sidecar_drain_idle_seconds: float = float(os.getenv("DISAGG_SIDECAR_DRAIN_IDLE_SECONDS", "1.0"))
        # <= 0 means wait indefinitely until sidecar pending queues are drained.
        self._sidecar_drain_timeout_seconds: float = float(os.getenv("DISAGG_SIDECAR_DRAIN_TIMEOUT_SECONDS", "0"))
        self._remote_proxy_start_timeout_seconds: float = float(os.getenv("DISAGG_REMOTE_PROXY_START_TIMEOUT_SECONDS", "20.0"))
        self._sidecar_reclaim_threads: list[Thread] = []
        self._shutting_down: bool = False
        self._enable_monitor: bool = False
        self._static_instance_slots: list[dict[str, Any]] = []
        self._free_slot_ids: set[int] = set()
        self._slot_reuse_block_until: dict[int, float] = {}
        self._local_host_aliases: set[str] = set()
        self._request_metrics_by_room: dict[int, dict[str, Any]] = {}

    def _is_monitor_enabled(self) -> bool:
        raw = os.getenv("ENABLE_MONITOR")
        if raw is None:
            return False
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _is_centralized_enabled(self) -> bool:
        raw = os.getenv("IS_CENTRALIZED")
        if raw is None:
            return False
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

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

    def _refresh_local_host_aliases(self):
        aliases: set[str] = {
            "127.0.0.1",
            "localhost",
            str(self._bootstrap_addr),
        }
        try:
            hostname = socket.gethostname()
            aliases.add(hostname)
            aliases.add(socket.getfqdn())
            host_info = socket.gethostbyname_ex(hostname)
            aliases.update(host_info[1])
            aliases.update(host_info[2])
        except Exception:
            pass
        self._local_host_aliases = {item.strip() for item in aliases if isinstance(item, str) and item.strip()}

    def _is_local_host(self, host: str) -> bool:
        normalized = str(host).strip()
        if not normalized:
            return False
        if normalized in self._local_host_aliases:
            return True
        try:
            return socket.gethostbyname(normalized) in self._local_host_aliases
        except Exception:
            return False

    def _ensure_rdma_preferred_ipv4_env(self, host: str, env: dict[str, str]) -> None:
        """So RoCE gid_index matches the data-plane IP on each worker (multi-node)."""
        if env.get("RDMA_PREFERRED_IPV4"):
            return
        h = str(host).strip()
        if not h:
            return
        try:
            env["RDMA_PREFERRED_IPV4"] = str(ipaddress.IPv4Address(h))
        except Exception:
            pass

    def _allocate_free_tcp_port(self, bind_host: str | None = None) -> int:
        host = str(bind_host or self._bootstrap_addr)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])

    def _build_service_command(self, instance_type: str, engine_rank: int, instance_cfg: dict[str, Any], service_config_json: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "lightx2v.disagg.examples.run_service",
            "--service",
            instance_type,
            "--engine_rank",
            str(engine_rank),
            "--model_cls",
            str(instance_cfg.get("model_cls", "wan2.1")),
            "--task",
            str(instance_cfg.get("task", "t2v")),
            "--model_path",
            str(instance_cfg.get("model_path")),
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

    def _maybe_wrap_service_command_with_nsys(
        self,
        *,
        host: str,
        instance_type: str,
        engine_rank: int,
        instance_cfg: dict[str, Any],
        command: list[str],
    ) -> list[str]:
        if not self._is_truthy(os.getenv("DISAGG_ENABLE_NSYS"), default=False):
            return command

        if not self._is_local_host(host):
            self.logger.info(
                "Skip nsys profiling for remote %s instance host=%s rank=%s",
                instance_type,
                host,
                engine_rank,
            )
            return command

        nsys_bin = shutil.which(os.getenv("DISAGG_NSYS_BIN", "nsys"))
        if nsys_bin is None:
            self.logger.warning("DISAGG_ENABLE_NSYS is set but nsys is not available, skip profiling for %s rank=%s", instance_type, engine_rank)
            return command

        output_dir_raw = os.getenv("DISAGG_NSYS_OUTPUT_DIR")
        if output_dir_raw:
            output_dir = Path(output_dir_raw)
        else:
            base_save_path = instance_cfg.get("save_path") or (self._runtime_config or {}).get("save_path") or str(Path(__file__).resolve().parents[3] / "save_results" / "wan22_i2v_dynamic.mp4")
            output_dir = Path(str(base_save_path)).parent / "nsys"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_name = f"{instance_type}_rank{engine_rank}"
        trace = os.getenv("DISAGG_NSYS_TRACE", "cuda,nvtx,osrt")
        extra_args = shlex.split(os.getenv("DISAGG_NSYS_EXTRA_ARGS", ""))

        profiled_command = [
            nsys_bin,
            "profile",
            "--force-overwrite=true",
            "--trace",
            trace,
            "-o",
            str(output_dir / output_name),
        ]
        profiled_command.extend(extra_args)
        profiled_command.extend(command)
        return profiled_command

    def _merge_request_metrics(self, existing: dict[str, Any] | None, update: dict[str, Any] | None) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if isinstance(existing, dict):
            merged.update(existing)
        if not isinstance(update, dict):
            return merged

        for key, value in update.items():
            if key != "stages" or not isinstance(value, dict):
                merged[key] = value
                continue

            merged_stages: dict[str, Any] = {}
            existing_stages = merged.get("stages")
            if isinstance(existing_stages, dict):
                for stage_name, stage_metrics in existing_stages.items():
                    merged_stages[stage_name] = dict(stage_metrics) if isinstance(stage_metrics, dict) else stage_metrics

            for stage_name, stage_metrics in value.items():
                if not isinstance(stage_metrics, dict):
                    continue
                base_stage_metrics = merged_stages.get(stage_name)
                if isinstance(base_stage_metrics, dict):
                    combined_stage_metrics = dict(base_stage_metrics)
                    combined_stage_metrics.update(stage_metrics)
                else:
                    combined_stage_metrics = dict(stage_metrics)
                merged_stages[stage_name] = combined_stage_metrics

            merged["stages"] = merged_stages

        return merged

    def _query_zmq(self, req_addr: str, payload: dict[str, Any], timeout_ms: int = 1000) -> dict[str, Any] | None:
        context = zmq.Context()
        req = context.socket(zmq.REQ)
        req.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        req.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        req.connect(req_addr)
        try:
            req.send_pyobj(payload)
            reply = req.recv_pyobj()
            if isinstance(reply, dict):
                return reply
            return None
        except Exception:
            return None
        finally:
            req.close(0)
            context.term()

    def _query_sidecar(self, req_addr: str, cmd: str) -> dict[str, Any] | None:
        return self._query_zmq(req_addr, {"cmd": str(cmd)}, timeout_ms=1000)

    def _run_centralized_ok_server(self, stop_event: Event, bind_host: str, bind_port: int):
        controller = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/ok":
                    self.send_response(404)
                    self.end_headers()
                    return

                content_length = int(self.headers.get("Content-Length", "0") or "0")
                raw_body = self.rfile.read(content_length) if content_length > 0 else b""
                try:
                    message = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                except Exception:
                    self.send_response(400)
                    self.end_headers()
                    return

                controller.logger.info(
                    "Received centralized OK control message: stage=%s room=%s",
                    message.get("stage_name"),
                    message.get("data_bootstrap_room"),
                )
                response = json.dumps({"ok": True, "control": "OK"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer((bind_host, bind_port), _Handler)
        server.timeout = 0.2
        try:
            while not stop_event.is_set():
                server.handle_request()
        finally:
            server.server_close()

    def _is_truthy(self, value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _remote_proxy_req_addr(self, slot: dict[str, Any]) -> str:
        host = str(slot["host"])
        proxy_req_port = int(slot["proxy_req_port"])
        return f"tcp://{host}:{proxy_req_port}"

    def _ensure_remote_instance_proxy(self, slot: dict[str, Any]):
        if not self._is_truthy(slot.get("use_remote_proxy", False)):
            return

        req_addr = self._remote_proxy_req_addr(slot)
        reply = self._query_zmq(req_addr, {"cmd": "ping"}, timeout_ms=800)
        if isinstance(reply, dict) and reply.get("ok", False):
            return

        python_executable = str(slot.get("python_executable", sys.executable))
        workdir = str(slot.get("workdir", Path(__file__).resolve().parents[3]))
        log_dir = str(slot.get("log_dir", "/tmp/lightx2v_disagg"))
        activate_cmd = str(slot.get("activate_cmd", "")).strip()
        proxy_req_port = int(slot["proxy_req_port"])
        proxy_log_path = str(slot.get("proxy_log_path", f"{log_dir}/instance_proxy.log"))

        script_lines = [
            "set -e",
            f"mkdir -p {shlex.quote(log_dir)}",
            f"cd {shlex.quote(workdir)}",
        ]
        if activate_cmd:
            script_lines.append(activate_cmd)
        script_lines.extend(
            [
                (
                    "nohup env PYTHONUNBUFFERED=1 "
                    f"{shlex.quote(python_executable)} -m lightx2v.disagg.services.instance_proxy "
                    f"--bind-addr {shlex.quote(f'tcp://0.0.0.0:{proxy_req_port}')} "
                    f"--workdir {shlex.quote(workdir)} --log-dir {shlex.quote(log_dir)} "
                    f"> {shlex.quote(proxy_log_path)} 2>&1 &"
                ),
                "echo PROXY_PID=$!",
            ]
        )
        script = "\n".join(script_lines)

        self._run_ssh_script(slot, script, timeout_seconds=30.0, check=True)

        deadline = time.time() + self._remote_proxy_start_timeout_seconds
        while time.time() < deadline:
            probe = self._query_zmq(req_addr, {"cmd": "ping"}, timeout_ms=800)
            if isinstance(probe, dict) and probe.get("ok", False):
                self.logger.info("Remote instance proxy is ready on host=%s req_addr=%s", slot.get("host"), req_addr)
                return
            time.sleep(0.2)

        raise RuntimeError(f"remote instance proxy failed to start on host={slot.get('host')} req_addr={req_addr}")

    def _start_sidecar_process(self, instance_type: str, gpu_id: str | int, bind_host: str | None = None) -> dict[str, Any]:
        host = str(bind_host or self._bootstrap_addr)
        push_port = self._allocate_free_tcp_port(host)
        req_port = self._allocate_free_tcp_port(host)
        push_addr = f"tcp://{host}:{push_port}"
        req_addr = f"tcp://{host}:{req_port}"

        cmd = [
            sys.executable,
            "-m",
            "lightx2v.disagg.services.data_mgr_sidecar",
            "--push-addr",
            push_addr,
            "--req-addr",
            req_addr,
        ]
        sidecar_env = os.environ.copy()
        sidecar_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        process = subprocess.Popen(
            cmd,
            env=sidecar_env,
            start_new_session=True,
        )

        deadline = time.time() + self._sidecar_start_timeout_seconds
        ready = False
        while time.time() < deadline:
            reply = self._query_sidecar(req_addr, "ping")
            if isinstance(reply, dict) and reply.get("ok", False):
                ready = True
                break
            time.sleep(0.1)

        if not ready:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise RuntimeError(f"sidecar server failed to start for {instance_type} gpu={gpu_id}")

        self.logger.info(
            "Started sidecar for %s gpu=%s pid=%s push=%s req=%s",
            instance_type,
            gpu_id,
            process.pid,
            push_addr,
            req_addr,
        )
        return {
            "process": process,
            "push_addr": push_addr,
            "req_addr": req_addr,
        }

    def _run_ssh_script(self, slot: dict[str, Any], script: str, timeout_seconds: float = 30.0, check: bool = True) -> subprocess.CompletedProcess:
        ssh_bin = str(slot.get("ssh_bin", "ssh"))
        ssh_target = str(slot.get("ssh_target", slot.get("host", ""))).strip()
        if not ssh_target:
            raise RuntimeError("remote slot missing ssh target")

        ssh_options = slot.get("ssh_options")
        ssh_cmd = [ssh_bin]
        if isinstance(ssh_options, list):
            ssh_cmd.extend(str(opt) for opt in ssh_options if str(opt).strip())
        ssh_cmd.extend([ssh_target, "bash", "-lc", script])
        return subprocess.run(
            ssh_cmd,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )

    def _launch_remote_instance(self, slot: dict[str, Any], instance_type: str, cmd: list[str], cuda_device: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._is_truthy(slot.get("use_remote_proxy", False)):
            return self._launch_remote_instance_via_proxy(slot, instance_type, cmd, cuda_device)

        host = str(slot["host"])
        engine_rank = int(slot["engine_rank"])
        python_executable = str(slot.get("python_executable", sys.executable))
        workdir = str(slot.get("workdir", Path(__file__).resolve().parents[3]))
        log_dir = str(slot.get("log_dir", "/tmp/lightx2v_disagg"))
        activate_cmd = str(slot.get("activate_cmd", "")).strip()
        push_port = int(slot["sidecar_push_port"])
        req_port = int(slot["sidecar_req_port"])
        push_addr = f"tcp://{host}:{push_port}"
        req_addr = f"tcp://{host}:{req_port}"
        service_log = f"{log_dir}/{instance_type}_{engine_rank}_service.log"
        sidecar_log = f"{log_dir}/{instance_type}_{engine_rank}_sidecar.log"

        extra_env = slot.get("env")
        normalized_env: dict[str, str] = {}
        if isinstance(extra_env, dict):
            for key, value in extra_env.items():
                normalized_env[str(key)] = str(value)
        if self._is_centralized_enabled():
            normalized_env["IS_CENTRALIZED"] = "1"
        if os.getenv("SYNC_COMM") is not None:
            normalized_env["SYNC_COMM"] = str(os.getenv("SYNC_COMM", "0"))
        self._ensure_rdma_preferred_ipv4_env(host, normalized_env)

        sidecar_env_vars = {
            **normalized_env,
            "CUDA_VISIBLE_DEVICES": str(cuda_device),
            "PYTHONUNBUFFERED": "1",
        }
        service_env_vars = {
            **normalized_env,
            "CUDA_VISIBLE_DEVICES": str(cuda_device),
            "LIGHTX2V_SIDECAR_PUSH_ADDR": push_addr,
            "LIGHTX2V_SIDECAR_REQ_ADDR": req_addr,
            "PYTHONUNBUFFERED": "1",
        }

        def _to_env_prefix(env_map: dict[str, str]) -> str:
            return " ".join(f"{key}={shlex.quote(value)}" for key, value in env_map.items())

        def _with_env(base_cmd: str, env_map: dict[str, str]) -> str:
            env_prefix = _to_env_prefix(env_map)
            if not env_prefix:
                return base_cmd
            return f"env {env_prefix} {base_cmd}"

        sidecar_cmd = _with_env(
            (
                f"{shlex.quote(python_executable)} "
                "-m lightx2v.disagg.services.data_mgr_sidecar "
                f"--push-addr {shlex.quote(push_addr)} --req-addr {shlex.quote(req_addr)}"
            ),
            sidecar_env_vars,
        )
        cmd_with_python = [python_executable, *cmd[1:]]
        service_cmd = _with_env(" ".join(shlex.quote(str(part)) for part in cmd_with_python), service_env_vars)

        script_lines = [
            "set -e",
            f"mkdir -p {shlex.quote(log_dir)}",
            f"cd {shlex.quote(workdir)}",
        ]
        if activate_cmd:
            script_lines.append(activate_cmd)
        script_lines.extend(
            [
                f"nohup {sidecar_cmd} > {shlex.quote(sidecar_log)} 2>&1 &",
                "sidecar_pid=$!",
                "sleep 0.5",
                f"nohup {service_cmd} > {shlex.quote(service_log)} 2>&1 &",
                "service_pid=$!",
                "echo SIDECAR_PID=$sidecar_pid",
                "echo SERVICE_PID=$service_pid",
            ]
        )
        script = "\n".join(script_lines)

        completed = self._run_ssh_script(slot, script, timeout_seconds=45.0, check=True)
        sidecar_pid: int | None = None
        service_pid: int | None = None
        for line in completed.stdout.splitlines():
            if line.startswith("SIDECAR_PID="):
                try:
                    sidecar_pid = int(line.split("=", 1)[1].strip())
                except ValueError:
                    sidecar_pid = None
            elif line.startswith("SERVICE_PID="):
                try:
                    service_pid = int(line.split("=", 1)[1].strip())
                except ValueError:
                    service_pid = None

        if sidecar_pid is None or service_pid is None:
            raise RuntimeError(
                f"failed to parse remote pids for {instance_type} rank={engine_rank} host={host}: stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )

        sidecar_meta = {
            "mode": "remote",
            "host": host,
            "req_addr": req_addr,
            "push_addr": push_addr,
            "pid": sidecar_pid,
            "log_path": sidecar_log,
        }
        process_meta = {
            "mode": "remote",
            "host": host,
            "pid": service_pid,
            "log_path": service_log,
        }
        return process_meta, sidecar_meta

    def _launch_remote_instance_via_proxy(self, slot: dict[str, Any], instance_type: str, cmd: list[str], cuda_device: str) -> tuple[dict[str, Any], dict[str, Any]]:
        self._ensure_remote_instance_proxy(slot)

        host = str(slot["host"])
        engine_rank = int(slot["engine_rank"])
        python_executable = str(slot.get("python_executable", sys.executable))
        workdir = str(slot.get("workdir", Path(__file__).resolve().parents[3]))
        log_dir = str(slot.get("log_dir", "/tmp/lightx2v_disagg"))
        push_port = int(slot["sidecar_push_port"])
        req_port = int(slot["sidecar_req_port"])
        push_addr = f"tcp://{host}:{push_port}"
        req_addr = f"tcp://{host}:{req_port}"
        service_log = f"{log_dir}/{instance_type}_{engine_rank}_service.log"
        sidecar_log = f"{log_dir}/{instance_type}_{engine_rank}_sidecar.log"

        extra_env = slot.get("env")
        normalized_env: dict[str, str] = {}
        if isinstance(extra_env, dict):
            for key, value in extra_env.items():
                normalized_env[str(key)] = str(value)
        self._ensure_rdma_preferred_ipv4_env(host, normalized_env)

        proxy_req_addr = self._remote_proxy_req_addr(slot)
        payload = {
            "cmd": "start_instance",
            "instance_type": str(instance_type),
            "engine_rank": int(engine_rank),
            "cuda_device": str(cuda_device),
            "python_executable": python_executable,
            "service_argv": [str(part) for part in cmd[1:]],
            "sidecar_push_addr": push_addr,
            "sidecar_req_addr": req_addr,
            "service_log_path": service_log,
            "sidecar_log_path": sidecar_log,
            "workdir": workdir,
            "log_dir": log_dir,
            "env": normalized_env,
        }
        reply = self._query_zmq(proxy_req_addr, payload, timeout_ms=10000)
        if not isinstance(reply, dict) or not reply.get("ok", False):
            raise RuntimeError(f"remote proxy failed to start instance on host={host}: {reply}")

        data = reply.get("data") if isinstance(reply.get("data"), dict) else {}
        sidecar_pid = int(data.get("sidecar_pid", 0) or 0)
        service_pid = int(data.get("service_pid", 0) or 0)
        if sidecar_pid <= 0 or service_pid <= 0:
            raise RuntimeError(f"remote proxy returned invalid pids for host={host}: {reply}")

        sidecar_meta = {
            "mode": "remote",
            "host": host,
            "req_addr": req_addr,
            "push_addr": push_addr,
            "pid": sidecar_pid,
            "log_path": sidecar_log,
            "proxy_req_addr": proxy_req_addr,
        }
        process_meta = {
            "mode": "remote",
            "host": host,
            "pid": service_pid,
            "log_path": service_log,
            "proxy_req_addr": proxy_req_addr,
        }
        return process_meta, sidecar_meta

    def _stop_remote_pid(self, slot: dict[str, Any], pid: int, graceful_timeout_seconds: float):
        if self._is_truthy(slot.get("use_remote_proxy", False)):
            req_addr = self._remote_proxy_req_addr(slot)
            timeout_seconds = max(1, int(graceful_timeout_seconds))
            payload = {
                "cmd": "stop_pid",
                "pid": int(pid),
                "timeout_seconds": timeout_seconds,
            }
            reply = self._query_zmq(req_addr, payload, timeout_ms=(timeout_seconds + 3) * 1000)
            if isinstance(reply, dict) and reply.get("ok", False):
                return
            self.logger.warning(
                "Remote proxy stop_pid failed, falling back to ssh kill: host=%s pid=%s reply=%s",
                slot.get("host"),
                pid,
                reply,
            )

        timeout_seconds = max(1, int(graceful_timeout_seconds))
        script = "\n".join(
            [
                "set +e",
                f"pid={int(pid)}",
                "if kill -0 ${pid} >/dev/null 2>&1; then",
                "  kill -TERM ${pid} >/dev/null 2>&1 || true",
                f"  deadline=$((SECONDS+{timeout_seconds}))",
                "  while kill -0 ${pid} >/dev/null 2>&1; do",
                "    if (( SECONDS >= deadline )); then",
                "      kill -KILL ${pid} >/dev/null 2>&1 || true",
                "      break",
                "    fi",
                "    sleep 0.2",
                "  done",
                "fi",
            ]
        )
        try:
            self._run_ssh_script(slot, script, timeout_seconds=float(timeout_seconds + 10), check=False)
        except Exception as exc:
            self.logger.warning("Failed to stop remote pid=%s on host=%s: %s", pid, slot.get("host"), exc)

    def _reclaim_sidecar_when_drained(self, instance_type: str, target_address: str, sidecar_meta: dict[str, Any]):
        req_addr = str(sidecar_meta.get("req_addr", ""))
        process = sidecar_meta.get("process")
        if not req_addr or process is None:
            return

        deadline = None
        if self._sidecar_drain_timeout_seconds > 0:
            deadline = time.time() + self._sidecar_drain_timeout_seconds

        while True:
            if process.poll() is not None:
                # Sidecar already exited.
                break

            reply = self._query_sidecar(req_addr, "get_stats")
            if isinstance(reply, dict) and reply.get("ok", False):
                data = reply.get("data") if isinstance(reply.get("data"), dict) else {}
                last_message_ts = float(data.get("last_message_ts", 0.0))
                idle_seconds = max(0.0, time.time() - last_message_ts)
                pending_input_watch = int(data.get("input_watch", 0))
                pending_output_watch = int(data.get("output_watch", 0))
                pending_transformer_request = int(data.get("transformer_request_pool", 0))
                pending_transformer_waiting = int(data.get("transformer_waiting_pool", 0))
                pending_transformer_active = int(data.get("transformer_active_rooms", 0))
                pending_active = pending_input_watch + pending_output_watch + pending_transformer_request + pending_transformer_waiting + pending_transformer_active

                if pending_active == 0 and idle_seconds >= self._sidecar_drain_idle_seconds:
                    break

            if deadline is not None and time.time() >= deadline:
                self.logger.warning(
                    "Sidecar drain timeout reached for %s address=%s, forcing shutdown",
                    instance_type,
                    target_address,
                )
                break

            time.sleep(0.2)

        try:
            self._query_sidecar(req_addr, "shutdown")
        except Exception:
            pass

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()

        self.logger.info(
            "Reclaimed sidecar for %s address=%s",
            instance_type,
            target_address,
        )

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

    def _load_warmup_duration_seconds(self, config: Mapping[str, Any]) -> float:
        stage_json = os.getenv("DISAGG_WORKLOAD_STAGES_JSON", "")
        if not stage_json:
            stage_json = str(config.get("workload_stages_json", "") or "").strip()

        if stage_json:
            stage_file = Path(stage_json)
        else:
            repo_root = Path(__file__).resolve().parents[3]
            stage_file = repo_root / "configs" / "disagg" / "wan22_i2v_workload_stages.json"

        if not stage_file.is_file():
            self.logger.warning("workload stages config not found, skip warmup scale guard: %s", stage_file)
            return 0.0

        try:
            with stage_file.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except Exception as exc:
            self.logger.warning("failed to load workload stages config %s: %s", stage_file, exc)
            return 0.0

        if not isinstance(loaded, list):
            self.logger.warning("invalid workload stages config format (expect list): %s", stage_file)
            return 0.0

        warmup_duration_s = 0.0
        for raw_stage in loaded:
            if not isinstance(raw_stage, Mapping):
                continue

            stage_name = str(raw_stage.get("name", "")).strip().lower()
            if stage_name != "warmup":
                if warmup_duration_s > 0.0:
                    break
                continue

            try:
                duration_s = float(raw_stage.get("duration_s", 0.0))
            except (TypeError, ValueError):
                duration_s = 0.0
            warmup_duration_s += max(duration_s, 0.0)

        self.logger.info(
            "Loaded workload warmup duration: file=%s warmup_duration_s=%.3f",
            stage_file,
            warmup_duration_s,
        )
        return warmup_duration_s

    def _sample_rdma_queue_pending(self) -> dict[str, int]:
        pending_by_service: dict[str, int] = {
            "encoder": 0,
            "transformer": 0,
            "decoder": 0,
        }
        buffer_by_service = {
            "encoder": self.rdma_buffer_request,
            "transformer": self.rdma_buffer_phase1,
            "decoder": self.rdma_buffer_phase2,
        }
        for service_type, rdma_buffer in buffer_by_service.items():
            if rdma_buffer is None:
                continue
            try:
                pending_by_service[service_type] = int(rdma_buffer.pending_count())
            except Exception as exc:
                self.logger.warning("Failed to sample RDMA pending count for %s: %s", service_type, exc)
        return pending_by_service

    def _calc_precompute_pending(self, service_type: str, queue_sizes: Any) -> int:
        if not isinstance(queue_sizes, dict):
            return -1

        normalized: dict[str, int] = {}
        for key, value in queue_sizes.items():
            try:
                normalized[str(key)] = int(value)
            except (TypeError, ValueError):
                continue

        if service_type == "encoder":
            keys = ("req_queue", "exec_queue")
            return sum(max(normalized.get(key, 0), 0) for key in keys)

        if service_type == "transformer":
            direct_keys = ("req_queue", "waiting_queue", "exec_queue")
            pending = sum(max(normalized.get(key, 0), 0) for key in direct_keys)
            # phase1_* are pre-compute ingress queues; phase2_* are post-compute egress queues.
            pending += sum(max(value, 0) for key, value in normalized.items() if key.startswith("phase1_"))
            return pending

        if service_type == "decoder":
            direct_keys = ("req_queue", "waiting_queue", "exec_queue")
            pending = sum(max(normalized.get(key, 0), 0) for key in direct_keys)
            # Decoder transfer_* represent ingress from transformer, still before decode compute.
            pending += sum(max(value, 0) for key, value in normalized.items() if key.startswith("transfer_"))
            return pending

        return -1

    def _monitor_callback(self, results):
        monitor_runtime = getattr(self, "_monitor_runtime", None)
        if self._shutting_down or not isinstance(monitor_runtime, dict):
            return

        warmup_duration_s = float(monitor_runtime.get("warmup_duration_s", 0.0))
        autoscale_start_mono = float(monitor_runtime.get("autoscale_start_mono", time.monotonic()))
        warmup_skip_logged = bool(monitor_runtime.get("warmup_skip_logged", False))
        warmup_end_logged = bool(monitor_runtime.get("warmup_end_logged", False))
        scale_out_threshold = float(monitor_runtime.get("scale_out_threshold", 80.0))
        scale_out_max_queue_threshold = int(monitor_runtime.get("scale_out_max_queue_threshold", 2))
        scale_in_threshold = float(monitor_runtime.get("scale_in_threshold", 20.0))
        scale_cooldown_seconds = float(monitor_runtime.get("scale_cooldown_seconds", 30.0))
        last_scale_ts = monitor_runtime.get("last_scale_ts")
        if not isinstance(last_scale_ts, dict):
            return

        if warmup_duration_s > 0.0:
            elapsed_s = max(0.0, time.monotonic() - autoscale_start_mono)
            if elapsed_s < warmup_duration_s:
                if not warmup_skip_logged:
                    self.logger.info(
                        "Skip autoscaling during warmup: elapsed_s=%.3f warmup_duration_s=%.3f",
                        elapsed_s,
                        warmup_duration_s,
                    )
                    warmup_skip_logged = True
                    monitor_runtime["warmup_skip_logged"] = True
                return
            if warmup_skip_logged and not warmup_end_logged:
                self.logger.info(
                    "Warmup finished, autoscaling enabled: elapsed_s=%.3f warmup_duration_s=%.3f",
                    elapsed_s,
                    warmup_duration_s,
                )
                warmup_end_logged = True
                monitor_runtime["warmup_end_logged"] = True

        service_metrics: dict[str, list[dict[str, Any]]] = {
            "encoder": [],
            "transformer": [],
            "decoder": [],
        }

        for item in results:
            self.logger.info("monitor: %s", item)
            if not isinstance(item, dict):
                continue

            service_type = str(item.get("service_type", ""))
            if service_type not in {"encoder", "transformer", "decoder"}:
                continue

            if service_type not in {"transformer", "decoder"}:
                continue

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
            queue_sizes = item.get("queue_sizes")
            precompute_pending = self._calc_precompute_pending(service_type, queue_sizes)

            service_metrics[service_type].append(
                {
                    "gpu_utilization": gpu_utilization,
                    "monitor_address": monitor_address,
                    "queue_total_pending": queue_total_pending_int,
                    "all_queues_empty": all_queues_empty,
                    "precompute_pending": precompute_pending,
                }
            )

        rdma_pending_by_service = self._sample_rdma_queue_pending()
        scale_out_candidates: list[dict[str, Any]] = []
        service_queue_scores: dict[str, float] = {}
        service_precompute_scores: dict[str, float] = {}

        for service_type, metrics in service_metrics.items():
            if not metrics:
                continue
            avg_queue_total_pending = sum(int(metric.get("queue_total_pending", 0)) for metric in metrics) / len(metrics)
            rdma_queue_pending = int(rdma_pending_by_service.get(service_type, 0))
            service_queue_scores[service_type] = float(rdma_queue_pending) + float(avg_queue_total_pending)

            precompute_values = [int(metric.get("precompute_pending", -1)) for metric in metrics if int(metric.get("precompute_pending", -1)) >= 0]
            if precompute_values:
                avg_precompute_pending = sum(precompute_values) / len(precompute_values)
                service_precompute_scores[service_type] = float(rdma_queue_pending) + float(avg_precompute_pending)
            else:
                service_precompute_scores[service_type] = float(rdma_queue_pending)

        max_precompute_score = max(service_precompute_scores.values(), default=0.0)

        for service_type, metrics in service_metrics.items():
            if not metrics:
                continue

            now = time.time()
            avg_gpu_utilization = sum(float(metric["gpu_utilization"]) for metric in metrics) / len(metrics)
            avg_queue_total_pending = sum(int(metric.get("queue_total_pending", 0)) for metric in metrics) / len(metrics)
            max_queue_total_pending = max(int(metric.get("queue_total_pending", -1)) for metric in metrics)
            rdma_queue_pending = int(rdma_pending_by_service.get(service_type, 0))
            current_queue_score = float(service_queue_scores.get(service_type, 0.0))
            current_precompute_score = float(service_precompute_scores.get(service_type, 0.0))

            scale_out_triggered = avg_gpu_utilization > scale_out_threshold or max_queue_total_pending > scale_out_max_queue_threshold

            if scale_out_triggered and now - float(last_scale_ts.get(service_type, 0.0)) >= scale_cooldown_seconds:
                scale_out_candidates.append(
                    {
                        "service_type": service_type,
                        "score": current_queue_score,
                        "avg_gpu_utilization": avg_gpu_utilization,
                        "avg_queue_total_pending": avg_queue_total_pending,
                        "max_queue_total_pending": max_queue_total_pending,
                        "rdma_queue_pending": rdma_queue_pending,
                        "now": now,
                    }
                )

            low_metric = min(metrics, key=lambda metric: float(metric["gpu_utilization"]))
            low_utilization = float(low_metric["gpu_utilization"])
            low_monitor_address = str(low_metric["monitor_address"])
            with self._instance_lock:
                service_instance_count = sum(1 for meta in self._managed_instances.values() if meta.get("instance_type") == service_type)

            low_precompute_pending = int(low_metric.get("precompute_pending", -1))
            if low_precompute_pending >= 0:
                queues_empty_for_service = low_precompute_pending == 0
            else:
                queues_empty_for_service = bool(low_metric.get("all_queues_empty", False)) and int(low_metric.get("queue_total_pending", -1)) == 0

            blocked_by_queue_score = current_precompute_score > 0.0 and current_precompute_score >= max_precompute_score

            scale_in_triggered = (
                low_utilization < scale_in_threshold and service_instance_count > 1 and queues_empty_for_service and now - float(last_scale_ts.get(service_type, 0.0)) >= scale_cooldown_seconds
            )

            if scale_in_triggered and blocked_by_queue_score:
                self.logger.info(
                    "Skip scale in for highest precompute-score service: service=%s precompute_score=%.2f max_precompute_score=%.2f total_score=%.2f",
                    service_type,
                    current_precompute_score,
                    max_precompute_score,
                    current_queue_score,
                )
                continue

            if scale_in_triggered:
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

        if scale_out_candidates:
            target = max(
                scale_out_candidates,
                key=lambda item: (item["score"], item["max_queue_total_pending"], item["avg_gpu_utilization"]),
            )
            target_service = str(target["service_type"])
            if float(target["now"]) - float(last_scale_ts.get(target_service, 0.0)) < scale_cooldown_seconds:
                return
            try:
                new_address = self.create_instance(target_service)
                last_scale_ts[target_service] = float(target["now"])
                self.logger.info(
                    "Auto-scale out triggered: service=%s score=%.2f rdma_queue_pending=%s avg_queue_total_pending=%.2f max_queue_total_pending=%s avg_gpu_utilization=%.2f new_instance=%s",
                    target_service,
                    float(target["score"]),
                    int(target["rdma_queue_pending"]),
                    float(target["avg_queue_total_pending"]),
                    int(target["max_queue_total_pending"]),
                    float(target["avg_gpu_utilization"]),
                    new_address,
                )
            except Exception:
                pass

    def _handle_decoder_result(
        self,
        result: Any,
        *,
        expected_rooms: set[int],
        received_rooms: set[int],
        received_results: list[dict],
    ):
        if not isinstance(result, dict):
            self.logger.warning("Ignored non-dict decoder result: %s", result)
            return

        message_type = str(result.get("message_type", "decoder_result"))
        room = result.get("data_bootstrap_room")
        if room is None:
            self.logger.warning("Ignored decoder result without data_bootstrap_room: %s", result)
            return
        room = int(room)

        if message_type == "stage_metrics":
            request_metrics = result.get("request_metrics")
            if not isinstance(request_metrics, dict):
                self.logger.warning("Ignored stage metrics update without request_metrics: %s", result)
                return
            merged_metrics = self._merge_request_metrics(self._request_metrics_by_room.get(room), request_metrics)
            self._request_metrics_by_room[room] = merged_metrics
            self.logger.info(
                "Stage metrics updated room=%s stage=%s metrics=%s",
                room,
                result.get("stage_name"),
                request_metrics.get("stages", {}),
            )
            return

        if room not in expected_rooms:
            self.logger.warning("Ignored decoder result for unexpected room=%s: %s", room, result)
            return
        if room in received_rooms:
            self.logger.info("Duplicate decoder result for room=%s ignored", room)
            return

        stored_metrics = self._request_metrics_by_room.get(room)
        request_metrics = result.get("request_metrics")
        if isinstance(request_metrics, dict):
            merged_metrics = self._merge_request_metrics(stored_metrics, request_metrics)
            self._request_metrics_by_room[room] = merged_metrics
            result["request_metrics"] = merged_metrics
        elif isinstance(stored_metrics, dict):
            result["request_metrics"] = stored_metrics

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

    def _drain_decoder_results_non_block(
        self,
        *,
        result_port: int,
        expected_rooms: set[int],
        received_rooms: set[int],
        received_results: list[dict],
    ):
        while True:
            result = self.req_mgr.receive_non_block(result_port)
            if result is None:
                break
            self._handle_decoder_result(
                result,
                expected_rooms=expected_rooms,
                received_rooms=received_rooms,
                received_results=received_results,
            )

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
        self._refresh_local_host_aliases()

        static_slots_raw = disagg_cfg.get("static_instance_slots")
        self._static_instance_slots = []
        self._free_slot_ids = set()
        self._slot_reuse_block_until = {}

        if isinstance(static_slots_raw, list) and static_slots_raw:
            default_workdir = str(disagg_cfg.get("remote_workdir", Path(__file__).resolve().parents[3]))
            default_python = str(disagg_cfg.get("remote_python_executable", sys.executable))
            default_log_dir = str(disagg_cfg.get("remote_log_dir", "/tmp/lightx2v_disagg"))
            default_activate_cmd = str(disagg_cfg.get("remote_activate_cmd", "")).strip()
            default_ssh_user = str(disagg_cfg.get("ssh_user", "")).strip()
            default_ssh_bin = str(disagg_cfg.get("ssh_bin", os.getenv("DISAGG_SSH_BIN", "ssh")))
            default_use_remote_proxy = self._is_truthy(disagg_cfg.get("use_remote_proxy"), default=self._is_truthy(os.getenv("DISAGG_USE_REMOTE_PROXY"), False))
            default_proxy_req_base_port = int(disagg_cfg.get("remote_proxy_req_base_port", 28000))

            default_ssh_options_raw = disagg_cfg.get("ssh_options", [])
            if isinstance(default_ssh_options_raw, str):
                default_ssh_options = shlex.split(default_ssh_options_raw)
            elif isinstance(default_ssh_options_raw, list):
                default_ssh_options = [str(opt) for opt in default_ssh_options_raw if str(opt).strip()]
            else:
                default_ssh_options = []

            default_slot_env = disagg_cfg.get("service_env", {})
            normalized_default_slot_env: dict[str, str] = {}
            if isinstance(default_slot_env, dict):
                for key, value in default_slot_env.items():
                    normalized_default_slot_env[str(key)] = str(value)

            sidecar_base_port = int(disagg_cfg.get("sidecar_base_port", 26000))
            seen_slot_keys: set[tuple[str, int]] = set()

            for index, raw_slot in enumerate(static_slots_raw):
                if not isinstance(raw_slot, dict):
                    raise ValueError(f"invalid static_instance_slots[{index}] (expect object)")

                instance_type = str(raw_slot.get("instance_type", "")).strip().lower()
                if instance_type not in {"encoder", "transformer", "decoder"}:
                    raise ValueError(f"invalid static_instance_slots[{index}].instance_type={instance_type!r}")

                host = str(raw_slot.get("host", "")).strip()
                if not host:
                    raise ValueError(f"static_instance_slots[{index}].host cannot be empty")

                engine_rank = int(raw_slot.get("engine_rank"))
                cuda_device = str(raw_slot.get("cuda_device", engine_rank))
                slot_key = (host, engine_rank)
                if slot_key in seen_slot_keys:
                    raise ValueError(f"duplicate static slot host/rank: {slot_key}")
                seen_slot_keys.add(slot_key)

                ssh_user = str(raw_slot.get("ssh_user", default_ssh_user)).strip()
                ssh_target = f"{ssh_user}@{host}" if ssh_user else host
                ssh_bin = str(raw_slot.get("ssh_bin", default_ssh_bin))

                ssh_options_raw = raw_slot.get("ssh_options", default_ssh_options)
                if isinstance(ssh_options_raw, str):
                    ssh_options = shlex.split(ssh_options_raw)
                elif isinstance(ssh_options_raw, list):
                    ssh_options = [str(opt) for opt in ssh_options_raw if str(opt).strip()]
                else:
                    ssh_options = list(default_ssh_options)

                slot_env = dict(normalized_default_slot_env)
                raw_slot_env = raw_slot.get("env", {})
                if isinstance(raw_slot_env, dict):
                    for key, value in raw_slot_env.items():
                        slot_env[str(key)] = str(value)

                push_port = int(raw_slot.get("sidecar_push_port", sidecar_base_port + engine_rank * 2))
                req_port = int(raw_slot.get("sidecar_req_port", sidecar_base_port + engine_rank * 2 + 1))
                use_remote_proxy = self._is_truthy(raw_slot.get("use_remote_proxy"), default=default_use_remote_proxy)
                proxy_req_port = int(raw_slot.get("proxy_req_port", default_proxy_req_base_port + engine_rank))
                proxy_log_path = str(raw_slot.get("proxy_log_path", f"{default_log_dir}/instance_proxy_{engine_rank}.log"))

                self._static_instance_slots.append(
                    {
                        "slot_id": index,
                        "instance_type": instance_type,
                        "host": host,
                        "engine_rank": engine_rank,
                        "cuda_device": cuda_device,
                        "workdir": str(raw_slot.get("workdir", default_workdir)),
                        "python_executable": str(raw_slot.get("python_executable", default_python)),
                        "log_dir": str(raw_slot.get("log_dir", default_log_dir)),
                        "activate_cmd": str(raw_slot.get("activate_cmd", default_activate_cmd)).strip(),
                        "ssh_target": ssh_target,
                        "ssh_bin": ssh_bin,
                        "ssh_options": ssh_options,
                        "sidecar_push_port": push_port,
                        "sidecar_req_port": req_port,
                        "use_remote_proxy": use_remote_proxy,
                        "proxy_req_port": proxy_req_port,
                        "proxy_log_path": proxy_log_path,
                        "env": slot_env,
                    }
                )

            self._free_slot_ids = {int(slot["slot_id"]) for slot in self._static_instance_slots}
            self.logger.info("Static multi-node mode enabled with %s slots", len(self._static_instance_slots))
            self._free_gpus = set()
            return

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
            use_static_slots = bool(self._static_instance_slots)
            selected_slot: dict[str, Any] | None = None

            if use_static_slots:
                if not self._free_slot_ids:
                    raise RuntimeError("no idle static slot available")

                now = time.time()
                for slot_id in sorted(self._free_slot_ids):
                    slot = self._static_instance_slots[slot_id]
                    if slot.get("instance_type") != instance_type:
                        continue
                    if now < self._slot_reuse_block_until.get(slot_id, 0.0):
                        continue

                    host = str(slot["host"])
                    engine_rank = int(slot["engine_rank"])
                    monitor_port = MONITOR_POLLING_PORT + engine_rank
                    if self._is_tcp_port_open(host, monitor_port):
                        self.logger.warning(
                            "Skip static slot=%s host=%s rank=%s for %s creation because monitor port %s is still in use",
                            slot_id,
                            host,
                            engine_rank,
                            instance_type,
                            monitor_port,
                        )
                        continue

                    selected_slot = slot
                    break

                if selected_slot is None:
                    raise RuntimeError(f"no idle static slot available for {instance_type}: all candidates cooling down or port is in use")

                engine_rank = int(selected_slot["engine_rank"])
                host = str(selected_slot["host"])
                cuda_device = str(selected_slot["cuda_device"])
            else:
                if not self._free_gpus:
                    raise RuntimeError("no idle GPU available")

                now = time.time()
                engine_rank: int | None = None
                host = self._bootstrap_addr
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

                    engine_rank = candidate_gpu
                    break

                if engine_rank is None:
                    raise RuntimeError(f"no idle GPU available for {instance_type}: all candidates cooling down or port is in use")
                cuda_device = str(engine_rank)

            instance_cfg = self._to_plain(self._runtime_config)
            instance_cfg["disagg_mode"] = instance_type
            if instance_type == "encoder":
                instance_cfg["encoder_engine_rank"] = engine_rank
            elif instance_type == "transformer":
                instance_cfg["transformer_engine_rank"] = engine_rank
            else:
                instance_cfg["decoder_engine_rank"] = engine_rank

            model_path = instance_cfg.get("model_path")
            config_json = instance_cfg.get("config_json")
            if not model_path or not config_json:
                raise RuntimeError("model_path and config_json are required to launch service subprocess")
            service_config_json = self._resolve_service_config_json(str(config_json), instance_type)

            cmd = self._build_service_command(instance_type, engine_rank, instance_cfg, service_config_json)
            cmd = self._maybe_wrap_service_command_with_nsys(
                host=host,
                instance_type=instance_type,
                engine_rank=engine_rank,
                instance_cfg=instance_cfg,
                command=cmd,
            )

            process: subprocess.Popen | None = None
            process_meta: dict[str, Any] | None = None
            sidecar_meta: dict[str, Any]
            launch_mode = "local"

            try:
                if use_static_slots and selected_slot is not None and not self._is_local_host(host):
                    launch_mode = "remote"
                    process_meta, sidecar_meta = self._launch_remote_instance(selected_slot, instance_type, cmd, cuda_device)
                else:
                    sidecar_meta = self._start_sidecar_process(instance_type, cuda_device, bind_host=host)
                    env = os.environ.copy()
                    if use_static_slots and selected_slot is not None:
                        slot_env = selected_slot.get("env")
                        if isinstance(slot_env, dict):
                            for key, value in slot_env.items():
                                env[str(key)] = str(value)
                    self._ensure_rdma_preferred_ipv4_env(host, env)
                    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
                    env["LIGHTX2V_SIDECAR_PUSH_ADDR"] = str(sidecar_meta["push_addr"])
                    env["LIGHTX2V_SIDECAR_REQ_ADDR"] = str(sidecar_meta["req_addr"])
                    process = subprocess.Popen(
                        cmd,
                        env=env,
                        start_new_session=True,
                    )

                monitor_port = MONITOR_POLLING_PORT + engine_rank
                if not self._wait_for_tcp_port_state(host, monitor_port, should_be_open=True, timeout_seconds=self._instance_start_timeout_seconds):
                    raise RuntimeError(f"service {instance_type} rank={engine_rank} host={host} failed to expose monitor port {monitor_port}")
            except Exception:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        process.kill()

                if launch_mode == "remote" and selected_slot is not None and process_meta is not None:
                    remote_pid = process_meta.get("pid")
                    if isinstance(remote_pid, int) and remote_pid > 0:
                        self._stop_remote_pid(selected_slot, remote_pid, self._graceful_reclaim_timeout_seconds)

                if "sidecar_meta" in locals():
                    if launch_mode == "remote" and selected_slot is not None:
                        sidecar_pid = sidecar_meta.get("pid")
                        if isinstance(sidecar_pid, int) and sidecar_pid > 0:
                            self._stop_remote_pid(selected_slot, sidecar_pid, self._force_kill_wait_seconds)
                    else:
                        sidecar_process = sidecar_meta.get("process")
                        if sidecar_process is not None and sidecar_process.poll() is None:
                            sidecar_process.terminate()
                            try:
                                sidecar_process.wait(timeout=2.0)
                            except subprocess.TimeoutExpired:
                                sidecar_process.kill()
                raise

            instance_address = f"{host}:{REQUEST_POLLING_PORT + engine_rank}"
            if use_static_slots and selected_slot is not None:
                self._free_slot_ids.remove(int(selected_slot["slot_id"]))
            else:
                self._free_gpus.remove(engine_rank)
            if self._enable_monitor:
                monitor_node = f"tcp://{host}:{MONITOR_POLLING_PORT + engine_rank}"
                if monitor_node not in self.monitor.nodes:
                    self.monitor.nodes.append(monitor_node)
            self._managed_instances[instance_address] = {
                "instance_type": instance_type,
                "gpu_id": engine_rank,
                "host": host,
                "launch_mode": launch_mode,
                "cuda_device": cuda_device,
                "process": process,
                "process_meta": process_meta,
                "sidecar": sidecar_meta,
                "slot_id": int(selected_slot["slot_id"]) if selected_slot is not None else None,
                "static_slot": self._to_plain(selected_slot) if selected_slot is not None else None,
            }
            self.started_instances.append((instance_type, instance_address))
            self.add_instance(instance_type, instance_address)
            self.logger.info(
                "Created %s instance host=%s rank=%s mode=%s address=%s",
                instance_type,
                host,
                engine_rank,
                launch_mode,
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
                if (instance_type, target_address) in self.started_instances:
                    self.started_instances.remove((instance_type, target_address))
                self.logger.warning(
                    "Skip reclaim for already-removed %s instance address=%s",
                    instance_type,
                    target_address,
                )
                return target_address
            if meta.get("instance_type") != instance_type:
                raise RuntimeError(f"instance type mismatch for {target_address}: expected={instance_type} got={meta.get('instance_type')}")

            process = meta.get("process")
            process_meta = meta.get("process_meta") if isinstance(meta.get("process_meta"), dict) else None
            gpu_id = int(meta.get("gpu_id"))
            sidecar_meta = meta.get("sidecar") if isinstance(meta.get("sidecar"), dict) else None
            host = str(meta.get("host", self._bootstrap_addr))
            launch_mode = str(meta.get("launch_mode", "local"))
            static_slot = meta.get("static_slot") if isinstance(meta.get("static_slot"), dict) else None
            slot_id_raw = meta.get("slot_id")
            slot_id = int(slot_id_raw) if slot_id_raw is not None else None

            self.remove_instance(instance_type, target_address)
            monitor_node = self._monitor_node_from_instance_address(target_address)

            if launch_mode == "remote":
                if static_slot is None:
                    raise RuntimeError(f"remote instance metadata missing static slot for {target_address}")

                remote_service_pid = None
                if process_meta is not None and isinstance(process_meta.get("pid"), int):
                    remote_service_pid = int(process_meta["pid"])
                if remote_service_pid is not None and remote_service_pid > 0:
                    self._stop_remote_pid(static_slot, remote_service_pid, self._graceful_reclaim_timeout_seconds)

                if sidecar_meta is not None and isinstance(sidecar_meta.get("pid"), int):
                    self._stop_remote_pid(static_slot, int(sidecar_meta["pid"]), self._force_kill_wait_seconds)
            else:
                if process is not None and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except Exception:
                        process.terminate()
                    try:
                        process.wait(timeout=self._graceful_reclaim_timeout_seconds)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except Exception:
                            process.kill()
                        try:
                            process.wait(timeout=self._force_kill_wait_seconds)
                        except subprocess.TimeoutExpired as exc:
                            raise RuntimeError(f"process did not exit after kill for {instance_type} instance {target_address}") from exc

            if self._enable_monitor and monitor_node in self.monitor.nodes:
                self.monitor.nodes.remove(monitor_node)

            monitor_port = MONITOR_POLLING_PORT + gpu_id
            if not self._wait_for_tcp_port_state(host, monitor_port, should_be_open=False, timeout_seconds=5.0):
                self.logger.warning(
                    "Monitor port still open after reclaim: service=%s host=%s rank=%s port=%s",
                    instance_type,
                    host,
                    gpu_id,
                    monitor_port,
                )

            if slot_id is not None and slot_id in range(len(self._static_instance_slots)):
                self._free_slot_ids.add(slot_id)
                self._slot_reuse_block_until[slot_id] = time.time() + self._gpu_reuse_grace_seconds
            else:
                self._free_gpus.add(gpu_id)
                self._gpu_reuse_block_until[gpu_id] = time.time() + self._gpu_reuse_grace_seconds
            self._managed_instances.pop(target_address, None)
            if (instance_type, target_address) in self.started_instances:
                self.started_instances.remove((instance_type, target_address))

            if sidecar_meta is not None and launch_mode != "remote":
                reclaim_thread = Thread(
                    target=self._reclaim_sidecar_when_drained,
                    args=(instance_type, target_address, sidecar_meta),
                    name=f"sidecar-reclaim-{instance_type}-{gpu_id}",
                    daemon=True,
                )
                reclaim_thread.start()
                self._sidecar_reclaim_threads.append(reclaim_thread)

            self.logger.info(
                "Reclaimed %s instance from host=%s rank=%s address=%s",
                instance_type,
                host,
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
        if not self._is_centralized_enabled():
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
        else:
            self.logger.info("IS_CENTRALIZED enabled, skip controller request RDMA ring initialization")

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
            slots if self.rdma_buffer_request is not None else 0,
            slot_size if self.rdma_buffer_request is not None else 0,
            need_bytes if self.rdma_buffer_request is not None else 0,
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

    def _build_latency_summary(self, result: dict[str, Any], controller_recv_ts: float) -> dict[str, float] | None:
        request_metrics = result.get("request_metrics")
        if not isinstance(request_metrics, dict):
            return None

        def _as_float(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _stage(name: str) -> dict[str, Any]:
            stages = request_metrics.get("stages")
            if not isinstance(stages, dict):
                return {}
            stage_metrics = stages.get(name)
            return stage_metrics if isinstance(stage_metrics, dict) else {}

        controller_send_ts = _as_float(request_metrics.get("controller_send_ts"))
        if controller_send_ts is None:
            return None

        centralized_mode = self._is_centralized_enabled()
        summary: dict[str, float] = {
            "end_to_end_delay_s": controller_recv_ts - controller_send_ts,
        }

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

        if centralized_mode:
            if encoder_recv_ts is not None:
                summary["controller_to_encoder_comm_delay_s"] = encoder_recv_ts - controller_send_ts
            if encoder_recv_ts is not None and encoder_compute_start_ts is not None:
                summary["encoder_scheduling_delay_s"] = encoder_compute_start_ts - encoder_recv_ts
            if encoder_compute_start_ts is not None and encoder_compute_end_ts is not None:
                summary["encoder_compute_delay_s"] = encoder_compute_end_ts - encoder_compute_start_ts
            if encoder_output_enqueued_ts is not None and transformer_recv_ts is not None:
                summary["encoder_communication_delay_s"] = transformer_recv_ts - controller_send_ts
            if transformer_recv_ts is not None and transformer_compute_start_ts is not None:
                summary["transformer_scheduling_delay_s"] = transformer_compute_start_ts - transformer_recv_ts
            if transformer_compute_start_ts is not None and transformer_compute_end_ts is not None:
                summary["transformer_compute_delay_s"] = transformer_compute_end_ts - transformer_compute_start_ts
            if transformer_recv_ts is not None:
                summary["transformer_communication_delay_s"] = transformer_recv_ts - controller_send_ts
            if decoder_recv_ts is not None and decoder_compute_start_ts is not None:
                summary["decoder_scheduling_delay_s"] = decoder_compute_start_ts - decoder_recv_ts
            if decoder_compute_start_ts is not None and decoder_compute_end_ts is not None:
                summary["decoder_compute_delay_s"] = decoder_compute_end_ts - decoder_compute_start_ts
            if decoder_recv_ts is not None:
                summary["decoder_communication_delay_s"] = decoder_recv_ts - controller_send_ts

            component_keys = [
                "controller_to_encoder_comm_delay_s",
                "encoder_scheduling_delay_s",
                "encoder_compute_delay_s",
                "encoder_communication_delay_s",
                "transformer_scheduling_delay_s",
                "transformer_compute_delay_s",
                "transformer_communication_delay_s",
                "decoder_scheduling_delay_s",
                "decoder_compute_delay_s",
                "decoder_communication_delay_s",
            ]
            if all(key in summary for key in component_keys):
                summary["sum_of_components_s"] = sum(summary[key] for key in component_keys)
        else:
            if encoder_recv_ts is not None:
                summary["controller_to_encoder_comm_delay_s"] = encoder_recv_ts - controller_send_ts
            if encoder_recv_ts is not None and encoder_compute_start_ts is not None:
                summary["encoder_scheduling_delay_s"] = encoder_compute_start_ts - encoder_recv_ts
            if encoder_compute_start_ts is not None and encoder_compute_end_ts is not None:
                summary["encoder_compute_delay_s"] = encoder_compute_end_ts - encoder_compute_start_ts
            if encoder_output_enqueued_ts is not None and transformer_recv_ts is not None:
                summary["encoder_communication_delay_s"] = transformer_recv_ts - encoder_output_enqueued_ts
            if transformer_recv_ts is not None and transformer_compute_start_ts is not None:
                summary["transformer_scheduling_delay_s"] = transformer_compute_start_ts - transformer_recv_ts
            if transformer_compute_start_ts is not None and transformer_compute_end_ts is not None:
                summary["transformer_compute_delay_s"] = transformer_compute_end_ts - transformer_compute_start_ts
            if transformer_output_enqueued_ts is not None and decoder_recv_ts is not None:
                summary["transformer_communication_delay_s"] = decoder_recv_ts - transformer_output_enqueued_ts
            if decoder_recv_ts is not None and decoder_compute_start_ts is not None:
                summary["decoder_scheduling_delay_s"] = decoder_compute_start_ts - decoder_recv_ts
            if decoder_compute_start_ts is not None and decoder_compute_end_ts is not None:
                summary["decoder_compute_delay_s"] = decoder_compute_end_ts - decoder_compute_start_ts
            if decoder_output_enqueued_ts is not None:
                summary["decoder_communication_delay_s"] = controller_recv_ts - decoder_output_enqueued_ts

            component_keys = [
                "controller_to_encoder_comm_delay_s",
                "encoder_scheduling_delay_s",
                "encoder_compute_delay_s",
                "encoder_communication_delay_s",
                "transformer_scheduling_delay_s",
                "transformer_compute_delay_s",
                "transformer_communication_delay_s",
                "decoder_scheduling_delay_s",
                "decoder_compute_delay_s",
                "decoder_communication_delay_s",
            ]
            if all(key in summary for key in component_keys):
                summary["sum_of_components_s"] = sum(summary[key] for key in component_keys)
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

        room_raw = config.get("data_bootstrap_room")
        try:
            room = int(room_raw)
        except (TypeError, ValueError):
            room = None

        request_metrics = config.get("request_metrics")
        if room is not None and isinstance(request_metrics, dict):
            self._request_metrics_by_room[room] = self._merge_request_metrics(None, request_metrics)

        if self._is_centralized_enabled():
            request_config = self._to_plain(config)

            encoder_address = self.encoder_policy.schedule()
            transformer_address = self.transformer_policy.schedule()
            decoder_address = self.decoder_policy.schedule()

            def _address_to_rank(instance_address: str) -> int:
                _, port_str = instance_address.rsplit(":", 1)
                return int(port_str) - REQUEST_POLLING_PORT

            encoder_rank = _address_to_rank(encoder_address)
            transformer_rank = _address_to_rank(transformer_address)
            decoder_rank = _address_to_rank(decoder_address)

            request_config["encoder_engine_rank"] = encoder_rank
            request_config["transformer_engine_rank"] = transformer_rank
            request_config["decoder_engine_rank"] = decoder_rank
            request_config["encoder_node_address"] = encoder_address
            request_config["transformer_node_address"] = transformer_address
            request_config["decoder_node_address"] = decoder_address
            request_config["controller_control_host"] = request_config.get("controller_result_host", self._bootstrap_addr)
            request_config["controller_control_port"] = int(request_config.get("controller_control_port", REQUEST_POLLING_PORT - 3))

            for instance_type, target_address in (
                ("encoder", encoder_address),
                ("transformer", transformer_address),
                ("decoder", decoder_address),
            ):
                host, port_str = target_address.rsplit(":", 1)
                self.req_mgr.send(host, int(port_str), request_config)
                self.logger.info("Request dispatched to %s via ZMQ: target=%s", instance_type, target_address)
            return

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
        control_port = int(config.get("controller_control_port", REQUEST_POLLING_PORT - 3))
        self._bootstrap_addr = str(bootstrap_addr)
        self._runtime_config = self._to_plain(config)
        self._init_gpu_pool(config)
        self._enable_monitor = self._is_monitor_enabled()
        centralized_mode = self._is_centralized_enabled()

        # self.encoder_policy = RoundRobinPolicy()
        # self.transformer_policy = RoundRobinPolicy()
        # self.decoder_policy = RoundRobinPolicy()

        self._init_request_rdma_buffer(bootstrap_addr, config)
        if centralized_mode:
            self.logger.info("IS_CENTRALIZED enabled, controller will dispatch requests via ZMQ")

        time.sleep(5.0)

        if self._static_instance_slots:
            self.logger.info(
                "Starting managed instances from static_instance_slots: %s",
                [slot["instance_type"] for slot in self._static_instance_slots],
            )
            for slot in self._static_instance_slots:
                self.create_instance(str(slot["instance_type"]))
        else:
            for instance_type in ("encoder", "transformer", "decoder"):
                self.create_instance(instance_type)
            for _ in range(5):
                self.create_instance("transformer")

        instance_warmup_wait_s = int(os.getenv("DISAGG_INSTANCE_WARMUP_WAIT_S", "30"))
        if instance_warmup_wait_s > 0:
            self.logger.info(
                "Managed instances created, waiting %ss before accepting requests",
                instance_warmup_wait_s,
            )
            time.sleep(instance_warmup_wait_s)

        monitor_stop_event: Event | None = None
        monitor_thread: Thread | None = None
        ok_gate_stop_event: Event | None = None
        ok_gate_thread: Thread | None = None
        self._monitor_runtime = None

        if self._enable_monitor:
            monitor_stop_event = Event()
            warmup_duration_s = self._load_warmup_duration_seconds(config)
            autoscale_start_mono = time.monotonic()
            warmup_skip_logged = False
            warmup_end_logged = False
            scale_out_threshold = 80.0
            scale_out_max_queue_threshold = 2
            scale_in_threshold = 20.0
            scale_cooldown_seconds = 30.0
            last_scale_ts: dict[str, float] = {
                "encoder": 0.0,
                "transformer": 0.0,
                "decoder": 0.0,
            }

            self._monitor_runtime = {
                "warmup_duration_s": warmup_duration_s,
                "autoscale_start_mono": autoscale_start_mono,
                "warmup_skip_logged": warmup_skip_logged,
                "warmup_end_logged": warmup_end_logged,
                "scale_out_threshold": scale_out_threshold,
                "scale_out_max_queue_threshold": scale_out_max_queue_threshold,
                "scale_in_threshold": scale_in_threshold,
                "scale_cooldown_seconds": scale_cooldown_seconds,
                "last_scale_ts": last_scale_ts,
            }

            monitor_thread = Thread(
                target=self.monitor.run_forever,
                kwargs={
                    "interval_seconds": 2.0,
                    "callback": self._monitor_callback,
                    "stop_event": monitor_stop_event,
                },
                name="controller-monitor",
                daemon=True,
            )
            monitor_thread.start()
            self.logger.info("ENABLE_MONITOR enabled, monitor thread started")
        else:
            self.logger.info("ENABLE_MONITOR is not set, skip monitor logic")

        if centralized_mode:
            ok_gate_stop_event = Event()
            ok_gate_thread = Thread(
                target=self._run_centralized_ok_server,
                args=(ok_gate_stop_event, self._bootstrap_addr, control_port),
                name="controller-ok-gate",
                daemon=True,
            )
            ok_gate_thread.start()
            self.logger.info("Centralized OK gate server started on %s:%s", self._bootstrap_addr, control_port)

        time.sleep(5.0)

        base_save_path = config.get("save_path")
        expected_rooms: set[int] = set()
        received_rooms: set[int] = set()
        received_results: list[dict] = []
        next_room = 0
        batch_request_start_ts: float | None = None
        load_from_user = str(os.getenv("LOAD_FROM_USER", "0")).strip().lower() in {"1", "true", "yes", "on"}
        auto_request_count_raw = config.get("request_count", os.getenv("DISAGG_AUTO_REQUEST_COUNT", "30"))
        try:
            auto_request_count = int(auto_request_count_raw)
        except (TypeError, ValueError):
            self.logger.warning(
                "Invalid request_count=%s, fallback to 30",
                auto_request_count_raw,
            )
            auto_request_count = 30
        if auto_request_count <= 0:
            self.logger.warning("request_count must be positive, fallback to 30")
            auto_request_count = 30

        try:
            generated_request_count = 0
            if load_from_user:
                self.logger.info("LOAD_FROM_USER enabled, waiting workload configs on port=%s", request_ingress_port)
            else:
                self.logger.info(
                    "LOAD_FROM_USER disabled, generating requests from config: count=%s",
                    auto_request_count,
                )

            while True:
                if load_from_user:
                    workload_config = self.req_mgr.receive(request_ingress_port)
                    if not isinstance(workload_config, dict):
                        self.logger.warning("Ignored invalid workload config packet: %s", workload_config)
                        continue

                    if workload_config.get("workload_end") or workload_config.get("end") or workload_config.get("stop"):
                        self.logger.info("Received workload end signal, stop accepting new configs.")
                        break
                else:
                    if generated_request_count >= auto_request_count:
                        break
                    workload_config = {}
                    generated_request_count += 1

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

                self._drain_decoder_results_non_block(
                    result_port=result_port,
                    expected_rooms=expected_rooms,
                    received_rooms=received_rooms,
                    received_results=received_results,
                )

            self.logger.info(
                "Waiting for decoder results: expected=%s on port=%s",
                sorted(expected_rooms),
                result_port,
            )
            while len(received_rooms) < len(expected_rooms):
                result = self.req_mgr.receive(result_port)
                self._handle_decoder_result(
                    result,
                    expected_rooms=expected_rooms,
                    received_rooms=received_rooms,
                    received_results=received_results,
                )

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
            if monitor_stop_event is not None:
                monitor_stop_event.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=2.0)
            if ok_gate_stop_event is not None:
                ok_gate_stop_event.set()
            if ok_gate_thread is not None:
                ok_gate_thread.join(timeout=2.0)
            self._monitor_runtime = None

            for instance_type, address in reversed(list(self.started_instances)):
                try:
                    self.reclaim_instance(instance_type, address)
                except Exception:
                    self.logger.exception("Failed to reclaim %s instance address=%s", instance_type, address)

            for thread in list(self._sidecar_reclaim_threads):
                if thread.is_alive():
                    thread.join(timeout=3.0)
