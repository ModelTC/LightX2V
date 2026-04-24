from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import zmq


class InstanceProxyServer:
    """Remote process proxy that creates/stops local disagg service processes.

    This server is intended to run on remote nodes where the local runtime
    environment is trusted. The controller sends simple commands to this proxy
    instead of assembling remote launch scripts for every instance operation.
    """

    def __init__(self, bind_addr: str, workdir: str, log_dir: str):
        self.bind_addr = str(bind_addr)
        self.workdir = str(workdir)
        self.log_dir = str(log_dir)
        self._running = True
        self._managed: dict[int, subprocess.Popen] = {}

    def _normalize_env(self, extra_env: Any, cuda_device: str) -> dict[str, str]:
        env = os.environ.copy()
        if isinstance(extra_env, Mapping):
            for key, value in extra_env.items():
                env[str(key)] = str(value)
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _terminate_pid(self, pid: int, timeout_seconds: float) -> bool:
        process = self._managed.get(pid)
        timeout_seconds = max(1.0, float(timeout_seconds))

        if process is not None:
            if process.poll() is not None:
                self._managed.pop(pid, None)
                return True

            try:
                os.killpg(process.pid, signal.SIGTERM)
            except Exception:
                process.terminate()

            deadline = time.time() + timeout_seconds
            while time.time() < deadline:
                if process.poll() is not None:
                    self._managed.pop(pid, None)
                    return True
                time.sleep(0.1)

            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()

            try:
                process.wait(timeout=2.0)
            except Exception:
                pass
            self._managed.pop(pid, None)
            return process.poll() is not None

        # Fallback for pids created before current proxy process lifetime.
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        except Exception:
            return False

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except Exception:
                break
            time.sleep(0.1)

        try:
            os.kill(pid, signal.SIGKILL)
            return True
        except ProcessLookupError:
            return True
        except Exception:
            return False

    def _start_instance(self, msg: dict[str, Any]) -> dict[str, Any]:
        instance_type = str(msg.get("instance_type", ""))
        engine_rank = int(msg.get("engine_rank", -1))
        cuda_device = str(msg.get("cuda_device", "0"))
        python_executable = str(msg.get("python_executable", "python"))
        service_argv = msg.get("service_argv", [])
        sidecar_push_addr = str(msg.get("sidecar_push_addr", "")).strip()
        sidecar_req_addr = str(msg.get("sidecar_req_addr", "")).strip()
        service_log_path = str(msg.get("service_log_path", "")).strip()
        sidecar_log_path = str(msg.get("sidecar_log_path", "")).strip()
        workdir = str(msg.get("workdir", self.workdir))
        log_dir = str(msg.get("log_dir", self.log_dir))
        extra_env = msg.get("env", {})

        if not instance_type:
            raise ValueError("instance_type is required")
        if engine_rank < 0:
            raise ValueError("engine_rank must be non-negative")
        if not isinstance(service_argv, list) or not service_argv:
            raise ValueError("service_argv must be a non-empty list")
        if not sidecar_push_addr or not sidecar_req_addr:
            raise ValueError("sidecar_push_addr and sidecar_req_addr are required")

        if not service_log_path:
            service_log_path = f"{log_dir}/{instance_type}_{engine_rank}_service.log"
        if not sidecar_log_path:
            sidecar_log_path = f"{log_dir}/{instance_type}_{engine_rank}_sidecar.log"

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(Path(service_log_path).parent, exist_ok=True)
        os.makedirs(Path(sidecar_log_path).parent, exist_ok=True)

        sidecar_env = self._normalize_env(extra_env, cuda_device)
        service_env = self._normalize_env(extra_env, cuda_device)
        service_env["LIGHTX2V_SIDECAR_PUSH_ADDR"] = sidecar_push_addr
        service_env["LIGHTX2V_SIDECAR_REQ_ADDR"] = sidecar_req_addr

        sidecar_cmd = [
            python_executable,
            "-m",
            "lightx2v.disagg.services.data_mgr_sidecar",
            "--push-addr",
            sidecar_push_addr,
            "--req-addr",
            sidecar_req_addr,
        ]
        service_cmd = [python_executable, *[str(part) for part in service_argv]]

        with open(sidecar_log_path, "w", encoding="utf-8") as sidecar_log:
            sidecar_proc = subprocess.Popen(
                sidecar_cmd,
                cwd=workdir,
                env=sidecar_env,
                stdout=sidecar_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        time.sleep(0.3)
        if sidecar_proc.poll() is not None:
            raise RuntimeError(f"failed to start sidecar process, exited with code={sidecar_proc.returncode}")

        with open(service_log_path, "w", encoding="utf-8") as service_log:
            service_proc = subprocess.Popen(
                service_cmd,
                cwd=workdir,
                env=service_env,
                stdout=service_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        if service_proc.poll() is not None:
            self._terminate_pid(sidecar_proc.pid, timeout_seconds=2.0)
            raise RuntimeError(f"failed to start service process, exited with code={service_proc.returncode}")

        self._managed[sidecar_proc.pid] = sidecar_proc
        self._managed[service_proc.pid] = service_proc

        return {
            "instance_type": instance_type,
            "engine_rank": engine_rank,
            "sidecar_pid": sidecar_proc.pid,
            "service_pid": service_proc.pid,
            "sidecar_log_path": sidecar_log_path,
            "service_log_path": service_log_path,
        }

    def handle(self, msg: dict[str, Any]) -> dict[str, Any]:
        cmd = str(msg.get("cmd", "")).strip()

        if cmd == "ping":
            return {"ok": True, "data": {"alive": True, "managed": len(self._managed)}}

        if cmd == "start_instance":
            data = self._start_instance(msg)
            return {"ok": True, "data": data}

        if cmd == "stop_pid":
            pid = int(msg.get("pid", -1))
            timeout_seconds = float(msg.get("timeout_seconds", 10.0))
            if pid <= 0:
                return {"ok": False, "error": "invalid pid"}
            stopped = self._terminate_pid(pid, timeout_seconds=timeout_seconds)
            return {"ok": bool(stopped), "data": {"pid": pid, "stopped": bool(stopped)}}

        if cmd == "shutdown":
            self._running = False
            return {"ok": True, "data": {"shutting_down": True}}

        if cmd == "stats":
            managed_alive = 0
            for process in self._managed.values():
                if process.poll() is None:
                    managed_alive += 1
            return {"ok": True, "data": {"managed_alive": managed_alive}}

        return {"ok": False, "error": f"unsupported command: {cmd}"}

    def serve(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.bind_addr)
        try:
            while self._running:
                try:
                    msg = socket.recv_pyobj()
                    if not isinstance(msg, dict):
                        socket.send_pyobj({"ok": False, "error": "request must be a dict"})
                        continue
                    reply = self.handle(msg)
                except Exception as exc:
                    reply = {"ok": False, "error": str(exc)}
                socket.send_pyobj(reply)
        finally:
            socket.close(0)
            context.term()
            for pid in list(self._managed.keys()):
                self._terminate_pid(pid, timeout_seconds=2.0)


def main():
    parser = argparse.ArgumentParser(description="Remote instance proxy for disagg services")
    parser.add_argument("--bind-addr", type=str, required=True)
    parser.add_argument("--workdir", type=str, default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--log-dir", type=str, default="/tmp/lightx2v_disagg")
    args = parser.parse_args()

    server = InstanceProxyServer(bind_addr=args.bind_addr, workdir=args.workdir, log_dir=args.log_dir)
    server.serve()


if __name__ == "__main__":
    main()
