from __future__ import annotations

import atexit
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .service import ServerRef, ServerSettings, same_gpu_policy

log = logging.getLogger(__name__)
TRANSIENT = (urllib.error.URLError, TimeoutError, ConnectionError)


class APIError(RuntimeError):
    def __init__(self, status: int, body: str):
        self.status = status
        super().__init__(f"LightX2V HTTP {status}: {body[:2000]}")


class Client:
    def __init__(self, port: int, timeout: float = 5):
        self.url = f"http://127.0.0.1:{port}"
        self.timeout = timeout
        # Local video prompts and paths must never go through HTTP_PROXY.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, method: str, path: str, body=None) -> dict:
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = urllib.request.Request(self.url + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise APIError(exc.code, exc.read(8192).decode(errors="replace")) from exc
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object from LightX2V")  # noqa: TRY004 - invalid HTTP payload
        return result


@dataclass
class Instance:
    server: ServerSettings
    process: subprocess.Popen
    client: Client
    directory: Path
    last_used: float
    busy: bool = True
    release_requested: bool = False
    cleanup_failed: bool = False
    _control_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def request_stop(self, *, force=False):
        # Only the owning supervisor receives commands; never signal a caller-supplied PID.
        with self._control_lock:
            pipe = self.process.stdin
            if pipe is not None and not pipe.closed:
                try:
                    if force:
                        pipe.write(b"K")
                        pipe.flush()
                except BrokenPipeError:
                    pass  # The supervisor may already have exited; _stop still verifies its state.
                finally:
                    try:
                        pipe.close()
                    except BrokenPipeError:
                        pass

    def state(self) -> dict:
        try:
            return json.loads((self.directory / "state.json").read_text())
        except (OSError, ValueError):
            return {}


class _RunInterrupt:
    """Stop this lease's process even when its HTTP request blocks."""

    def __init__(self, callback):
        self._callback = callback
        self._mutex = threading.Lock()
        self._done = threading.Event()
        self.error = None
        self._instance = None
        self._thread = threading.Thread(target=self._watch, name="lightx2v-interrupt", daemon=True)
        self._thread.start()

    def _watch(self):
        while not self._done.is_set():
            try:
                self._callback()
            except BaseException as exc:  # noqa: BLE001 - preserve Comfy interrupt exceptions
                with self._mutex:
                    self.error = exc
                    if self._instance is not None:
                        self._instance.request_stop(force=True)
                return
            self._done.wait(0.05)

    def bind(self, instance):
        with self._mutex:
            self._instance = instance
            if self.error is not None:
                instance.request_stop(force=True)

    def check(self):
        with self._mutex:
            error = self.error
        if error is not None:
            raise error

    def close(self):
        self._done.set()
        self._thread.join()  # Cannot let an old lease consume the next prompt's interrupt.


def _ports() -> tuple[int, int]:
    # Hold both reservations simultaneously, so HTTP and metrics never get the same port.
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        return first.getsockname()[1], second.getsockname()[1]


class Manager:
    """A pool of resident backends, with serialized inference and scoped ownership."""

    def __init__(self, root: Path, before_inference=lambda: None):
        if os.name != "posix":
            raise RuntimeError("Managed server mode currently requires Linux/POSIX process groups")
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.before_inference = before_inference
        self._lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._instances = {}
        self.same_gpu_policy = same_gpu_policy()
        self._slot_file = None
        self._closed = False
        self._wake = threading.Event()
        self._reaper = threading.Thread(target=self._idle_loop, name="lightx2v-idle", daemon=True)
        self._reaper.start()
        atexit.register(self.close)

    def _claim_slot(self):
        import fcntl

        if self._slot_file is not None:
            return
        handle = (self.root / "manager.lock").open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RuntimeError("Another manager owns this runtime directory")
        self._slot_file = handle

    def _release_slot(self):
        if not self._instances and self._slot_file is not None:
            self._slot_file.close()
            self._slot_file = None

    def _stop(self, instance, *, force=False):
        # Caller holds _state_lock. Keep an unconfirmed instance in the pool.
        process = instance.process
        instance.cleanup_failed = True
        instance.request_stop(force=force)
        try:
            process.wait(timeout=8 if force else instance.server.settings["stop_grace"] + 8)
        except subprocess.TimeoutExpired as exc:
            # Do not kill the supervisor and abandon its GPU children; refuse any new launch.
            raise RuntimeError(f"Supervisor cleanup timed out; inspect {instance.directory}") from exc
        state = instance.state()
        if state.get("phase") != "stopped" and not (state.get("phase") == "error" and state.get("pid") is None):
            raise RuntimeError(f"GPU process cleanup is unconfirmed; inspect {instance.directory}")
        log.info("Stopped LightX2V %s (PID %s)", instance.server.name, state.get("pid"))
        self._instances.pop(instance.server.service_key, None)
        self._release_slot()

    def _start(self, server: ServerSettings):
        self._claim_slot()
        s = server.settings
        directory = self.root / "instances" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        cache = directory / "cache"
        (cache / "outputs").mkdir(parents=True)
        config_path = directory / "config.json"
        config_path.write_text(server.config_json, encoding="utf-8")
        port, metric_port = _ports()
        env = {"CUDA_VISIBLE_DEVICES": s["CUDA_VISIBLE_DEVICES"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "LIGHTX2V_CACHE_DIR": str(cache), "PYTHONPATH": s["lightx2v_path"], "PYTHONUNBUFFERED": "1"}
        spec = {
            "argv": server.command(config_path, port, metric_port),
            "env": env,
            "cwd": s["lightx2v_path"],
            "stop_grace": s["stop_grace"],
            "state_path": str(directory / "state.json"),
            "log_path": str(directory / "server.log"),
            "fingerprint": server.fingerprint,
            "service_key": server.service_key,
        }
        spec_path = directory / "launch.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        os.chmod(spec_path, 0o600)
        try:
            with (directory / "supervisor.log").open("ab", buffering=0) as supervisor_log:
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).with_name("supervisor.py")), str(spec_path)],
                    stdin=subprocess.PIPE,
                    stdout=supervisor_log,
                    stderr=supervisor_log,
                    close_fds=True,
                    start_new_session=True,
                    pass_fds=(self._slot_file.fileno(),),
                )
        except BaseException:
            self._release_slot()
            raise
        instance = Instance(server, process, Client(port, s["http_timeout"]), directory, time.monotonic())
        self._instances[server.service_key] = instance
        log.info("Starting %s; log: %s", server.name, directory / "server.log")
        return instance

    def _wait_ready(self, instance: Instance, interrupt):
        server, process, directory = instance.server, instance.process, instance.directory
        s = server.settings
        deadline = time.monotonic() + s["startup_timeout"]
        while time.monotonic() < deadline:
            interrupt()
            if process.poll() is not None:
                raise RuntimeError(f"LightX2V exited during startup; see {directory / 'server.log'}")
            try:
                metadata = instance.client.request("GET", "/v1/service/metadata")
            except TRANSIENT:
                time.sleep(min(s["poll_interval"], 0.25))
                continue
            if metadata.get("model_cls") != s["model_cls"] or os.path.abspath(str(metadata.get("model_path", ""))) != s["model_path"] or int(metadata.get("nproc_per_node", 0)) != s["nproc_per_node"]:
                raise RuntimeError("Service metadata does not match the launched server")
            if instance.state().get("phase") != "running" or process.poll() is not None:
                raise RuntimeError("Supervisor exited before service became ready")
            log.info("Ready: %s at %s", server.name, instance.client.url)
            return instance
        raise TimeoutError(f"LightX2V startup timed out; see {directory / 'server.log'}")

    def _ensure(self, server: ServerSettings, interruption):
        with self._state_lock:
            interruption.check()
            if any(item.cleanup_failed for item in self._instances.values()):
                raise RuntimeError("GPU process cleanup is unconfirmed; release the failed service before starting work")
            instance = self._instances.get(server.service_key)
            if instance and instance.server.fingerprint == server.fingerprint and instance.process.poll() is None:
                instance.busy = True
                interruption.bind(instance)
                try:
                    status = instance.client.request("GET", "/v1/service/status")
                    interruption.check()
                    if status.get("service_status") == "idle" and not status.get("current_task"):
                        return instance
                except (APIError, ValueError) + TRANSIENT:
                    pass
            interruption.check()
            if instance:
                self._stop(instance)
            if self.same_gpu_policy == "exclusive":
                devices = set(server.settings["CUDA_VISIBLE_DEVICES"].split(","))
                for other in list(self._instances.values()):
                    if devices.intersection(other.server.settings["CUDA_VISIBLE_DEVICES"].split(",")):
                        self._stop(other)
            interruption.check()
            instance = self._start(server)
            interruption.bind(instance)
        # Do not hold the state lock during model loading. Release can be queued now.
        return self._wait_ready(instance, interruption.check)

    def _cancel(self, instance: Instance, task_id: str):
        try:
            instance.client.request("DELETE", f"/v1/tasks/{task_id}")
            deadline = time.monotonic() + instance.server.settings["cancel_grace"]
            while time.monotonic() < deadline and instance.process.poll() is None:
                status = instance.client.request("GET", "/v1/service/status")
                # A task's CANCELLED flag alone does NOT mean CUDA work has stopped.
                if status.get("service_status") == "idle" and not status.get("current_task"):
                    break
                time.sleep(0.1)
        except Exception:
            log.debug("Cancel endpoint unavailable; stopping owned process group", exc_info=True)

    def run(self, server: ServerSettings | ServerRef, request: dict, *, interrupt=lambda: None, release_after=False, finalize=lambda path: path, media_type="video"):
        """Run once, finalize the file under the lease, then optionally release GPU memory."""
        extensions = {"video": ".mp4", "image": ".png"}
        if media_type not in extensions:
            raise ValueError("Expected video or image media_type")
        while not self._lock.acquire(timeout=0.1):
            interrupt()
        instance = None
        task_id = None
        attempted_key = None
        interruption = None
        try:
            interrupt()
            interruption = _RunInterrupt(interrupt)
            if self._closed:
                raise RuntimeError("Manager is closed")
            if isinstance(server, ServerRef):
                server = server.resolve()
            if {"task_id", "save_result_path", "reuse", "presigned_url"} & request.keys():
                raise ValueError("Task IDs, output paths and reuse are managed by the bridge")
            # Applies to reuse too: native Comfy nodes may have loaded weights in the meantime.
            self.before_inference()
            attempted_key = server.service_key
            instance = self._ensure(server, interruption)
            s = server.settings
            task_id = uuid.uuid4().hex
            relative = task_id + extensions[media_type]
            output_root = instance.directory / "cache" / "outputs"
            expected = (output_root / relative).resolve()
            payload = dict(request, task_id=task_id, save_result_path=relative)
            try:
                response = instance.client.request("POST", f"/v1/tasks/{media_type}/", payload)
                if response.get("task_id") != task_id:
                    raise RuntimeError("Task submission returned an unexpected ID")
            except TRANSIENT:
                # An accepted POST may lose its response. Query exactly the same ID; never resubmit.
                instance.client.request("GET", f"/v1/tasks/{task_id}/status")
            deadline = time.monotonic() + s["task_timeout"]
            failures = 0
            while time.monotonic() < deadline:
                interruption.check()
                if instance.process.poll() is not None:
                    raise RuntimeError(f"LightX2V exited; see {instance.directory / 'server.log'}")
                try:
                    result = instance.client.request("GET", f"/v1/tasks/{task_id}/status")
                    failures = 0
                except TRANSIENT:
                    failures += 1
                    if failures >= 5:
                        raise
                    time.sleep(s["poll_interval"])
                    continue
                status = result.get("status")
                if status == "completed":
                    service = instance.client.request("GET", "/v1/service/status")
                    if service.get("service_status") != "idle" or service.get("current_task"):
                        time.sleep(s["poll_interval"])
                        continue
                    reported = result.get("save_result_path")
                    if not reported:
                        raise RuntimeError("Server completed without a result file")
                    actual = Path(reported)
                    actual = actual if actual.is_absolute() else output_root / actual
                    if actual.resolve() != expected or not expected.is_file() or expected.stat().st_size == 0:
                        raise RuntimeError("Missing or unexpected result path: " + str(actual))
                    value = finalize(expected)
                    interruption.close()
                    interruption.check()
                    if release_after:
                        instance.release_requested = True
                    return value
                if status in ("failed", "cancelled"):
                    raise RuntimeError(f"LightX2V task {status}: {result.get('error') or result.get('error_type') or task_id}")
                if status not in ("pending", "processing"):
                    raise RuntimeError("Unknown task status: " + str(status))
                time.sleep(s["poll_interval"])
            raise TimeoutError("LightX2V task timed out")
        except BaseException:
            if interruption is not None:
                interruption.close()
            cancelled = interruption is not None and interruption.error is not None
            if not cancelled and instance is not None and task_id is not None:
                self._cancel(instance, task_id)
            try:
                with self._state_lock:
                    affected = self._instances.get(attempted_key)
                    if affected and affected.busy:
                        try:
                            self._stop(affected, force=cancelled)
                        finally:
                            affected.busy = False
                    self._release_slot()
            except Exception:
                # Keep the poisoned instance, so the next job cannot start over unknown GPU work.
                log.exception("Could not confirm process cleanup; manager remains unavailable")
            if cancelled:
                interruption.check()
            raise
        finally:
            try:
                if interruption is not None:
                    interruption.close()
                with self._state_lock:
                    if instance is not None and self._instances.get(instance.server.service_key) is instance:
                        instance.last_used = time.monotonic()
                        instance.busy = False
                        if instance.release_requested and not instance.cleanup_failed:
                            self._stop(instance)
            finally:
                self._lock.release()

    def config_snapshot(self, service_key: str) -> dict | None:
        """Read the immutable configuration loaded by an owned service, without starting it."""
        with self._state_lock:
            instance = self._instances.get(service_key)
            if instance is None:
                return None
            return {
                "config_json": instance.server.settings["config_json"],
                "content": json.dumps(json.loads(instance.server.config_json), ensure_ascii=False, indent=2),
                "fingerprint": instance.server.fingerprint,
                "busy": instance.busy,
                "instance_id": instance.directory.name,
            }

    def snapshots(self) -> list[dict]:
        with self._state_lock:
            result = []
            for key, instance in self._instances.items():
                s = instance.server.settings
                elapsed = max(0, time.monotonic() - instance.last_used)
                result.append(
                    dict(
                        instance.state(),
                        service_key=key,
                        instance_id=instance.directory.name,
                        server=instance.server.name,
                        url=instance.client.url,
                        directory=str(instance.directory),
                        CUDA_VISIBLE_DEVICES=s["CUDA_VISIBLE_DEVICES"],
                        busy=instance.busy,
                        release_requested=instance.release_requested,
                        cleanup_failed=instance.cleanup_failed,
                        idle_ttl=s["idle_ttl"],
                        idle_seconds=0 if instance.busy else elapsed,
                        expires_in=None if instance.busy or not s["idle_ttl"] else max(0, s["idle_ttl"] - elapsed),
                    )
                )
            return result

    def snapshot(self) -> dict:
        """Return one service snapshot, or the pool state when empty/shared."""
        items = self.snapshots()
        return items[0] if len(items) == 1 else {"phase": "running" if items else "stopped", "instances": items}

    def release(self, service_key=None):
        """Release idle services now; active leases finish before their service stops."""
        with self._state_lock:
            if service_key is None:
                items = list(self._instances.values())
            else:
                instance = self._instances.get(service_key)
                items = [instance] if instance else []
            pending = False
            for instance in items:
                if instance.busy and not instance.cleanup_failed:
                    instance.release_requested = True
                    pending = True
                else:
                    self._stop(instance)
            return {"status": "release_pending" if pending else "released" if items else "not_running", "service_key": service_key}

    def _idle_loop(self):
        while not self._wake.wait(0.25):
            if not self._state_lock.acquire(blocking=False):
                continue
            try:
                for instance in list(self._instances.values()):
                    ttl = instance.server.settings["idle_ttl"]
                    if not instance.busy and not instance.cleanup_failed and (instance.process.poll() is not None or (ttl > 0 and time.monotonic() - instance.last_used >= ttl)):
                        self._stop(instance)
            except Exception:
                log.exception("Idle cleanup failed")
            finally:
                self._state_lock.release()

    def close(self):
        self._wake.set()
        with self._lock, self._state_lock:
            self._closed = True
            errors = []
            for instance in list(self._instances.values()):
                try:
                    self._stop(instance)
                except Exception as exc:  # noqa: BLE001 - attempt cleanup of every owned process
                    errors.append(exc)
            self._release_slot()
            if errors:
                raise errors[0]
