from __future__ import annotations

import argparse
import os
import threading
import time
from collections import deque
from multiprocessing import resource_tracker, shared_memory
from typing import TYPE_CHECKING, Any, Deque

import zmq

if TYPE_CHECKING:
    from lightx2v.disagg.conn import DataReceiver, DataSender


STATUS_FAILED = 0
STATUS_SUCCESS = 4
_SHM_TRACKING_PATCHED = False


def _disable_shared_memory_tracking_for_process():
    """Disable multiprocessing resource_tracker registration for shared_memory.

    Python 3.12 does not expose SharedMemory(track=False). In fail-fast paths where
    processes are terminated quickly, tracker warnings/noise can dominate logs even
    when manual cleanup is performed by sidecar ownership logic.
    """

    global _SHM_TRACKING_PATCHED
    if _SHM_TRACKING_PATCHED:
        return

    original_register = resource_tracker.register
    original_unregister = resource_tracker.unregister

    def _register(name, rtype):
        if rtype == "shared_memory":
            return
        return original_register(name, rtype)

    def _unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return original_unregister(name, rtype)

    resource_tracker.register = _register
    resource_tracker.unregister = _unregister
    _SHM_TRACKING_PATCHED = True


class DataMgrSidecarServer:
    """Controller-managed sidecar server process.

    Services push transfer-state events to this process and pop aggregated events
    through request/reply calls.
    """

    def __init__(self, push_addr: str, req_addr: str):
        _disable_shared_memory_tracking_for_process()
        self.push_addr = str(push_addr)
        self.req_addr = str(req_addr)

        self._input_watch: set[int] = set()
        self._output_watch: set[int] = set()
        self._ready_inputs: Deque[int] = deque()
        self._failed_inputs: Deque[int] = deque()
        self._completed_outputs: Deque[tuple[int, int]] = deque()

        self._total_messages = 0
        self._last_message_ts = time.time()
        self._running = True

        self._transformer_phase2_mgr: Any | None = None
        self._transformer_phase2_rooms: dict[int, dict[str, Any]] = {}
        self._transformer_phase2_output_watch: set[int] = set()
        self._transformer_phase2_last_status: dict[int, int] = {}

    def _mark_activity(self):
        self._total_messages += 1
        self._last_message_ts = time.time()

    def _handle_push(self, msg: dict):
        cmd = str(msg.get("cmd", ""))
        room = int(msg.get("room", -1))

        if cmd == "watch_input" and room >= 0:
            self._input_watch.add(room)
            self._mark_activity()
            return
        if cmd == "unwatch_input" and room >= 0:
            self._input_watch.discard(room)
            self._mark_activity()
            return
        if cmd == "watch_output" and room >= 0:
            self._output_watch.add(room)
            self._mark_activity()
            return
        if cmd == "unwatch_output" and room >= 0:
            self._output_watch.discard(room)
            self._mark_activity()
            return

        if cmd == "input_status" and room >= 0:
            status = int(msg.get("status", STATUS_FAILED))
            self._input_watch.discard(room)
            if status == STATUS_SUCCESS:
                self._ready_inputs.append(room)
            else:
                self._failed_inputs.append(room)
            self._mark_activity()
            return

        if cmd == "output_status" and room >= 0:
            status = int(msg.get("status", STATUS_FAILED))
            self._output_watch.discard(room)
            self._completed_outputs.append((room, status))
            self._mark_activity()
            return

        if cmd == "shutdown":
            self._running = False
            self._mark_activity()

    def _ensure_transformer_phase2_mgr(self):
        if self._transformer_phase2_mgr is not None:
            return self._transformer_phase2_mgr

        from lightx2v.disagg.conn import DataManager, DisaggregationMode, DisaggregationPhase

        self._transformer_phase2_mgr = DataManager(DisaggregationPhase.PHASE2, DisaggregationMode.TRANSFORMER)
        return self._transformer_phase2_mgr

    def _create_shared_memory(self, size: int) -> shared_memory.SharedMemory:
        # Keep lifecycle in this process and avoid resource_tracker duplicate cleanup at shutdown.
        try:
            return shared_memory.SharedMemory(create=True, size=int(size), track=False)
        except TypeError:
            return shared_memory.SharedMemory(create=True, size=int(size))

    def _close_unlink_shared_memory(self, shm: shared_memory.SharedMemory):
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def _cleanup_transformer_phase2_room(self, room: int):
        room = int(room)
        info = self._transformer_phase2_rooms.pop(room, None)
        self._transformer_phase2_output_watch.discard(room)

        mgr = self._transformer_phase2_mgr
        if mgr is not None:
            try:
                mgr.remove(room)
            except Exception:
                pass

        if not isinstance(info, dict):
            return

        shms = info.get("shms")
        if isinstance(shms, list):
            for shm in shms:
                self._close_unlink_shared_memory(shm)

    def _init_transformer_output_room(
        self,
        room: int,
        sender_engine_rank: int,
        receiver_engine_rank: int,
        data_lens: list[int],
        bootstrap_addr: str,
    ) -> dict[str, Any]:
        room = int(room)
        sender_engine_rank = int(sender_engine_rank)
        receiver_engine_rank = int(receiver_engine_rank)
        normalized_lens = [int(v) for v in list(data_lens)]
        if not normalized_lens or any(v <= 0 for v in normalized_lens):
            raise ValueError(f"invalid data_lens for room={room}: {normalized_lens}")

        self._cleanup_transformer_phase2_room(room)
        self._transformer_phase2_last_status.pop(room, None)
        mgr = self._ensure_transformer_phase2_mgr()

        import numpy as np
        import torch

        from lightx2v.disagg.conn import DataArgs, DataSender

        shms: list[shared_memory.SharedMemory] = []
        arrays: list[Any] = []
        tensors: list[Any] = []
        data_ptrs: list[int] = []
        shm_names: list[str] = []

        try:
            for nbytes in normalized_lens:
                shm = self._create_shared_memory(int(nbytes))
                arr = np.ndarray((int(nbytes),), dtype=np.uint8, buffer=shm.buf)
                tensor = torch.from_numpy(arr)
                tensor.zero_()

                shms.append(shm)
                arrays.append(arr)
                tensors.append(tensor)
                data_ptrs.append(int(tensor.data_ptr()))
                shm_names.append(str(shm.name))

            data_args = DataArgs(
                sender_engine_rank=sender_engine_rank,
                receiver_engine_rank=receiver_engine_rank,
                data_ptrs=data_ptrs,
                data_lens=normalized_lens,
                data_item_lens=normalized_lens,
                ib_device=None,
            )
            mgr.init(data_args, room)
            sender = DataSender(mgr, bootstrap_addr, room)

            self._transformer_phase2_rooms[room] = {
                "sender": sender,
                "data_ptrs": data_ptrs,
                "shms": shms,
                "arrays": arrays,
                "tensors": tensors,
            }

            self._mark_activity()
            return {
                "room": room,
                "shm_names": shm_names,
                "data_lens": normalized_lens,
                "host": str(mgr.get_localhost()),
                "session_id": str(mgr.get_session_id()),
            }
        except Exception:
            for shm in shms:
                self._close_unlink_shared_memory(shm)
            try:
                mgr.remove(room)
            except Exception:
                pass
            raise

    def _send_transformer_output_room(self, room: int):
        room = int(room)
        info = self._transformer_phase2_rooms.get(room)
        if not isinstance(info, dict):
            raise KeyError(f"transformer output room not initialized: {room}")

        sender = info.get("sender")
        data_ptrs = info.get("data_ptrs")
        if sender is None or not isinstance(data_ptrs, list):
            raise RuntimeError(f"transformer output room metadata invalid: {room}")

        sender.send(list(data_ptrs))
        self._transformer_phase2_output_watch.add(room)
        self._mark_activity()

    def _get_transformer_output_status(self, room: int) -> int:
        room = int(room)
        info = self._transformer_phase2_rooms.get(room)
        if not isinstance(info, dict):
            return int(self._transformer_phase2_last_status.get(room, STATUS_FAILED))
        sender = info.get("sender")
        if sender is None:
            return int(self._transformer_phase2_last_status.get(room, STATUS_FAILED))
        try:
            return int(sender.poll())
        except Exception:
            return int(self._transformer_phase2_last_status.get(room, STATUS_FAILED))

    def _get_transformer_output_backlog(self) -> dict[str, int]:
        mgr = self._transformer_phase2_mgr
        if mgr is None:
            return {
                "request_pool": 0,
                "waiting_pool": 0,
                "request_status": 0,
            }
        try:
            data = mgr.get_backlog_counts()
        except Exception:
            data = {}
        return {
            "request_pool": int(data.get("request_pool", 0)),
            "waiting_pool": int(data.get("waiting_pool", 0)),
            "request_status": int(data.get("request_status", 0)),
        }

    def _poll_transformer_output_watch(self):
        for room in list(self._transformer_phase2_output_watch):
            status_val = self._get_transformer_output_status(room)
            if status_val in (STATUS_SUCCESS, STATUS_FAILED):
                self._transformer_phase2_output_watch.discard(room)
                self._transformer_phase2_last_status[int(room)] = int(status_val)
                self._completed_outputs.append((int(room), int(status_val)))
                self._cleanup_transformer_phase2_room(room)
                self._mark_activity()

    def _release_transformer_phase2_mgr(self):
        for room in list(self._transformer_phase2_rooms.keys()):
            self._cleanup_transformer_phase2_room(room)
        mgr = self._transformer_phase2_mgr
        self._transformer_phase2_mgr = None
        if mgr is not None:
            try:
                mgr.release()
            except Exception:
                pass

    def _get_pending_counts(self) -> dict[str, int]:
        transformer_backlog = self._get_transformer_output_backlog()
        output_watch = len(self._output_watch) + len(self._transformer_phase2_output_watch)
        return {
            "input_watch": len(self._input_watch),
            "output_watch": output_watch,
            "ready_inputs": len(self._ready_inputs),
            "failed_inputs": len(self._failed_inputs),
            "completed_outputs": len(self._completed_outputs),
            "transformer_request_pool": int(transformer_backlog.get("request_pool", 0)),
            "transformer_waiting_pool": int(transformer_backlog.get("waiting_pool", 0)),
            "transformer_active_rooms": len(self._transformer_phase2_rooms),
        }

    def _handle_req(self, req: dict) -> dict:
        cmd = str(req.get("cmd", ""))

        if cmd == "ping":
            return {"ok": True}

        if cmd == "get_pending_counts":
            return {"ok": True, "data": self._get_pending_counts()}

        if cmd == "get_stats":
            counts = self._get_pending_counts()
            return {
                "ok": True,
                "data": {
                    **counts,
                    "total_messages": int(self._total_messages),
                    "last_message_ts": float(self._last_message_ts),
                },
            }

        if cmd == "init_transformer_output_room":
            try:
                room = int(req.get("room", -1))
                sender_engine_rank = int(req.get("sender_engine_rank", -1))
                receiver_engine_rank = int(req.get("receiver_engine_rank", -1))
                data_lens_raw = req.get("data_lens")
                bootstrap_addr = str(req.get("bootstrap_addr", "127.0.0.1"))
                if room < 0 or sender_engine_rank < 0 or receiver_engine_rank < 0:
                    raise ValueError("room/sender_engine_rank/receiver_engine_rank must be non-negative")
                if not isinstance(data_lens_raw, list):
                    raise ValueError("data_lens must be a list")
                data = self._init_transformer_output_room(
                    room=room,
                    sender_engine_rank=sender_engine_rank,
                    receiver_engine_rank=receiver_engine_rank,
                    data_lens=[int(v) for v in data_lens_raw],
                    bootstrap_addr=bootstrap_addr,
                )
                return {"ok": True, "data": data}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        if cmd == "send_transformer_output_room":
            try:
                room = int(req.get("room", -1))
                if room < 0:
                    raise ValueError("room must be non-negative")
                self._send_transformer_output_room(room)
                return {"ok": True, "data": True}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        if cmd == "get_transformer_output_status":
            room = int(req.get("room", -1))
            if room < 0:
                return {"ok": False, "error": "room must be non-negative"}
            return {"ok": True, "data": self._get_transformer_output_status(room)}

        if cmd == "remove_transformer_output_room":
            room = int(req.get("room", -1))
            if room < 0:
                return {"ok": False, "error": "room must be non-negative"}
            self._cleanup_transformer_phase2_room(room)
            self._mark_activity()
            return {"ok": True, "data": True}

        if cmd == "get_transformer_output_backlog":
            return {"ok": True, "data": self._get_transformer_output_backlog()}

        if cmd == "get_transformer_output_identity":
            mgr = self._transformer_phase2_mgr
            if mgr is None:
                return {"ok": False, "error": "transformer phase2 manager not initialized"}
            return {
                "ok": True,
                "data": {
                    "host": str(mgr.get_localhost()),
                    "session_id": str(mgr.get_session_id()),
                },
            }

        if cmd == "pop_ready_inputs":
            items = list(self._ready_inputs)
            self._ready_inputs.clear()
            return {"ok": True, "data": items}

        if cmd == "pop_failed_inputs":
            items = list(self._failed_inputs)
            self._failed_inputs.clear()
            return {"ok": True, "data": items}

        if cmd == "pop_completed_outputs":
            items = list(self._completed_outputs)
            self._completed_outputs.clear()
            return {"ok": True, "data": items}

        if cmd == "shutdown":
            self._running = False
            self._mark_activity()
            return {"ok": True}

        return {"ok": False, "error": f"unknown command: {cmd}"}

    def run_forever(self):
        context = zmq.Context()
        pull = context.socket(zmq.PULL)
        rep = context.socket(zmq.REP)

        pull.bind(self.push_addr)
        rep.bind(self.req_addr)

        poller = zmq.Poller()
        poller.register(pull, zmq.POLLIN)
        poller.register(rep, zmq.POLLIN)

        try:
            while self._running:
                events = dict(poller.poll(timeout=100))
                if pull in events:
                    try:
                        self._handle_push(pull.recv_pyobj())
                    except Exception:
                        pass

                if rep in events:
                    try:
                        reply = self._handle_req(rep.recv_pyobj())
                    except Exception as exc:
                        reply = {"ok": False, "error": str(exc)}
                    rep.send_pyobj(reply)

                self._poll_transformer_output_watch()
        finally:
            self._release_transformer_phase2_mgr()
            pull.close(0)
            rep.close(0)
            context.term()


class _LocalDataMgrSidecar:
    """Fallback local sidecar used when controller-managed endpoints are absent."""

    def __init__(self, poll_interval_s: float = 0.01):
        self.poll_interval_s = max(float(poll_interval_s), 0.001)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False

        self._input_watch: dict[int, DataReceiver] = {}
        self._output_watch: dict[int, DataSender] = {}

        self._ready_inputs: Deque[int] = deque()
        self._failed_inputs: Deque[int] = deque()
        self._completed_outputs: Deque[tuple[int, int]] = deque()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            self._started = True
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="data-mgr-sidecar-local",
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def stop(self):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._started = False

    def watch_input(self, room: int, receiver: DataReceiver):
        if not self._started:
            self.start()
        with self._lock:
            self._input_watch[int(room)] = receiver

    def unwatch_input(self, room: int):
        with self._lock:
            self._input_watch.pop(int(room), None)

    def watch_output(self, room: int, sender: DataSender):
        if not self._started:
            self.start()
        with self._lock:
            self._output_watch[int(room)] = sender

    def unwatch_output(self, room: int):
        with self._lock:
            self._output_watch.pop(int(room), None)

    def pop_ready_inputs(self) -> list[int]:
        with self._lock:
            items = list(self._ready_inputs)
            self._ready_inputs.clear()
            return items

    def pop_failed_inputs(self) -> list[int]:
        with self._lock:
            items = list(self._failed_inputs)
            self._failed_inputs.clear()
            return items

    def pop_completed_outputs(self) -> list[tuple[int, int]]:
        with self._lock:
            items = list(self._completed_outputs)
            self._completed_outputs.clear()
            return items

    def get_pending_counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "input_watch": len(self._input_watch),
                "output_watch": len(self._output_watch),
                "ready_inputs": len(self._ready_inputs),
                "failed_inputs": len(self._failed_inputs),
                "completed_outputs": len(self._completed_outputs),
            }

    def init_transformer_output_room(
        self,
        room: int,
        sender_engine_rank: int,
        receiver_engine_rank: int,
        data_lens: list[int],
        bootstrap_addr: str,
    ) -> dict[str, Any] | None:
        return None

    def send_transformer_output_room(self, room: int) -> bool:
        return False

    def get_transformer_output_status(self, room: int) -> int:
        return STATUS_FAILED

    def remove_transformer_output_room(self, room: int) -> bool:
        return False

    def get_transformer_output_backlog(self) -> dict[str, int]:
        return {
            "request_pool": 0,
            "waiting_pool": 0,
            "request_status": 0,
        }

    def get_transformer_output_identity(self, room: int | None = None) -> dict[str, Any] | None:
        return None

    def _run(self):
        while not self._stop_event.is_set():
            with self._lock:
                input_items = list(self._input_watch.items())
                output_items = list(self._output_watch.items())

            if not input_items and not output_items:
                time.sleep(self.poll_interval_s)
                continue

            for room, receiver in input_items:
                try:
                    status = receiver.poll()
                except Exception:
                    status = STATUS_FAILED

                status_val = int(status)
                if status_val == STATUS_SUCCESS:
                    with self._lock:
                        self._input_watch.pop(room, None)
                        self._ready_inputs.append(room)
                elif status_val == STATUS_FAILED:
                    with self._lock:
                        self._input_watch.pop(room, None)
                        self._failed_inputs.append(room)

            for room, sender in output_items:
                try:
                    status = sender.poll()
                except Exception:
                    status = STATUS_FAILED

                status_val = int(status)
                if status_val in (STATUS_SUCCESS, STATUS_FAILED):
                    with self._lock:
                        self._output_watch.pop(room, None)
                        self._completed_outputs.append((room, status_val))

            time.sleep(self.poll_interval_s)


class _RemoteDataMgrSidecarClient:
    """Service-side client for controller-managed sidecar process."""

    def __init__(self, push_addr: str, req_addr: str, poll_interval_s: float = 0.01):
        self.push_addr = str(push_addr)
        self.req_addr = str(req_addr)
        self.poll_interval_s = max(float(poll_interval_s), 0.001)

        self._context = zmq.Context.instance()
        self._push = self._context.socket(zmq.PUSH)
        self._push.connect(self.push_addr)

        self._req = self._context.socket(zmq.REQ)
        self._req.connect(self.req_addr)
        self._req.setsockopt(zmq.RCVTIMEO, 1500)
        self._req.setsockopt(zmq.SNDTIMEO, 1500)

        self._req_lock = threading.Lock()
        self._watch_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False

        self._input_watch: dict[int, DataReceiver] = {}
        self._output_watch: dict[int, DataSender] = {}

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            self._started = True
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="data-mgr-sidecar-remote-client", daemon=True)
        self._thread.start()
        self._started = True

    def stop(self):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._started = False

        try:
            with self._watch_lock:
                rooms_in = list(self._input_watch.keys())
                rooms_out = list(self._output_watch.keys())
                self._input_watch.clear()
                self._output_watch.clear()
            for room in rooms_in:
                self._push_cmd({"cmd": "unwatch_input", "room": int(room)})
            for room in rooms_out:
                self._push_cmd({"cmd": "unwatch_output", "room": int(room)})
        except Exception:
            pass

    def watch_input(self, room: int, receiver: DataReceiver):
        if not self._started:
            self.start()
        room = int(room)
        with self._watch_lock:
            self._input_watch[room] = receiver
        self._push_cmd({"cmd": "watch_input", "room": room})

    def unwatch_input(self, room: int):
        room = int(room)
        with self._watch_lock:
            self._input_watch.pop(room, None)
        self._push_cmd({"cmd": "unwatch_input", "room": room})

    def watch_output(self, room: int, sender: DataSender):
        if not self._started:
            self.start()
        room = int(room)
        with self._watch_lock:
            self._output_watch[room] = sender
        self._push_cmd({"cmd": "watch_output", "room": room})

    def unwatch_output(self, room: int):
        room = int(room)
        with self._watch_lock:
            self._output_watch.pop(room, None)
        self._push_cmd({"cmd": "unwatch_output", "room": room})

    def pop_ready_inputs(self) -> list[int]:
        data = self._req_cmd("pop_ready_inputs")
        if isinstance(data, list):
            return [int(v) for v in data]
        return []

    def pop_failed_inputs(self) -> list[int]:
        data = self._req_cmd("pop_failed_inputs")
        if isinstance(data, list):
            return [int(v) for v in data]
        return []

    def pop_completed_outputs(self) -> list[tuple[int, int]]:
        data = self._req_cmd("pop_completed_outputs")
        if not isinstance(data, list):
            return []
        items: list[tuple[int, int]] = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                items.append((int(item[0]), int(item[1])))
        return items

    def get_pending_counts(self) -> dict[str, int]:
        data = self._req_cmd("get_pending_counts")
        if not isinstance(data, dict):
            return {
                "input_watch": 0,
                "output_watch": 0,
                "ready_inputs": 0,
                "failed_inputs": 0,
                "completed_outputs": 0,
            }
        return {
            "input_watch": int(data.get("input_watch", 0)),
            "output_watch": int(data.get("output_watch", 0)),
            "ready_inputs": int(data.get("ready_inputs", 0)),
            "failed_inputs": int(data.get("failed_inputs", 0)),
            "completed_outputs": int(data.get("completed_outputs", 0)),
        }

    def _push_cmd(self, cmd: dict):
        try:
            self._push.send_pyobj(cmd)
        except Exception:
            pass

    def _req_cmd(self, cmd: str, payload: dict[str, Any] | None = None):
        try:
            req_payload: dict[str, Any] = {"cmd": str(cmd)}
            if isinstance(payload, dict):
                req_payload.update(payload)
            with self._req_lock:
                self._req.send_pyobj(req_payload)
                reply = self._req.recv_pyobj()
            if isinstance(reply, dict) and reply.get("ok", False):
                return reply.get("data")
            return None
        except Exception:
            return None

    def init_transformer_output_room(
        self,
        room: int,
        sender_engine_rank: int,
        receiver_engine_rank: int,
        data_lens: list[int],
        bootstrap_addr: str,
    ) -> dict[str, Any] | None:
        data = self._req_cmd(
            "init_transformer_output_room",
            {
                "room": int(room),
                "sender_engine_rank": int(sender_engine_rank),
                "receiver_engine_rank": int(receiver_engine_rank),
                "data_lens": [int(v) for v in data_lens],
                "bootstrap_addr": str(bootstrap_addr),
            },
        )
        if isinstance(data, dict):
            return data
        return None

    def send_transformer_output_room(self, room: int) -> bool:
        data = self._req_cmd("send_transformer_output_room", {"room": int(room)})
        return bool(data)

    def get_transformer_output_status(self, room: int) -> int:
        data = self._req_cmd("get_transformer_output_status", {"room": int(room)})
        if data is None:
            return STATUS_FAILED
        try:
            return int(data)
        except Exception:
            return STATUS_FAILED

    def remove_transformer_output_room(self, room: int) -> bool:
        data = self._req_cmd("remove_transformer_output_room", {"room": int(room)})
        return bool(data)

    def get_transformer_output_backlog(self) -> dict[str, int]:
        data = self._req_cmd("get_transformer_output_backlog")
        if not isinstance(data, dict):
            return {
                "request_pool": 0,
                "waiting_pool": 0,
                "request_status": 0,
            }
        return {
            "request_pool": int(data.get("request_pool", 0)),
            "waiting_pool": int(data.get("waiting_pool", 0)),
            "request_status": int(data.get("request_status", 0)),
        }

    def get_transformer_output_identity(self, room: int | None = None) -> dict[str, Any] | None:
        payload: dict[str, Any] = {}
        if room is not None:
            payload["room"] = int(room)
        data = self._req_cmd("get_transformer_output_identity", payload)
        if isinstance(data, dict):
            return {
                "host": str(data.get("host", "")),
                "session_id": str(data.get("session_id", "")),
            }
        return None

    def _run(self):
        while not self._stop_event.is_set():
            with self._watch_lock:
                input_items = list(self._input_watch.items())
                output_items = list(self._output_watch.items())

            if not input_items and not output_items:
                time.sleep(self.poll_interval_s)
                continue

            for room, receiver in input_items:
                try:
                    status = receiver.poll()
                except Exception:
                    status = STATUS_FAILED

                status_val = int(status)
                if status_val in (STATUS_SUCCESS, STATUS_FAILED):
                    with self._watch_lock:
                        self._input_watch.pop(room, None)
                    self._push_cmd(
                        {
                            "cmd": "input_status",
                            "room": int(room),
                            "status": status_val,
                        }
                    )

            for room, sender in output_items:
                try:
                    status = sender.poll()
                except Exception:
                    status = STATUS_FAILED

                status_val = int(status)
                if status_val in (STATUS_SUCCESS, STATUS_FAILED):
                    with self._watch_lock:
                        self._output_watch.pop(room, None)
                    self._push_cmd(
                        {
                            "cmd": "output_status",
                            "room": int(room),
                            "status": status_val,
                        }
                    )

            time.sleep(self.poll_interval_s)


class DataMgrSidecar:
    """Service-facing sidecar facade.

    If controller-side endpoints exist, use controller-managed remote sidecar.
    Otherwise fallback to in-process local sidecar for standalone runs.
    """

    def __init__(self, poll_interval_s: float = 0.01):
        push_addr = str(os.getenv("LIGHTX2V_SIDECAR_PUSH_ADDR", "")).strip()
        req_addr = str(os.getenv("LIGHTX2V_SIDECAR_REQ_ADDR", "")).strip()

        if push_addr and req_addr:
            self._impl = _RemoteDataMgrSidecarClient(push_addr=push_addr, req_addr=req_addr, poll_interval_s=poll_interval_s)
        else:
            self._impl = _LocalDataMgrSidecar(poll_interval_s=poll_interval_s)

    def start(self):
        self._impl.start()

    def stop(self):
        self._impl.stop()

    def watch_input(self, room: int, receiver: DataReceiver):
        self._impl.watch_input(room, receiver)

    def unwatch_input(self, room: int):
        self._impl.unwatch_input(room)

    def watch_output(self, room: int, sender: DataSender):
        self._impl.watch_output(room, sender)

    def unwatch_output(self, room: int):
        self._impl.unwatch_output(room)

    def pop_ready_inputs(self) -> list[int]:
        return self._impl.pop_ready_inputs()

    def pop_failed_inputs(self) -> list[int]:
        return self._impl.pop_failed_inputs()

    def pop_completed_outputs(self) -> list[tuple[int, int]]:
        return self._impl.pop_completed_outputs()

    def get_pending_counts(self) -> dict[str, int]:
        return self._impl.get_pending_counts()

    def init_transformer_output_room(
        self,
        room: int,
        sender_engine_rank: int,
        receiver_engine_rank: int,
        data_lens: list[int],
        bootstrap_addr: str,
    ) -> dict[str, Any] | None:
        return self._impl.init_transformer_output_room(
            room=room,
            sender_engine_rank=sender_engine_rank,
            receiver_engine_rank=receiver_engine_rank,
            data_lens=data_lens,
            bootstrap_addr=bootstrap_addr,
        )

    def send_transformer_output_room(self, room: int) -> bool:
        return self._impl.send_transformer_output_room(room)

    def get_transformer_output_status(self, room: int) -> int:
        return self._impl.get_transformer_output_status(room)

    def remove_transformer_output_room(self, room: int) -> bool:
        return self._impl.remove_transformer_output_room(room)

    def get_transformer_output_backlog(self) -> dict[str, int]:
        return self._impl.get_transformer_output_backlog()

    def get_transformer_output_identity(self, room: int | None = None) -> dict[str, Any] | None:
        return self._impl.get_transformer_output_identity(room)


def _run_server_from_cli():
    parser = argparse.ArgumentParser(description="Run DataMgr sidecar server process")
    parser.add_argument("--push-addr", type=str, required=True)
    parser.add_argument("--req-addr", type=str, required=True)
    args = parser.parse_args()

    server = DataMgrSidecarServer(push_addr=args.push_addr, req_addr=args.req_addr)
    server.run_forever()


if __name__ == "__main__":
    _run_server_from_cli()
