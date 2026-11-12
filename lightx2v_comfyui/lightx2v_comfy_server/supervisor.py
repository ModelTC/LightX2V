"""Own a backend process group; ComfyUI's pipe EOF triggers cleanup even on SIGKILL.

Executed as a script with the ComfyUI interpreter. No Comfy/torch imports.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        if sys.platform != "darwin":
            raise
        # Darwin can return EPERM for a group containing only an unreaped zombie.
        rows = subprocess.check_output(["ps", "-axo", "pgid=,stat="], text=True).splitlines()
        return any(int(parts[0]) == pgid and not parts[1].startswith("Z") for row in rows if len(parts := row.split()) >= 2)
    # Linux init/container PID 1 may leave orphan zombies unreaped. They own no CUDA context.
    if sys.platform.startswith("linux"):
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid and fields[0] != "Z":
                    return True
            except (OSError, ValueError, IndexError):
                continue
        return False
    return True


def signal_group(pgid: int, sig: int):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        if group_alive(pgid):
            raise


def write_state(path: Path, **values):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values), encoding="utf-8")
    os.replace(temporary, path)


def linux_descendants() -> dict:
    """A subreaper retains ownership even when torchrun ranks create new sessions."""
    rows = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            rows[int(entry.name)] = (int(fields[1]), fields[0], fields[19])
        except (FileNotFoundError, ProcessLookupError):
            pass
    owned = {os.getpid()}
    while True:
        more = {pid for pid, row in rows.items() if row[0] in owned}
        if more <= owned:
            break
        owned |= more
    return {pid: rows[pid] for pid in owned - {os.getpid()} if rows[pid][1] != "Z"}


def signal_descendants(sig):
    for pid, row in linux_descendants().items():
        try:
            # Recheck starttime to avoid signalling a PID that has since been recycled.
            current = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if current[19] == row[2]:
                os.kill(pid, sig)
        except (FileNotFoundError, ProcessLookupError):
            pass


def reap(child):
    child.poll()
    if sys.platform.startswith("linux") and child.returncode is not None:
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass


def main(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    state_path = Path(spec["state_path"])
    stop = threading.Event()
    force = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    def watch_parent():
        try:
            # os.read avoids holding Python's buffered-stdin lock during interpreter shutdown.
            while data := os.read(sys.stdin.fileno(), 64):
                if b"K" in data:
                    force.set()
                    break
        finally:
            stop.set()

    threading.Thread(target=watch_parent, daemon=True).start()
    child = None
    try:
        if sys.platform.startswith("linux"):
            # PR_SET_CHILD_SUBREAPER: orphaned ranks/compile workers reparent here, not PID 1.
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(36, 1, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "Cannot enable Linux child subreaper")
        if stop.is_set():
            write_state(state_path, phase="stopped", pid=None)
            return 0
        env = os.environ.copy()
        env.update(spec["env"])
        with open(spec["log_path"], "ab", buffering=0) as logfile:
            child = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=env, stdin=subprocess.DEVNULL, stdout=logfile, stderr=logfile, start_new_session=True, close_fds=True)
        write_state(state_path, phase="running", pid=child.pid)
        while child.poll() is None and not stop.wait(0.1):
            pass
    except BaseException as exc:  # noqa: BLE001 - cleanup also runs on interruption
        write_state(state_path, phase="error", error=str(exc), pid=child.pid if child else None)
    finally:
        if child is not None:
            linux = sys.platform.startswith("linux")
            if not force.is_set():
                signal_group(child.pid, signal.SIGTERM)
                if linux:
                    signal_descendants(signal.SIGTERM)
            deadline = time.monotonic() + spec["stop_grace"]

            def alive():
                return bool(linux_descendants()) if linux else group_alive(child.pid)

            while not force.is_set() and alive() and time.monotonic() < deadline:
                reap(child)
                time.sleep(0.05)
            signal_group(child.pid, signal.SIGKILL)
            if linux:
                signal_descendants(signal.SIGKILL)
            child.wait()
            deadline = time.monotonic() + 5
            while alive() and time.monotonic() < deadline:
                if linux:
                    signal_descendants(signal.SIGKILL)
                reap(child)
                time.sleep(0.05)
            if alive():
                write_state(state_path, phase="cleanup_failed", pid=child.pid)
                # Keep the inherited slot lock until all owned GPU processes actually exit.
                while alive():
                    if linux:
                        signal_descendants(signal.SIGKILL)
                    else:
                        signal_group(child.pid, signal.SIGKILL)
                    reap(child)
                    time.sleep(1)
            reap(child)
            write_state(state_path, phase="stopped", pid=child.pid, returncode=child.returncode)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
