"""One CPU supervisor per platform node, bootstrapped by torchrun."""

import argparse
import concurrent.futures
import datetime
import ipaddress
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from lightx2v_rl.paths import REPOSITORY_ROOT

TORCHRUN_VARIABLES = {
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "ROLE_NAME",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NODE_RANK",
}
RECIPES = {f"qwen_image21_turbo_{algorithm}_{nodes * 8}gpu": nodes for algorithm in ("geo", "cps", "flow_grpo") for nodes in (4, 5)}


def recipe_node_count(env):
    recipe = env.get("LIGHTX2V_RL_CONFIG", "qwen_image21_turbo_geo_40gpu")
    if recipe not in RECIPES:
        raise ValueError("The platform launcher requires a qwen_image21_turbo_*_32gpu or *_40gpu recipe.")
    return RECIPES[recipe]


def validate_topology(env):
    nodes = recipe_node_count(env)
    try:
        rank = int(env["RANK"])
        valid = int(env["WORLD_SIZE"]) == nodes and int(env["LOCAL_WORLD_SIZE"]) == 1
        valid = valid and int(env["LOCAL_RANK"]) == 0 and rank in range(nodes)
        port = int(env["MASTER_PORT"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Launch with torchrun --nnodes={nodes} --nproc-per-node=1 and a rendezvous endpoint.") from exc
    if not valid:
        raise ValueError(f"The selected recipe requires WORLD_SIZE={nodes}, LOCAL_WORLD_SIZE=1 and LOCAL_RANK=0.")
    if not env.get("MASTER_ADDR") or not 0 < port < 65536 or port == 6379:
        raise ValueError("MASTER_ADDR and MASTER_PORT are required; the rendezvous port must differ from Ray's 6379.")
    return rank


def validate_addresses(addresses, num_nodes=4):
    if len(addresses) != num_nodes or len(set(addresses)) != num_nodes:
        raise ValueError(f"Expected {num_nodes} distinct node IPv4 addresses (one supervisor per node).")
    for value in addresses:
        address = ipaddress.IPv4Address(value)
        if address.is_loopback or address.is_unspecified or address.is_multicast:
            raise ValueError(f"Node address must be routable between nodes: {value}")
    return list(addresses)


def discover_address(env):
    address = env.get("LIGHTX2V_NODE_IP")
    if not address:
        destination = socket.gethostbyname(env["MASTER_ADDR"])
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect((destination, int(env["MASTER_PORT"])))
            address = connection.getsockname()[0]
    parsed = ipaddress.IPv4Address(address)
    if parsed.is_loopback or parsed.is_unspecified or parsed.is_multicast:
        raise ValueError(f"Cannot discover a routable node IPv4 address: {address}; set LIGHTX2V_NODE_IP per node.")
    # A configured override must actually belong to this host.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.bind((address, 0))
    return address


def child_environment(env, addresses, rank):
    validate_addresses(addresses, recipe_node_count(env))
    result = {key: value for key, value in env.items() if key not in TORCHRUN_VARIABLES and not key.startswith("TORCHELASTIC_")}
    result.update(
        {
            "LIGHTX2V_HEAD_IP": addresses[0],
            "LIGHTX2V_NODE_IP": addresses[rank],
            "LIGHTX2V_REWARD_HOST": addresses[-1],
            "LIGHTX2V_RL_CONFIG": env.get("LIGHTX2V_RL_CONFIG", "qwen_image21_turbo_geo_40gpu"),
            "RAY_ADDRESS": f"{addresses[0]}:6379",
        }
    )
    return result


class OwnedProcesses:
    """Track descendants by PID and creation time, including detached Ray children."""

    def __init__(self, root, env, overrides, log_dir):
        self.root = Path(root)
        self.env = env
        self.overrides = list(overrides)
        self.log_dir = Path(log_dir)
        self.children = {}
        self.descendants = {}
        self.streams = []

    def command(self, role):
        scripts = self.root / "lightx2v_rl" / "scripts"
        if role == "reward":
            return ["bash", str(scripts / "rewards" / "start_qwen21_reward_pool.sh")]
        return ["bash", str(scripts / "train" / "qwen21_cluster.sh"), role, *(self.overrides if role == "train" else [])]

    def start(self, role):
        if role in self.children:
            raise RuntimeError(f"Refusing to start a second {role} process.")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{role}.log"
        stream = None
        command = self.command(role)
        environment = self.env
        if role == "train":
            # tee drains both streams continuously and finishes before bash
            # exposes its exit status. Arguments remain positional, never shell
            # source; pipefail preserves a failing training command's status.
            command = ["bash", "-o", "pipefail", "-c", 'rl_train_log=$1; shift; "$@" 2>&1 | tee -- "$rl_train_log"', "lightx2v-train-tee", str(log_path), *command]
            environment = dict(self.env)
            environment.setdefault("PYTHONUNBUFFERED", "1")
        else:
            stream = log_path.open("w")
            self.streams.append(stream)
        self.children[role] = subprocess.Popen(
            command,
            cwd=self.root,
            env=environment,
            start_new_session=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        print(f"Started {role}, pid={self.children[role].pid}; log={log_path}" + (" (also platform stdout)" if role == "train" else ""), flush=True)
        self._remember_children()

    def _remember_children(self):
        import psutil

        for child in self.children.values():
            if child.poll() is not None:
                continue
            try:
                parent = psutil.Process(child.pid)
                if child.pid in self.descendants and parent.create_time() != self.descendants[child.pid]:
                    continue
                for process in [parent, *parent.children(recursive=True)]:
                    self.descendants[process.pid] = process.create_time()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def poll(self, role):
        self._remember_children()
        return self.children[role].poll()

    def close(self):
        import psutil

        self._remember_children()
        # Give the reward supervisor and Ray --block a chance to stop their own children.
        for child in self.children.values():
            if child.poll() is None:
                child.terminate()
        live = []
        for pid, created in self.descendants.items():
            try:
                process = psutil.Process(pid)
                if process.create_time() == created:
                    live.append(process)
            except psutil.NoSuchProcess:
                continue
        _, remaining = psutil.wait_procs(live, timeout=12)
        for process in remaining:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, remaining = psutil.wait_procs(remaining, timeout=3)
        for process in remaining:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        for child in self.children.values():
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        for stream in self.streams:
            stream.close()


def service_failure_message(role, code, log_dir=None):
    message = f"Training exited with status {code}." if role == "train" else f"{role} exited unexpectedly ({code}); inspect its service log."
    if log_dir is None:
        return message
    path = Path(log_dir) / f"{role}.log"
    message += f"\nLog: {path}"
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 4096))
            tail = stream.read(4096).decode("utf-8", errors="replace")
        tail = "\n".join(tail.splitlines()[-30:]).strip()
        if tail:
            message += f"\nLast service output:\n{tail}"
    except OSError as exc:
        message += f"\nCannot read service log: {exc}"
    return message


@contextmanager
def _ignore_cleanup_signals():
    """Let owned children finish cleanup when torchrun cancels sibling ranks."""
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, signal.SIG_IGN)
    try:
        yield
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def _close_resources(*callbacks):
    """Attempt all cleanup while keeping an existing training failure primary."""
    original_failure = sys.exc_info()[0] is not None
    failures = []
    with _ignore_cleanup_signals():
        for callback in callbacks:
            try:
                callback()
            except Exception as exc:
                failures.append(exc)
                print(f"Platform cleanup error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    if failures and not original_failure:
        raise failures[0]


def supervise(rank, processes, exchange, *, head_ready, cluster_ready, reward_ready, num_nodes=4, startup_timeout=1800, interval=1.0, clock=time.monotonic, sleep=time.sleep):
    """All ranks exchange a small CPU status record, including during long training."""
    deadline = clock() + startup_timeout
    state = dict(error=None, head_ready=False, ready=False, training_started=False, done=False)
    reward_rank = num_nodes - 1
    service = "head" if rank == 0 else "reward" if rank == reward_rank else None
    service_started = False
    train_started = False
    next_status = clock() + 30
    try:
        while True:
            try:
                if service and not service_started:
                    processes.start(service)
                    service_started = True
                if service_started:
                    code = processes.poll(service)
                    if code is not None:
                        raise RuntimeError(service_failure_message(service, code, getattr(processes, "log_dir", None)))
                if rank == 0:
                    if not state["head_ready"]:
                        state["head_ready"] = head_ready()
                    if state["head_ready"] and not state["ready"]:
                        state["ready"] = cluster_ready()
                    if train_started:
                        code = processes.poll("train")
                        if code is not None:
                            if code:
                                raise RuntimeError(
                                    service_failure_message(
                                        "train",
                                        code,
                                        getattr(processes, "log_dir", None),
                                    )
                                )
                            state["done"] = True
                elif rank == reward_rank:
                    if not state["ready"]:
                        state["ready"] = reward_ready()
                else:
                    state["ready"] = service_started
                if not state["training_started"] and clock() >= deadline:
                    raise TimeoutError(f"Cluster/reward startup exceeded {startup_timeout}s.")
            except Exception as exc:
                state["error"] = f"rank {rank}: {type(exc).__name__}: {exc}"

            states = exchange(dict(state))
            errors = [item["error"] for item in states if item["error"]]
            if errors:
                raise RuntimeError("Platform job failed: " + "; ".join(errors))
            if states[0]["done"]:
                print("Training completed; stopping this node's services.", flush=True)
                return
            state["training_started"] = states[0]["training_started"]
            if rank == 0 and not train_started and clock() >= next_status:
                print(f"Waiting for startup: head={states[0]['head_ready']}, cluster={states[0]['ready']}, reward={states[reward_rank]['ready']}", flush=True)
                next_status = clock() + 30
            if 0 < rank < reward_rank and states[0]["head_ready"]:
                service = "worker"
            if rank == 0 and not train_started and all(item["ready"] for item in states):
                try:
                    print(f"All {reward_rank} Ray nodes and eight reward services are ready; starting one training driver.", flush=True)
                    processes.start("train")
                    train_started = state["training_started"] = True
                except Exception as exc:
                    state["error"] = f"rank 0: {type(exc).__name__}: {exc}"
            sleep(interval)
    finally:
        _close_resources(processes.close)


def head_is_ready(address):
    try:
        with socket.create_connection((address, 6379), timeout=1):
            return True
    except OSError:
        return False


def rewards_are_ready(address):
    def ready(port):
        try:
            with urllib.request.urlopen(f"http://{address}:{port}/health", timeout=1) as response:
                return response.status == 200
        except Exception:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        return all(pool.map(ready, range(8791, 8799)))


class RayReadiness:
    """Run the Ray client separately: a stalled connection cannot block heartbeats."""

    def __init__(self, root, env, log_dir):
        self.root, self.env, self.log_dir = Path(root), env, Path(log_dir)
        self.process = None
        self.stream = None
        self.next_probe = 0
        self.started = 0

    def __call__(self):
        now = time.monotonic()
        if self.process is not None:
            code = self.process.poll()
            if code is None and now - self.started < 30:
                return False
            if code is None:
                self.process.kill()
                self.process.wait(timeout=5)
            self.process = None
            if code == 0:
                return True
            self.next_probe = now + 3
        if now < self.next_probe:
            return False
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.stream is None:
            self.stream = (self.log_dir / "ray-readiness.log").open("w")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "lightx2v_rl.deployment.qwen_image.platform", "--probe-ray"],
            cwd=self.root,
            env=self.env,
            stdout=self.stream,
            stderr=subprocess.STDOUT,
        )
        self.started = now
        return False

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)
        if self.stream is not None:
            self.stream.close()


def probe_ray():
    import ray

    ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
    try:
        nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0)]
        expected = recipe_node_count(os.environ) - 1
        return 0 if len(nodes) == expected and all(node["Resources"]["GPU"] == 8 for node in nodes) else 1
    finally:
        ray.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-ray", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("overrides", nargs="*", help="Training key=value overrides, forwarded only on rank 0")
    args = parser.parse_args(argv)
    if args.probe_ray:
        return probe_ray()
    rank = validate_topology(os.environ)
    num_nodes = recipe_node_count(os.environ)
    for item in args.overrides:
        if "=" not in item or item.startswith(("+", "~")):
            raise ValueError("Use plain training key=value overrides.")
    timeout = int(os.environ.get("LIGHTX2V_PLATFORM_STARTUP_TIMEOUT", "1800"))
    if timeout <= 0:
        raise ValueError("LIGHTX2V_PLATFORM_STARTUP_TIMEOUT must be positive.")
    address = discover_address(os.environ)
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("Each platform node must expose eight CUDA GPUs to its single supervisor.")

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Platform supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=180))

    def exchange(record):
        result = [None] * num_nodes
        dist.all_gather_object(result, record)
        return result

    probe = None
    try:
        records = exchange(dict(ip=address, config=os.environ.get("LIGHTX2V_RL_CONFIG", "qwen_image21_turbo_geo_40gpu"), overrides=args.overrides))
        addresses = validate_addresses([record["ip"] for record in records], num_nodes)
        if any((record["config"], record["overrides"]) != (records[0]["config"], records[0]["overrides"]) for record in records):
            raise ValueError("All nodes must receive the same recipe and training overrides.")
        root = REPOSITORY_ROOT
        env = child_environment(os.environ, addresses, rank)
        env["LIGHTX2V_RL_PYTHON"] = sys.executable
        env["PYTHONPATH"] = str(root / "lightx2v_rl") + os.pathsep + env.get("PYTHONPATH", "")
        run_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", os.environ.get("TORCHELASTIC_RUN_ID", "platform"))[:100]
        log_dir = root / "lightx2v_rl" / "outputs" / "platform" / run_id / f"rank{rank}-{os.getpid()}"
        print(f"Platform rank {rank}: {address}; head={addresses[0]}, reward={addresses[-1]}; logs={log_dir}", flush=True)
        processes = OwnedProcesses(root, env, args.overrides, log_dir)
        probe = RayReadiness(root, env, log_dir)
        supervise(
            rank,
            processes,
            exchange,
            head_ready=lambda: head_is_ready(addresses[0]),
            cluster_ready=probe,
            reward_ready=lambda: rewards_are_ready(addresses[-1]),
            num_nodes=num_nodes,
            startup_timeout=timeout,
        )
        return 0
    finally:
        callbacks = [probe.close] if probe is not None else []
        _close_resources(*callbacks, dist.destroy_process_group)


if __name__ == "__main__":
    raise SystemExit(main())
