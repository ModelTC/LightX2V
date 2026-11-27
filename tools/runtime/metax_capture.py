"""MetaX telemetry in the existing gpu_watch sample and ancestry format."""

import json
import re
import subprocess
import time
from pathlib import Path

from gpu_watch import check, lineage


def parse_sample(output, card):
    identity = re.search(r"GPU#(\d+)\s+\S+\s+([0-9a-fA-F:.]+)", output)
    memory = re.search(r"^\s+vram used\s*:\s*(\d+) KB", output, re.MULTILINE)
    utilization = re.search(r"^\s+GPU\s*:\s*(\d+) %", output, re.MULTILINE)
    if not identity or int(identity[1]) != card or not memory or not utilization or "| Process:" not in output:
        raise ValueError("Incomplete mx-smi telemetry or device identity")
    processes = []
    for match in re.finditer(r"^\|\s*(\d+)\s+(\d+)\s+\S+\s+(\d+)\s*\|", output, re.MULTILINE):
        if int(match[1]) != card:
            raise ValueError("mx-smi returned another card's process")
        pid = int(match[2])
        ancestors = lineage(pid)
        if not ancestors:
            raise ValueError(f"GPU PID {pid} is unavailable in this PID namespace")
        processes.append({"pid": pid, "type": "C", "memory_mib": int(match[3]), "ancestors": ancestors})
    if not processes and "no process found" not in output:
        raise ValueError("Unrecognized mx-smi process inventory")
    return {
        "uuid": "MetaX-" + identity[2].lower(),
        "physical_card": card,
        "pci_bus_id": identity[2].lower(),
        "memory_mib": int(memory[1]) / 1024,
        "utilization_pct": int(utilization[1]),
        "processes": processes,
    }


def sample(card, timeout=2.0):
    result = subprocess.run(["mx-smi", "-i", str(card), "--show-memory", "--show-usage", "--show-process"], capture_output=True, text=True, check=True, timeout=timeout)
    return parse_sample(result.stdout, card)


def verify(samples, task, max_gap, idle_memory):
    from types import SimpleNamespace

    return check(SimpleNamespace(samples=Path(samples), task=Path(task), max_gap=max_gap, idle_memory=idle_memory))


def main():
    import argparse
    import signal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--query-timeout", type=float, default=2.0)
    parser.add_argument("--seconds", type=float, default=3500)
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600 or args.interval <= 0 or args.query_timeout <= 0:
        parser.error("Invalid capture budget")
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    deadline = time.monotonic() + args.seconds
    count = 0
    with (args.output / "samples.jsonl").open("x") as stream:
        while True:
            point = {"start": time.monotonic(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
            try:
                gpu = sample(args.card, args.query_timeout)
                point.update(devices=[gpu["uuid"]], gpus=[gpu])
                if args.pid_file.exists():
                    requested_pid = json.loads(args.pid_file.read_text())["pid"]
                    point["task_identity"] = lineage(requested_pid)[:1]
            except (OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
                point["error"] = f"{type(error).__name__}: {error}"
            point["end"] = time.monotonic()
            stream.write(json.dumps(point) + "\n")
            stream.flush()
            temporary = args.output / "latest.tmp"
            temporary.write_text(json.dumps(point) + "\n")
            temporary.replace(args.output / "latest.json")
            count += 1
            if stopped or (args.output / "stop").exists() or time.monotonic() >= deadline:
                break
            time.sleep(args.interval)
    (args.output / "finished.json").write_text(json.dumps({"samples": count}) + "\n")


if __name__ == "__main__":
    main()
