"""MLU host telemetry using cnmon and the existing gpu_watch resource policy."""

import json
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from gpu_watch import check, lineage


_IDENTITIES = {}


def identity(card, timeout):
    # cnmon -j defaults to writing cnmon_info.json in cwd; always give it an
    # isolated explicit destination so observing never overwrites caller files.
    with tempfile.TemporaryDirectory(prefix="lightx2v-cnmon-") as directory:
        target = Path(directory) / "device.json"
        subprocess.run(["cnmon", "info", "-c", str(card), "-o", str(target)],
                       capture_output=True, text=True, check=True, timeout=timeout, cwd=directory)
        records = json.loads(target.read_text())["Cnmon Info"]
    if len(records) != 1 or records[0]["CardNum"] != card or records[0]["Mode"] != "FULL":
        raise ValueError("Exactly one full physical MLU card is required")
    record = records[0]
    device_uuid = str(uuid.UUID(record["UUID"]))
    pci = record["PCI"]
    bus = f"{pci['Domain ID']}:{pci['Bus num']}:{pci['Device']}.{pci['Function']}".lower()
    return {"uuid": "MLU-" + device_uuid, "pci_bus_id": bus, "physical_card": card}


def parse_sample(output, card, device):
    if device["physical_card"] != card:
        raise ValueError("MLU identity card differs")
    inventory = output.split("| Processes:")
    if len(inventory) != 2:
        raise ValueError("Unavailable cnmon process inventory")
    heading, processes_text = inventory
    match = re.search(r"\|\s*(\d+)\s+(/|\d+)\s+\S+\s+\S+\s*\|\s*([0-9a-fA-F:.]+)\s*\|\s*(\d+)%", heading)
    memory = re.search(r"\|\s*\d+\s+\d+C\s+[0-9.]+ W/\s*[0-9.]+ W\s*\|\s*(\d+) MiB/\s*\d+ MiB\s*\|\s*(\S+)\s+(\S+)\s*\|", heading)
    if not match or not memory or int(match[1]) != card or match[2] != "/" or memory[2] != "FULL":
        raise ValueError("Incomplete cnmon physical-card telemetry or partitioned card")
    if match[3].lower() != device["pci_bus_id"]:
        raise ValueError("MLU PCI identity changed")
    processes = []
    for line in processes_text.splitlines():
        if not re.match(r"\|\s*\d", line):
            continue
        row = re.fullmatch(r"\|\s*(\d+)\s+(/|\d+)\s+(\d+)\s+(.+?)\s+(\d+) MiB\s*\|", line)
        if not row or int(row[1]) != card or row[2] != "/":
            raise ValueError("Unrecognized or partitioned MLU process row")
        pid = int(row[3])
        ancestors = lineage(pid)
        if not ancestors:
            raise ValueError(f"MLU PID {pid} unavailable in host PID namespace")
        processes.append({"pid": pid, "type": "C", "memory_mib": int(row[5]), "ancestors": ancestors})
    if not processes and "No running processes found" not in processes_text:
        raise ValueError("Unrecognized empty cnmon process inventory")
    if processes and "No running processes found" in processes_text:
        raise ValueError("Contradictory cnmon process inventory")
    return {**device, "memory_mib": int(memory[1]), "utilization_pct": int(match[4]), "processes": processes}


def sample(card, timeout=2.0):
    if card not in _IDENTITIES:
        _IDENTITIES[card] = identity(card, timeout)
    result = subprocess.run(["cnmon", "-c", str(card)], capture_output=True, text=True, check=True, timeout=timeout)
    return parse_sample(result.stdout, card, _IDENTITIES[card])


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
