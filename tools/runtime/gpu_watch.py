"""Record host GPU activity and check one task interval; execution belongs to the Agent."""

import argparse
import json
import math
import signal
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path


def lineage(pid):
    ancestors = []
    while pid and pid not in [item[0] for item in ancestors]:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            break
        ancestors.append([pid, int(fields[19])])
        pid = int(fields[1])
    return ancestors


def sample(devices, timeout):
    xml = subprocess.check_output(["nvidia-smi", "-q", "-x", "-i", ",".join(devices)], timeout=timeout)
    gpus = []
    for gpu in ET.fromstring(xml).findall("gpu"):
        apps = gpu.find("processes")
        if gpu.findtext("mig_mode/current_mig") not in ("Disabled", "N/A") or apps is None or (apps.text or "").strip():
            raise ValueError("unsupported MIG or unavailable process inventory")
        processes = []
        for app in apps.findall("process_info"):
            pid = int(app.findtext("pid"))
            if pid <= 0:
                raise ValueError("GPU PID unavailable")
            processes.append({"pid": pid, "type": app.findtext("type"), "ancestors": lineage(pid)})
        gpus.append({"uuid": gpu.findtext("uuid"), "memory_mib": int(gpu.findtext("fb_memory_usage/used").split()[0]),
                     "utilization_pct": int(gpu.findtext("utilization/gpu_util").split()[0]), "processes": processes})
    if sorted(gpu["uuid"] for gpu in gpus) != sorted(devices):
        raise ValueError("GPU inventory mismatch")
    return gpus


def capture(args):
    args.output.mkdir(parents=True, exist_ok=False)
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    deadline = time.monotonic() + args.seconds

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    with (args.output / "samples.jsonl").open("x") as log:
        def observe():
            point = {"start": time.monotonic(), "boot_id": boot_id, "devices": args.devices}
            try:
                remaining = max(.001, deadline - time.monotonic())
                point["gpus"] = sample(args.devices, min(args.query_timeout, remaining))
            except (OSError, ValueError, TypeError, ET.ParseError, subprocess.SubprocessError) as error:
                point["error"] = f"{type(error).__name__}: {error}"
            point["end"] = time.monotonic()
            log.write(json.dumps(point) + "\n")
            log.flush()
        try:
            while time.monotonic() < deadline:
                observe()
                time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
        except KeyboardInterrupt:
            pass
        finally:
            observe()


def check(args):
    rows = [json.loads(line) for line in args.samples.read_text().splitlines()]
    task = json.loads(args.task.read_text())
    start, end = task["start"], task["end"]
    root = [task["pid"], task["start_ticks"]]
    if not all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in [start, end]
    ) or start >= end:
        raise ValueError("invalid task interval")
    if not all(type(x) is int and x > 0 for x in root):
        raise ValueError("invalid task identity")
    before = [i for i, row in enumerate(rows) if row["end"] <= start]
    after = [i for i, row in enumerate(rows) if row["start"] >= end]
    if not before or not after:
        raise ValueError("task interval has no before/after sample")
    window = rows[before[-1]:after[0] + 1]
    devices = task["devices"]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("invalid task devices")
    if not window or any(
        not all(
            isinstance(row[key], (int, float))
            and not isinstance(row[key], bool)
            and math.isfinite(row[key])
            for key in ("start", "end")
        )
        for row in window
    ):
        raise ValueError("invalid sample times")
    for previous, point in zip(window, window[1:]):
        if point["start"] < previous["end"] or point["end"] - previous["end"] > args.max_gap:
            raise ValueError("sample ordering or coverage gap")
    foreign = []
    for point in window:
        if point.get("error") or point["end"] < point["start"] or point["end"] - point["start"] > args.max_gap:
            raise ValueError("failed or excessively slow sample")
        if point["boot_id"] != task["boot_id"] or sorted(point["devices"]) != sorted(devices):
            raise ValueError("host or device identity mismatch")
        if sorted(gpu["uuid"] for gpu in point["gpus"]) != sorted(devices):
            raise ValueError("incomplete GPU sample")
        for gpu in point["gpus"]:
            if gpu["memory_mib"] < 0 or not 0 <= gpu["utilization_pct"] <= 100:
                raise ValueError("invalid GPU telemetry")
            for process in gpu["processes"]:
                if process.get("type") in ("M", "M+C"):
                    raise ValueError("MPS process attribution unsupported")
                if not process["ancestors"]:
                    raise ValueError("GPU process identity unavailable")
                if root not in process["ancestors"]:
                    foreign.append({"uuid": gpu["uuid"], "pid": process["pid"]})
    if any(
        gpu["processes"] or gpu["utilization_pct"] or gpu["memory_mib"] > args.idle_memory
        for gpu in window[0]["gpus"]
    ):
        return {"resources_clear": False, "reason": "devices_not_idle_before_start"}
    return {"resources_clear": not foreign, "reason": "foreign_process" if foreign else "no_conflict_observed",
            "foreign": foreign, "sample_count": len(window), "start": start, "end": end}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("capture")
    record.add_argument("--devices", nargs="+", required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--seconds", type=float, default=3600)
    record.add_argument("--interval", type=float, default=0.5)
    record.add_argument("--query-timeout", type=float, default=2)
    verify = commands.add_parser("check")
    verify.add_argument("--samples", type=Path, required=True)
    verify.add_argument("--task", type=Path, required=True)
    verify.add_argument("--max-gap", type=float, default=3)
    verify.add_argument("--idle-memory", type=int, default=256)
    args = parser.parse_args()
    if args.command == "capture":
        if not all(
            math.isfinite(x) and x > 0 for x in [args.seconds, args.interval, args.query_timeout]
        ) or args.seconds > 3600:
            parser.error("invalid recording budget")
        try:
            args.devices = [value[:4] + value[4:].lower() for value in args.devices]
            if len(set(args.devices)) != len(args.devices) or any(
                not value.startswith("GPU-") or str(uuid.UUID(value[4:])) != value[4:]
                for value in args.devices
            ):
                raise ValueError("provide distinct full GPU UUIDs")
        except ValueError as error:
            parser.error(str(error))
        capture(args)
        return 0
    try:
        if not math.isfinite(args.max_gap) or args.max_gap <= 0 or args.idle_memory < 0:
            raise ValueError("invalid verification policy")
        result = check(args)
    except (OSError, KeyError, IndexError, TypeError, ValueError) as error:
        result = {"resources_clear": False, "reason": f"unverified: {error}"}
    print(json.dumps(result))
    return 0 if result["resources_clear"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
