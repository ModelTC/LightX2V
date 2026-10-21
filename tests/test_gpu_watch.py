"""Deterministic CPU checks; real GPU/container execution is verified separately."""

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools/runtime/gpu_watch.py"
spec = importlib.util.spec_from_file_location("gpu_watch", SOURCE)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)
GPU = "GPU-11111111-1111-1111-1111-111111111111"


def evidence():
    task = dict(pid=11, start_ticks=31, start=10.0, end=12.0, boot_id="test-boot", devices=[GPU])
    rows = []
    for timestamp in [9.0, 10.5, 11.5, 12.5]:
        apps = [{"pid": 101, "type": "C", "ancestors": [[101, 99], [11, 31]]}] if 10 < timestamp < 12 else []
        rows.append(dict(start=timestamp, end=timestamp + .01, boot_id="test-boot", devices=[GPU],
                         gpus=[dict(uuid=GPU, memory_mib=0, utilization_pct=0, processes=apps)]))
    return task, rows


def inputs(tmp_path, task, rows):
    args = SimpleNamespace(task=tmp_path / "task.json", samples=tmp_path / "samples.jsonl", max_gap=3, idle_memory=256)
    args.task.write_text(json.dumps(task))
    args.samples.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return args


def test_owned_descendant_is_clear(tmp_path):
    result = watch.check(inputs(tmp_path, *evidence()))
    assert result["resources_clear"] and result["sample_count"] == 4


@pytest.mark.parametrize("record, field, value, expected_error", [
    ("idle_gpu", "utilization_pct", 50, False),
    ("process", "ancestors", [[101, 99], [12, 32]], False),
    ("task", "start_ticks", 32, False),
    ("process", "ancestors", [], True),
    ("process", "type", "M", True),
    ("process", "type", "M+C", True),
    ("sample", "error", "query timed out", True),
    ("task", "start", 8, True),
    ("task", "end", 13, True),
    ("sample", "start", float("nan"), True),
    ("sample", "boot_id", "other-host", True),
    ("sample", "devices", [], True),
    ("sample", "gpus", [], True),
    ("gpu", "utilization_pct", -1, True),
    ("task", "pid", True, True),
    ("task", "end", 10, True),
])
def test_incomplete_or_conflicting_evidence_is_never_clear(tmp_path, record, field, value, expected_error):
    task, rows = evidence()
    records = {"task": task, "sample": rows[1], "idle_gpu": rows[0]["gpus"][0],
               "gpu": rows[1]["gpus"][0], "process": rows[1]["gpus"][0]["processes"][0]}
    records[record][field] = value
    args = inputs(tmp_path, task, rows)
    if expected_error:
        with pytest.raises(ValueError):
            watch.check(args)
    else:
        assert watch.check(args)["resources_clear"] is False


def test_sample_gap_is_rejected(tmp_path):
    args = inputs(tmp_path, *evidence())
    args.max_gap = 1
    with pytest.raises(ValueError, match="coverage gap"):
        watch.check(args)


def test_xml_and_proc_identity_have_independent_expected_values(monkeypatch):
    stats = {"/proc/73/stat": "73 (worker with spaces) S 41 " + "0 " * 17 + "1234",
             "/proc/41/stat": "41 (parent) S 0 " + "0 " * 17 + "5678"}
    monkeypatch.setattr(Path, "read_text", lambda path: stats[str(path)])
    raw = f"""<nvidia_smi_log><gpu><uuid>{GPU}</uuid><mig_mode><current_mig>Disabled</current_mig></mig_mode>
    <fb_memory_usage><used>1 MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization>
    <processes><process_info><pid>73</pid><type>G</type></process_info></processes></gpu></nvidia_smi_log>"""
    monkeypatch.setattr(watch.subprocess, "check_output", lambda *args, **kwargs: raw.encode())
    assert watch.sample([GPU], .1)[0]["processes"] == [{"pid": 73, "type": "G", "ancestors": [[73, 1234], [41, 5678]]}]
    monkeypatch.setattr(watch.subprocess, "check_output", lambda *args, **kwargs: raw.replace("Disabled", "Enabled").encode())
    with pytest.raises(ValueError):
        watch.sample([GPU], .1)


@pytest.mark.parametrize("query_fails", [False, True])
def test_capture_budget_final_sample_and_error_retention(tmp_path, monkeypatch, query_fails):
    now, sleeps, timeouts = [0.0], [], []
    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    def observe(devices, timeout):
        timeouts.append(timeout)
        now[0] += min(.25, timeout)
        if query_fails:
            raise subprocess.TimeoutExpired("nvidia-smi", timeout)
        return []
    monkeypatch.setattr(watch, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep))
    monkeypatch.setattr(watch, "sample", observe)
    monkeypatch.setattr(watch.signal, "signal", lambda *args: None)
    args = SimpleNamespace(output=tmp_path / "watch", devices=[GPU], seconds=1, interval=99, query_timeout=99)
    watch.capture(args)
    rows = [json.loads(line) for line in (args.output / "samples.jsonl").read_text().splitlines()]
    assert timeouts == [1, .001] and sleeps == [.75]
    assert [row["start"] for row in rows] == [0, 1]
    assert all("TimeoutExpired" in row["error"] if query_fails else "gpus" in row for row in rows)
    with pytest.raises(FileExistsError):
        watch.capture(args)
