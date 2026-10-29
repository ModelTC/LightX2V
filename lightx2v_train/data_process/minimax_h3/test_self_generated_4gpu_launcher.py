"""Exercise download/encode ordering and child cleanup without models or network."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/download_encode_minimax_h3_self_generated_4gpu.sh"
FAKE_PYTHON = r"""
import json, os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
def record(stage, **extra):
    payload = json.dumps({"stage": stage, "args": args, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), **extra}) + "\n"
    fd = os.open(os.environ["FAKE_EVENTS"], os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    os.write(fd, payload.encode()); os.close(fd)
if args[0] == "-c":
    sys.exit(int(os.environ.get("FAKE_MISSING_MODELSCOPE", "0")) if 'find_spec("modelscope")' in args[1] else 0)
if args[:2] == ["-m", "pip"]:
    record("pip"); sys.exit(0)
if args[0] == "-u": args.pop(0)
name = Path(args[0]).name
if name.startswith("prepare_"):
    stage = args[1]
    record(stage)
    if stage == os.environ.get("FAKE_FAIL_STAGE"):
        sys.exit(17)
    sys.exit(0)
shard = int(args[args.index("--shard-index") + 1])
record("encode_start", shard=shard)
def cancel(*_):
    record("encode_cancel", shard=shard)
    sys.exit(143)
signal.signal(signal.SIGTERM, cancel)
if os.environ.get("FAKE_FAIL_STAGE") == "encode":
    if shard == 1:
        time.sleep(0.05)
        record("encode_fail", shard=shard)
        sys.exit(23)
    time.sleep(30)
else:
    time.sleep(float(os.environ.get("FAKE_ENCODE_SECONDS", "0.1")))
record("encode_end", shard=shard)
"""


class SelfGeneratedFourGpuLauncherTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "paths with spaces"
        self.root.mkdir()
        self.code = self.root / "code"
        helper_dir = self.code / "lightx2v_train/data_process/minimax_h3"
        helper_dir.mkdir(parents=True)
        for name in ("prepare_modelscope_h3_dataset.py", "encode_modelscope_h3_dataset.py"):
            (helper_dir / name).write_text("# Executed through the fake interpreter.\n")
        (self.root / "model/Ref2VA").mkdir(parents=True)
        for component in (
            "vae/config.json",
            "audio_vae/config.json",
            "text_encoder/config.json",
            "tokenizer/tokenizer_config.json",
            "processor/preprocessor_config.json",
            "Ref2VA/text_encoder/config.json",
            "Ref2VA/tokenizer/tokenizer_config.json",
            "Ref2VA/processor/preprocessor_config.json",
            "Ref2VA/video_vae/config.json",
            "Ref2VA/audio_vae/config.json",
        ):
            path = self.root / "model" / component
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        fake_python = self.root / "fake python"
        fake_python.write_text(f"#!{sys.executable}\n" + FAKE_PYTHON)
        fake_python.chmod(0o700)
        self.events = self.root / "events.jsonl"
        self.environment = {
            **os.environ,
            "H3_CODE_ROOT": str(self.code),
            "H3_PYTHON": str(fake_python),
            "H3_MODEL_PATH": str(self.root / "model"),
            "H3_REF_MODEL_PATH": str(self.root / "model/Ref2VA"),
            "H3_SELFGEN_RAW": str(self.root / "raw"),
            "H3_SELFGEN_CACHE": str(self.root / "cache"),
            "H3_SELFGEN_SUBSETS": "all",
            "H3_SELFGEN_ALLOW_FAILURES": "0",
            "H3_SELFGEN_ALLOW_MISSING_MEDIA": "1",
            "CUDA_VISIBLE_DEVICES": "4,5,6,7",
            "FAKE_EVENTS": str(self.events),
        }
        for name in ("H3_SELFGEN_MAX_SAMPLES", "FAKE_FAIL_STAGE", "FAKE_MISSING_MODELSCOPE", "FAKE_ENCODE_SECONDS"):
            self.environment.pop(name, None)

    def run_script(self, *args, **environment):
        return subprocess.run(["bash", str(SCRIPT), *args], env={**self.environment, **environment}, text=True, capture_output=True, timeout=10)

    def records(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()] if self.events.exists() else []

    def test_download_then_four_disjoint_gpu_workers_then_merge(self):
        result = self.run_script(H3_SELFGEN_MAX_SAMPLES="2")
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.records()
        stages = [row["stage"] for row in records]
        self.assertEqual(stages[:2], ["download", "prepare"])
        self.assertEqual(stages[-1], "merge")
        workers = [row for row in records if row["stage"] == "encode_start"]
        self.assertEqual({row["shard"]: row["gpu"] for row in workers}, {0: "4", 1: "5", 2: "6", 3: "7"})
        self.assertEqual(stages.count("encode_end"), 4)
        self.assertIn("--max-samples-per-subset", records[1]["args"])
        self.assertIn("--allow-missing-media", records[1]["args"])
        self.assertTrue(all("--allow-failures" not in row["args"] for row in records))
        self.assertIn("Finished:", result.stdout)

    def test_download_failure_never_starts_prepare_or_gpu_workers(self):
        result = self.run_script(FAKE_FAIL_STAGE="download")
        self.assertEqual(result.returncode, 17)
        self.assertEqual([row["stage"] for row in self.records()], ["download"])

    def test_missing_model_component_fails_before_downloading(self):
        (self.root / "model/vae/config.json").unlink()
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing local H3 component", result.stderr)
        self.assertFalse(self.records())

    def test_worker_failure_stops_other_owned_workers_and_never_merges(self):
        start = time.monotonic()
        result = self.run_script(FAKE_FAIL_STAGE="encode")
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertLess(time.monotonic() - start, 8)
        records = self.records()
        self.assertNotIn("merge", [row["stage"] for row in records])
        started = {row["shard"] for row in records if row["stage"] == "encode_start"}
        canceled = {row["shard"] for row in records if row["stage"] == "encode_cancel"}
        self.assertEqual(canceled, started - {1})

    def test_completed_short_workers_are_still_waited_before_merge(self):
        result = self.run_script(FAKE_ENCODE_SECONDS="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.records()
        self.assertEqual(sum(row["stage"] == "encode_end" for row in records), 4)
        self.assertEqual(records[-1]["stage"], "merge")

    def test_dry_run_is_read_only_and_does_not_invoke_interpreter(self):
        result = self.run_script("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.events.exists())
        self.assertFalse((self.root / "raw").exists())
        self.assertFalse((self.root / "cache").exists())
        self.assertEqual(result.stdout.count("--shard-index"), 4)
        self.assertIn("all_metadata.jsonl", SCRIPT.read_text())

    def test_missing_modelscope_installs_before_downloading(self):
        result = self.run_script(FAKE_MISSING_MODELSCOPE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([row["stage"] for row in self.records()][:3], ["pip", "download", "prepare"])

    def test_invalid_gpu_lists_and_flags_fail_without_work(self):
        for devices in ("", "0,1,2", "0,1,2,2", "0,1,2,3,", "0,1, 2,3"):
            with self.subTest(devices=devices):
                result = self.run_script("--dry-run", CUDA_VISIBLE_DEVICES=devices)
                self.assertNotEqual(result.returncode, 0)
        result = self.run_script("--dry-run", H3_SELFGEN_ALLOW_FAILURES="yes")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.events.exists())

    def test_strict_media_and_allow_encode_failure_flags_are_independent(self):
        result = self.run_script(H3_SELFGEN_ALLOW_MISSING_MEDIA="0", H3_SELFGEN_ALLOW_FAILURES="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.records()
        self.assertNotIn("--allow-missing-media", records[1]["args"])
        for row in records:
            if row["stage"] in {"encode_start", "merge"}:
                self.assertIn("--allow-failures", row["args"])


if __name__ == "__main__":
    unittest.main()
