"""CPU tests for telemetry staging and durable result publication, not CUDA proof."""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluation_archive_io", ROOT / "evaluate.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
WATCH = r"""import json,signal,sys,time
from pathlib import Path
if sys.argv[1]=='check':
 print(json.dumps({'resources_clear':True}));raise SystemExit(0)
args=sys.argv
output=Path(args[args.index('--output')+1]);output.mkdir(parents=True)
running=True
def stop(*unused):
 global running
 running=False
signal.signal(signal.SIGTERM,stop)
with (output/'samples.jsonl').open('w') as log:
 while running:
  now=time.monotonic()
  point={'start':now,'end':now,'gpus':[{'uuid':'GPU-00000000-0000-0000-0000-000000000001','processes':[],'utilization_pct':0,'memory_mib':0}]}
  log.write(json.dumps(point)+'\n');log.flush();time.sleep(.02)
"""


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("timeout"), "Linux /proc and GNU timeout required")
class CaptureArchiveTests(unittest.TestCase):
    def exercise(self, fail_copy=False):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            watch, task = root / "watch.py", root / "task.py"
            watch.write_text(WATCH)
            task.write_text("assert sum(range(10)) == 45\n")
            evidence, artifact, scratch = root / "result", root / "artifact", root / "local"
            artifact.mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PLATFORM="cuda", DTYPE="BF16", HF_HUB_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1")
            with patch.dict(os.environ, {"LIGHTX2V_EVAL_RESOURCE_SCRATCH": str(scratch), "LIGHTX2V_EVAL_RESOURCE_BACKEND": "xml"}):
                arguments = ([sys.executable, str(task)], env, sys.executable, watch, evidence, artifact, "GPU-00000000-0000-0000-0000-000000000001")
                if fail_copy:
                    with patch("shutil.copytree", side_effect=OSError("synthetic archive storage failure")), self.assertRaises(OSError):
                        M.observed_execution(*arguments)
                else:
                    M.observed_execution(*arguments)
            manifest = json.loads((evidence / "staging.json").read_text())
            local = Path(manifest["directory"])
            if fail_copy:
                self.assertEqual(manifest["state"], "RUNNING")
                self.assertTrue((local / "resources.json").is_file())
                self.assertFalse((artifact / "resources.json").exists())
            else:
                self.assertEqual(manifest["state"], "ARCHIVED")
                self.assertFalse(local.exists())
                self.assertTrue((evidence / "watch/samples.jsonl").is_file())
                self.assertEqual(json.loads((artifact / "resources.json").read_text()), json.loads((evidence / "resources.json").read_text()))
                self.assertTrue(json.loads((artifact / "resources.json").read_text())["task_cleanup_verified"])

    def test_archive_is_complete_before_chunk_is_resumable(self):
        self.exercise()

    def test_archive_failure_preserves_local_evidence_without_publishing_admissibility(self):
        self.exercise(fail_copy=True)


if __name__ == "__main__":
    unittest.main()
