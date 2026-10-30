"""Metadata publication must work on shared mounts without hardlink support."""

import errno
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_minimax_h3_cm_pdmd_stages.py"
SPEC = importlib.util.spec_from_file_location("_h3_portable_publish", SCRIPT)
STAGES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGES)


class PortablePublishTest(unittest.TestCase):
    def test_no_hardlinks_publish_idempotently_reject_mismatch_and_clean_temporary_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stages/plan.json"
            value = {"task": "ref2av", "stages": [1500, 1500000]}
            with patch.object(STAGES.os, "link", side_effect=OSError(errno.ENOTSUP, "shared mount has no hardlinks")) as link:
                STAGES.publish_once(path, value)
                self.assertEqual(json.loads(path.read_text()), value)
                STAGES.publish_once(path, value)
                with self.assertRaisesRegex(RuntimeError, "Existing staged recipe differs"):
                    STAGES.publish_once(path, {**value, "task": "fl2av"})
                self.assertEqual(link.call_count, 1)
            self.assertEqual(json.loads(path.read_text()), value)
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
