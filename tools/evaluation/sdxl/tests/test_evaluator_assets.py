import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("prepare_assets", Path(__file__).resolve().parents[1] / "prepare_assets.py")
ASSETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSETS)


class EvaluatorAssetTests(unittest.TestCase):
    def test_missing_modified_and_valid_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "configs"
            (config / "nvidia").mkdir(parents=True)
            (config / "common").mkdir()
            (config / "nvidia/source-lock.json").write_text(json.dumps({"geneval": {"repo": "example/test", "revision": "fixed"}}))
            content = b"chair\n"
            lock = {"files": [{"path": "evaluation/object_names.txt", "sha256": hashlib.sha256(content).hexdigest()}]}
            (config / "common/evaluator-assets.json").write_text(json.dumps(lock))
            target = root / "source/evaluation/object_names.txt"
            with patch.object(ASSETS, "HERE", root):
                with self.assertRaises(ValueError):
                    ASSETS.prepare(root / "source", check_only=True)
                target.parent.mkdir(parents=True)
                target.write_bytes(b"wrong\n")
                with self.assertRaises(ValueError):
                    ASSETS.prepare(root / "source", check_only=True)
                target.write_bytes(content)
                self.assertEqual(ASSETS.prepare(root / "source", check_only=True)["verified"], lock["files"])


if __name__ == "__main__":
    unittest.main()
