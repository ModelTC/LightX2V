"""Source reconstruction is pinned, checked and does not depend on author caches."""

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluate as evaluation


class SourcePreparationTests(unittest.TestCase):
    def test_reference_edits_preserve_exact_output_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "model.py").write_text("first\nold\nlast\n")
            expected = b"first\nnew  \nlast\n"
            edits = root / "edits.json"
            evaluation.write_json(
                edits,
                {
                    "format": "line_edits_v1",
                    "files": [
                        {
                            "path": "model.py",
                            "base_sha256": evaluation.sha256(source / "model.py"),
                            "result_sha256": hashlib.sha256(expected).hexdigest(),
                            "hunks": [{"start": 1, "delete": 1, "insert": ["new  \n"]}],
                        }
                    ],
                },
            )
            evaluation.apply_reference_edits(source, edits, evaluation.sha256(edits))
            self.assertEqual((source / "model.py").read_bytes(), expected)

    def test_reference_edits_reject_paths_outside_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            edits = root / "edits.json"
            evaluation.write_json(edits, {"format": "line_edits_v1", "files": [{"path": "../escape.py"}]})
            with self.assertRaisesRegex(ValueError, "escapes"):
                evaluation.apply_reference_edits(root, edits, evaluation.sha256(edits))

    def test_prepare_reconstructs_a_pinned_git_revision_and_refuses_modified_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upstream = root / "upstream"
            subprocess.run(["git", "init", "--quiet", "--initial-branch=main", str(upstream)], check=True)
            (upstream / "model.py").write_text("VALUE = 42\n")
            subprocess.run(["git", "-C", str(upstream), "add", "model.py"], check=True)
            subprocess.run(["git", "-C", str(upstream), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "fixture"], check=True)
            revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
            project = root / "project"
            evaluation.write_json(
                project / "configs/mlu/source-lock.json",
                {
                    "reference-test": {
                        "repository": str(upstream),
                        "revision": revision,
                        "files": [{"path": "model.py", "bytes": (upstream / "model.py").stat().st_size, "sha256": evaluation.sha256(upstream / "model.py")}],
                    }
                },
            )
            with patch.object(evaluation, "ROOT", project):
                evaluation.prepare_sources()
                prepared = project / "sources/reference-test/model.py"
                self.assertEqual(prepared.read_bytes(), (upstream / "model.py").read_bytes())
                prepared.write_text("changed\n")
                with self.assertRaisesRegex(ValueError, "differs or is missing"):
                    evaluation.prepare_sources()
                self.assertEqual(prepared.read_text(), "changed\n")


if __name__ == "__main__":
    unittest.main()
