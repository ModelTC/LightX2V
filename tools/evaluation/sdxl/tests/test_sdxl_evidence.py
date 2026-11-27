import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluation", ROOT / "evaluate.py")
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.chunk = self.root / "chunks/000000"
        self.chunk.mkdir(parents=True)
        self.expected = [{"id": "a"}, {"id": "b"}]
        self.source_lock = EVALUATION.read_json(ROOT / "configs/nvidia/source-lock.json")
        self.metadata = {
            "model": EVALUATION.MODEL,
            "dataset": "fixture",
            "backend": "native",
            "evaluation_sha256": "fixed-generation",
            "manifest_sha256": "fixed-input",
            "weight_lock_sha256": "fixed-weights",
            "source_lock": self.source_lock,
            "status": "COMPLETE",
            "count": 2,
            "environment": {
                "visible_device": "GPU-fixed",
                "generation_environment": EVALUATION.read_json(ROOT / "configs/nvidia/runtime.json")["generation_environment"],
            },
        }
        self.resources = {"resources_clear": True, "task_cleanup_verified": True, "returncode": 0}
        self.records = []
        for name in ("a", "b"):
            path = self.chunk / f"{name}.bin"
            path.write_bytes(name.encode())
            self.records.append({"id": name, "status": "SUCCESS", "e2e_latency_s": 1.0, "artifact": path.name, "sha256": EVALUATION.sha256(path)})
        self.save()
        self.input_patch = patch.object(EVALUATION, "inputs", return_value=self.expected)
        self.suite_patch = patch.object(EVALUATION, "suite", return_value={"datasets": {"fixture": {"sha256": "fixed-input"}}})
        self.input_patch.start()
        self.suite_patch.start()
        self.addCleanup(self.input_patch.stop)
        self.addCleanup(self.suite_patch.stop)

    def save(self):
        (self.chunk / "run.json").write_text(json.dumps(self.metadata))
        (self.chunk / "resources.json").write_text(json.dumps(self.resources))
        (self.chunk / "samples.jsonl").write_text("".join(json.dumps(record) + "\n" for record in self.records))

    def verify(self):
        return EVALUATION.completed_runs(self.root, "fixture", "native")

    def test_complete_is_accepted_and_latency_count_matches(self):
        expected, generated = self.verify()
        self.assertEqual(expected, self.expected)
        self.assertEqual(set(generated), {"a", "b"})
        self.assertEqual(EVALUATION.latency_summary(generated)["count"], 2)

    def test_missing_failure_and_duplicate_cannot_shrink_denominator(self):
        original = copy.deepcopy(self.records)
        for changed in (original[:1], [{**original[0], "status": "FAILED"}, original[1]], [original[0], original[0]]):
            with self.subTest(records=changed):
                self.records = changed
                self.save()
                with self.assertRaises(ValueError):
                    self.verify()

    def test_output_hash_tampering_is_rejected(self):
        (self.chunk / "a.bin").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.verify()

    def test_unverified_resources_failed_exit_and_cleanup_are_rejected(self):
        for field, value in (("resources_clear", False), ("task_cleanup_verified", False), ("returncode", 1)):
            with self.subTest(field=field):
                self.resources = {"resources_clear": True, "task_cleanup_verified": True, "returncode": 0, field: value}
                self.save()
                with self.assertRaises(ValueError):
                    self.verify()

    def test_changed_source_or_dataset_is_rejected(self):
        original = copy.deepcopy(self.metadata)
        for field, value in (("source_lock", {}), ("manifest_sha256", "different")):
            with self.subTest(field=field):
                self.metadata = {**original, field: value}
                self.save()
                with self.assertRaises(ValueError):
                    self.verify()

    def test_changed_numerical_environment_is_rejected(self):
        for key, value in (("CUBLAS_WORKSPACE_CONFIG", ":4096:8"), ("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4")):
            with self.subTest(key=key):
                original = copy.deepcopy(self.metadata)
                self.metadata["environment"]["generation_environment"][key] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.verify()
                self.metadata = original

    def test_different_cards_for_one_dataset_are_rejected(self):
        self.metadata["count"] = 1
        second = self.records.pop()
        self.save()
        other = self.root / "chunks/000001"
        other.mkdir()
        (other / "b.bin").write_bytes(b"b")
        metadata = copy.deepcopy(self.metadata)
        metadata["environment"]["visible_device"] = "GPU-other"
        (other / "run.json").write_text(json.dumps(metadata))
        (other / "resources.json").write_text(json.dumps(self.resources))
        (other / "samples.jsonl").write_text(json.dumps(second) + "\n")
        with self.assertRaises(ValueError):
            self.verify()


if __name__ == "__main__":
    unittest.main()
