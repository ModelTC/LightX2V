import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluation", ROOT / "evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class ProtocolTests(unittest.TestCase):
    def test_locked_complete_unique_fixtures(self):
        self.assertEqual({name: entry["count"] for name, entry in evaluation.suite()["datasets"].items()}, {"geneval": 553, "parti": 1632, "coco": 5000})
        for dataset, entry in evaluation.suite()["datasets"].items():
            records = evaluation.inputs(dataset)
            self.assertEqual(len(records), entry["count"])
        self.assertEqual(evaluation.suite()["repetitions"], 1)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                evaluation.resolve({"root": root}, "root", "../secret")

    def test_quality_nonfinite_and_mismatch_rejected(self):
        self.assertFalse(evaluation.quality_aligned({"clip_score": float("nan")}, {"clip_score": float("nan")}))
        self.assertFalse(evaluation.quality_aligned({"clip_score": 0.5}, {"clip_score": 0.51}))
        self.assertTrue(evaluation.quality_aligned({"clip_score": 0.5}, {"clip_score": 0.5000001}))

    def test_incomplete_campaign_rejected(self):
        dataset = next(iter(evaluation.suite()["datasets"]))
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                evaluation.completed_runs(root, dataset, "native")

    def test_other_platforms_fail_closed(self):
        for platform in ("mlu", "metax", "ascend_npu"):
            value = json.loads((ROOT / "configs" / platform / "runtime.json").read_text())
            self.assertEqual(value["status"], "not_implemented")


if __name__ == "__main__":
    unittest.main()
