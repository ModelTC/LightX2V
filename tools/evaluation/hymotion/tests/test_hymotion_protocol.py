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
        for dataset, entry in evaluation.suite()["datasets"].items():
            records = evaluation.inputs(dataset)
            self.assertEqual(len(records), entry["count"])
            if dataset == "gso":
                self.assertEqual(len({r["object_id"] for r in records}), 1023)
                self.assertTrue(all(r["view"] == 0 and r["input"].endswith(".thumbnail_0.jpg") for r in records))
            if dataset == "humanml3d":
                self.assertEqual(sum(r["evaluator_eligible"] for r in records), 4198)
        self.assertEqual(evaluation.suite()["repetitions"], 1)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                evaluation.resolve({"root": root}, "root", "../secret")

    def test_quality_nonfinite_and_mismatch_rejected(self):
        self.assertFalse(evaluation.quality_aligned({"fid": float("nan")}, {"fid": float("nan")}))
        self.assertFalse(evaluation.quality_aligned({"fid": 0.5}, {"fid": 0.51}))
        self.assertTrue(evaluation.quality_aligned({"fid": 0.5}, {"fid": 0.5000001}))

    def test_incomplete_campaign_rejected(self):
        dataset = next(iter(evaluation.suite()["datasets"]))
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                evaluation.completed_runs(root, dataset, "native")

    def test_other_platforms_fail_closed(self):
        mlu = json.loads((ROOT / "configs" / "mlu" / "runtime.json").read_text())
        self.assertEqual(mlu["status"], "smoke_ready")
        for platform in ("metax", "ascend_npu"):
            value = json.loads((ROOT / "configs" / platform / "runtime.json").read_text())
            self.assertEqual(value["status"], "not_implemented")


if __name__ == "__main__":
    unittest.main()
