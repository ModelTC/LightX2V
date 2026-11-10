"""Verify the one-command workflow keeps failures visible and model processes isolated."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import evaluate as evaluation


class CampaignTests(unittest.TestCase):
    def test_model_and_metric_workers_have_different_device_access(self):
        args = SimpleNamespace(card=3, image="vendor-image", reuse_runs=None, repeats=2, limit=1, data_root=Path("/data-source"))
        _, model = evaluation.container_command(args, Path("/output"), "reference")
        _, scoring = evaluation.container_command(args, Path("/output"), "postprocess")
        self.assertIn("/dev/cambricon_dev3", model)
        self.assertNotIn("--device", scoring)
        self.assertIn("/eval/evaluate.py", model)
        self.assertIn("/eval/evaluate.py", scoring)

    def test_missing_runs_still_produce_reports_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            evaluation.write_json(output / "manifest.json", {"sample_count": 1, "repeats": 2, "reused_predictions": True, "stage_exit_codes": {}})
            args = SimpleNamespace(output=output, data_root=output / "data", reuse_runs=[output / "missing-reference", output / "missing-lightx2v"])
            self.assertEqual(evaluation.postprocess(args), 1)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(summary["results"]["reference"]["missing_measured_count"], 2)
            self.assertTrue(summary["errors"])
            for name in ("report.md", "requests.csv", "metrics.csv", "summary.csv", "comparison.json", "failures.json", "provenance.json"):
                self.assertTrue((output / name).is_file(), name)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            marker = output / "existing-result"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                evaluation.run_campaign(SimpleNamespace(output=output))
            self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
