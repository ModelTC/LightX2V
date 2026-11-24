"""Changed worker bounds must preserve successful inputs and reject invalid history."""

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluation_resume", ROOT / "evaluate.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class ResumeTests(unittest.TestCase):
    def exercise(self, invalid=False, overlap=False):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output = base / "output"
            dataset = next(iter(M.suite()["datasets"]))
            records = [{"id": str(i)} for i in range(6)]
            paths = {"lightx2v": str(base), "native_python": "python", "lightx2v_python": "python"}
            paths_file = base / "paths.json"
            M.write_json(paths_file, paths)
            fingerprint = M.sha256(ROOT / "evaluate.py")
            identity = {"model": M.MODEL, "dataset": dataset, "gpu": "GPU-00000000-0000-0000-0000-000000000001", "count": 6, "chunk_size": 4, "evaluate_sha256": fingerprint, "paths": paths}
            output.mkdir()
            M.write_json(output / "campaign.json", identity)

            def history(backend, start, count):
                folder = output / backend / "chunks" / f"{start:06d}"
                folder.mkdir(parents=True)
                M.write_json(
                    folder / "run.json",
                    {
                        "model": M.MODEL,
                        "dataset": dataset,
                        "backend": backend,
                        "start": start,
                        "count": count,
                        "status": "COMPLETE",
                        "evaluation_sha256": fingerprint,
                        "source_lock": M.read_json(ROOT / "configs/nvidia/source-lock.json"),
                        "manifest_sha256": M.suite()["datasets"][dataset]["sha256"],
                    },
                )
                M.write_json(folder / "resources.json", {"resources_clear": not invalid, "task_cleanup_verified": True, "returncode": 0})
                (folder / "samples.jsonl").write_text("".join(json.dumps({"id": str(i), "status": "SUCCESS"}) + "\n" for i in range(start, start + count)))

            history("native", 0, 4)
            history("lightx2v", 0, 2)
            if overlap:
                history("native", 2, 2)
            args = argparse.Namespace(gpu=identity["gpu"], count=2, dataset=dataset, output=output, paths=paths_file, skip_score=True)
            observed = []

            def workload(command, *unused):
                def value(flag):
                    return command[command.index(flag) + 1]

                observed.append((value("--backend"), int(value("--start")), int(value("--count"))))

            with (
                patch.object(M, "inputs", return_value=records),
                patch.object(M, "check_inputs"),
                patch.object(M, "verify_source"),
                patch.object(M, "checked_worker_script", return_value=ROOT / "evaluate.py"),
                patch.object(M, "observed_execution", side_effect=workload),
            ):
                if invalid or overlap:
                    with self.assertRaises(ValueError):
                        M.campaign(args, paths)
                    self.assertEqual(observed, [])
                else:
                    M.campaign(args, paths)
                    self.assertEqual(observed, [("lightx2v", 2, 2), ("native", 4, 2), ("lightx2v", 4, 2)])
                    self.assertEqual(M.read_json(output / "campaign.json"), identity)
                    plans = list(output.glob("controller-plan-*.json"))
                    self.assertEqual(len(plans), 1)
                    self.assertEqual(M.read_json(plans[0])["requested_chunk_size"], 2)

    def test_smaller_chunks_never_repeat_successful_inputs(self):
        self.exercise()

    def test_resource_invalid_chunk_cannot_be_reused(self):
        self.exercise(invalid=True)

    def test_overlapping_completed_ranges_are_rejected(self):
        self.exercise(overlap=True)


if __name__ == "__main__":
    unittest.main()
