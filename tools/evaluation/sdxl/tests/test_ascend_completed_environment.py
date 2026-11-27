import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ascend_completed_environment", ROOT / "evaluate.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class AscendCompletedEnvironmentTests(unittest.TestCase):
    def setUp(self):
        runtime = M.read_json(ROOT / "configs/ascend_npu/runtime.json")
        self.metadata = dict(platform="ascend_npu", backend="lightx2v", seed=42,
            environment=dict(platform="ascend_npu", precision_policy=runtime["precision_policy"],
                request_hf32_conv=False, request_hf32_matmul=False, torch_npu="2.7.1.post8",
                device="Ascend910B2C", visible_device="2", physical_card="Ascend-0000:52:00.0"))
        self.resources = dict(physical_card=2, resources_clear=True, task_cleanup_verified=True,
            returncode=0, environment=dict(PLATFORM="ascend_npu", DTYPE="BF16", HF_HUB_OFFLINE="1",
                OMP_NUM_THREADS="8", ASCEND_RT_VISIBLE_DEVICES="2", ASCEND_PHYSICAL_CARD="Ascend-0000:52:00.0"))

    def test_full_result_uses_npu_provenance_without_cuda_fields(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, PLATFORM="ascend_npu"):
            root = Path(temporary)
            chunk = root / "chunks/000000"
            chunk.mkdir(parents=True)
            artifact = chunk / "output.png"
            artifact.write_bytes(b"verified output")
            self.metadata.update(model="sdxl", dataset="geneval", count=1, status="COMPLETE",
                evaluation_sha256="frozen-generation", manifest_sha256=M.suite()["datasets"]["geneval"]["sha256"],
                weight_lock_sha256="frozen-weights", source_lock=M.read_json(ROOT / "configs/ascend_npu/source-lock.json"))
            (chunk / "run.json").write_text(json.dumps(self.metadata))
            (chunk / "resources.json").write_text(json.dumps(self.resources))
            (chunk / "samples.jsonl").write_text(json.dumps(dict(id="fixture", status="SUCCESS", artifact="output.png", sha256=M.sha256(artifact))) + "\n")
            with patch.object(M, "inputs", return_value=[dict(id="fixture")]), patch.object(M, "verify_generation_environment", side_effect=AssertionError("CUDA check called for NPU")):
                expected, generated = M.completed_runs(root, "geneval", "lightx2v")
            self.assertEqual(len(expected), 1)
            self.assertEqual(generated["fixture"]["gpu"], "Ascend-0000:52:00.0")
            self.resources["environment"]["OMP_NUM_THREADS"] = "4"
            (chunk / "resources.json").write_text(json.dumps(self.resources))
            with patch.object(M, "inputs", return_value=[dict(id="fixture")]), self.assertRaises(ValueError):
                M.completed_runs(root, "geneval", "lightx2v")

    def test_missing_precision_or_launch_fields_are_rejected(self):
        for key in ("platform", "precision_policy", "request_hf32_conv", "request_hf32_matmul", "torch_npu", "device", "visible_device", "physical_card"):
            metadata = copy.deepcopy(self.metadata)
            metadata["environment"].pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.verify_ascend_generation_environment(metadata, self.resources)
        for key in self.resources["environment"]:
            resources = copy.deepcopy(self.resources)
            resources["environment"].pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.verify_ascend_generation_environment(self.metadata, resources)

    def test_other_platform_completed_results_keep_cuda_environment_check(self):
        for platform in ("cuda", "metax_cuda"):
            with tempfile.TemporaryDirectory() as temporary:
                chunk = Path(temporary) / "chunks/000000"
                chunk.mkdir(parents=True)
                metadata = dict(self.metadata, model="sdxl", dataset="geneval", status="COMPLETE", platform=platform,
                    evaluation_sha256="e", manifest_sha256="m", weight_lock_sha256="w", source_lock={})
                (chunk / "run.json").write_text(json.dumps(metadata))
                with patch.object(M, "inputs", return_value=[]), patch.object(M, "verify_generation_environment", side_effect=ValueError("original CUDA check")) as check:
                    with self.assertRaisesRegex(ValueError, "original CUDA check"):
                        M.completed_runs(Path(temporary), "geneval", "lightx2v")
                    check.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
