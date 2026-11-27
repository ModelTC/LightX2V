"""Precision/source isolation and explicit single-backend selection."""

import argparse
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluation_bf16", ROOT / "evaluate.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class BF16ProfileTests(unittest.TestCase):
    def test_profile_has_distinct_source_and_bf16_model(self):
        with patch.dict(os.environ, {"LIGHTX2V_EVAL_PRECISION_PROFILE": "bf16"}):
            self.assertEqual(M.nvidia_config("model.json"), ROOT / "configs/nvidia/bf16/model.json")
            self.assertEqual(M.read_json(M.nvidia_config("model.json"))["inference_dtype"], "bf16")
            old = M.read_json(ROOT / "configs/nvidia/source-lock.json")["lightx2v"]["revision"]
            new = M.read_json(M.nvidia_config("source-lock.json"))["lightx2v"]["revision"]
            self.assertNotEqual(old, new)
            policy = M.read_json(M.nvidia_config("runtime.json"))["precision_policy"]
            self.assertTrue(policy["bf16"])
            self.assertTrue(policy["fp32"])

    def test_bf16_never_silently_runs_or_labels_native_precision(self):
        with patch.dict(os.environ, {"LIGHTX2V_EVAL_PRECISION_PROFILE": "bf16"}):
            self.assertEqual(M.campaign_backends(argparse.Namespace(backend="lightx2v")), ("lightx2v",))
            for backend in (None, "native"):
                with self.assertRaises(ValueError):
                    M.campaign_backends(argparse.Namespace(backend=backend))

    def test_default_profile_keeps_serial_parity_and_explicit_selection(self):
        with patch.dict(os.environ, {"LIGHTX2V_EVAL_PRECISION_PROFILE": "native"}):
            self.assertEqual(M.campaign_backends(argparse.Namespace()), ("native", "lightx2v"))
            self.assertEqual(M.campaign_backends(argparse.Namespace(backend="lightx2v")), ("lightx2v",))
            self.assertEqual(M.nvidia_config("model.json"), ROOT / "configs/nvidia/model.json")

    def test_unknown_profile_is_rejected(self):
        with patch.dict(os.environ, {"LIGHTX2V_EVAL_PRECISION_PROFILE": "ambiguous"}):
            with self.assertRaises(ValueError):
                M.nvidia_config("source-lock.json")


if __name__ == "__main__":
    unittest.main()
