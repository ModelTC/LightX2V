"""Retired projection-only entry points fail before running Python or torchrun."""

import os
import subprocess
import unittest
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[4]


class RetiredPDMDLauncherTest(unittest.TestCase):
    def test_retired_h3_entry_points_fail_without_model_cache_or_python(self):
        for name in (
            "run_minimax_h3_ref2av_omni_imageonly_dmd8_fsdp32_32gpu_acp.sh",
            "run_minimax_h3_ref2av_omni_imageonly_pdmd8_fsdp40_40gpu_acp.sh",
            "run_minimax_h3_ref2av_omni_imageonly_pdmd8_count_random_fsdp32_32gpu_acp.sh",
        ):
            for args in ((), ("--dry-run",)):
                with self.subTest(name=name, args=args):
                    result = subprocess.run(
                        ["bash", str(TRAIN_ROOT / "scripts" / name), *args],
                        env={"PATH": os.environ["PATH"], "H3_PYTHON": "/nonexistent/python"},
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("projection-only PDMD launcher has been retired", result.stderr)
                    self.assertIn("pdmd8_official_fsdp32_32gpu_acp.sh", result.stderr)
                    self.assertEqual(result.stdout, "")

    def test_old_wan_comparison_switch_fails_before_python_or_torchrun(self):
        for name in ("run_wan21_dmd_pdmd_fsdp2.sh", "run_wan21_dmd_head_comparison_fsdp2.sh"):
            with self.subTest(name=name):
                result = subprocess.run(
                    ["bash", str(TRAIN_ROOT / "scripts" / name)],
                    env={
                        "PATH": os.environ["PATH"],
                        "WAN_DMD_MODEL": "/missing/model",
                        "WAN_DMD_PROMPTS": "/missing/prompts",
                        "WAN_DMD_OUTPUT": "/missing/output",
                        "WAN_DMD_PYTHON": "/nonexistent/python",
                        "CUDA_VISIBLE_DEVICES": "0,1",
                        "WAN_DMD_PROJECTED": "true",
                    },
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("projection-only PDMD route was removed", result.stderr)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
