"""Check that platform settings are consumed and snapshots retain directory structure."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evaluate as evaluation


class PlatformConfigurationTests(unittest.TestCase):
    def test_platform_profile_controls_devices_environment_and_python(self):
        profile = copy.deepcopy(evaluation.load_platform_config())
        profile["docker_devices"] = ["/dev/custom_mlu{card}"]
        profile["visibility_environment"] = {"CUSTOM_DEVICE": "{card}"}
        profile["environment"]["CUSTOM_RUNTIME"] = "enabled"
        profile["python_executable"] = "/custom/python"
        args = SimpleNamespace(card=3, image="test-image", reuse_runs=None, repeats=2, limit=1, data_root=Path("/data-source"))
        with patch.object(evaluation, "load_platform_config", return_value=profile):
            _, command = evaluation.container_command(args, Path("/output"), "lightx2v")
            _, scoring = evaluation.container_command(args, Path("/output"), "postprocess")
        self.assertIn("/dev/custom_mlu3", command)
        self.assertIn("CUSTOM_DEVICE=3", command)
        self.assertIn("CUSTOM_RUNTIME=enabled", command)
        self.assertIn("/custom/python", command)
        self.assertNotIn("--device", scoring)
        self.assertNotIn("CUSTOM_DEVICE=3", scoring)

    def test_unsupported_platform_cannot_silently_run_as_mlu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation.write_json(root / "configs/mlu/runtime.json", {"platform": "nvidia", "device_type": "cuda"})
            with patch.object(evaluation, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "Invalid configs/mlu/runtime.json"):
                    evaluation.load_platform_config()

    def test_run_snapshot_copies_common_and_platform_configs_recursively(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "configs"
            evaluation.copy_active_configs(destination)
            for relative in evaluation.CONFIG_LOCATIONS.values():
                self.assertEqual((destination / relative).read_bytes(), (evaluation.ROOT / "configs" / relative).read_bytes())
            self.assertFalse((destination / "suite.json").exists())
            self.assertTrue((destination / "common/suite.json").is_file())
            self.assertTrue((destination / "mlu/runtime.json").is_file())


if __name__ == "__main__":
    unittest.main()
