"""MetaX selection must retain the physical device and LightX2V-only scope."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import evaluate as ev

class MetaXConfigurationTests(unittest.TestCase):
    def test_profile_and_container_scope(self):
        previous = ev.ACTIVE_PLATFORM
        try:
            ev.select_platform("metax")
            profile = ev.load_platform_config()
            self.assertEqual(profile["idle_memory_mib_max"], 900)
            self.assertEqual(profile["environment"]["PLATFORM"], "metax_cuda")
            sources = json.loads(ev.config_path("source-lock.json").read_text())
            self.assertEqual(list(sources), ["lightx2v-" + ev.MODEL])
            args = SimpleNamespace(card=3, image="test", reuse_runs=None, repeats=2, limit=1, data_root=Path("/data-source"))
            _, worker = ev.container_command(args, Path("/output"), "lightx2v")
            _, scoring = ev.container_command(args, Path("/output"), "postprocess")
            self.assertIn("CUDA_VISIBLE_DEVICES=3", worker)
            self.assertIn("MACA_VISIBLE_DEVICES=3", worker)
            self.assertIn("/dev/mxcd", worker)
            self.assertNotIn("--device", scoring)
            with tempfile.TemporaryDirectory() as d:
                ev.copy_active_configs(Path(d))
                self.assertTrue((Path(d) / "metax/runtime-requirements.txt").is_file())
        finally:
            ev.select_platform(previous)
