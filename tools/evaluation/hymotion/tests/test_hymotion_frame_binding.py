import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hymotion_binding", ROOT / "evaluate.py")
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


class FrameBindingTests(unittest.TestCase):
    def test_request_matches_original_duration_rounding_in_both_directions(self):
        calls = []

        class Runner:
            def prepare_request(self, request):
                return request

            def run_request(self, request):
                calls.append(request)

            def close(self):
                pass

        modules = {}
        for name in ("lightx2v", "lightx2v.models", "lightx2v.models.runners", "lightx2v.utils", "lightx2v.models.runners.runner_factory", "lightx2v.utils.set_config"):
            module = types.ModuleType(name)
            module.__path__ = []
            modules[name] = module
        modules["lightx2v.models.runners.runner_factory"].build_runner = lambda config: Runner()
        modules["lightx2v.utils.set_config"].build_startup_config = lambda config: config
        paths = {key: "/unused" for key in ("lightx2v", "weights", "qwen", "clip")}
        with patch.dict(sys.modules, modules), patch.object(sys, "path", list(sys.path)):
            infer = EVALUATION.load_adapter("lightx2v", paths)
        for duration, theoretical_frames, native_frames in ((4.15, 124, 125), (2.05, 62, 61), (9.95, 298, 298), (0.5, 15, 15)):
            with self.subTest(duration=duration):
                infer({"prompt": "A person walks.", "duration_s": duration, "expected_output_frames_30fps": theoretical_frames}, Path("/unused/output.npz"))
                self.assertEqual(calls[-1]["num_frames"], native_frames)


if __name__ == "__main__":
    unittest.main()
