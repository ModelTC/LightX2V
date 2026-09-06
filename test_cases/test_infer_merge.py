import argparse
import ast
import os
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch


class InferMergeTest(unittest.TestCase):
    def test_profiler_is_opt_in_and_preserves_json_warmup(self):
        path = Path(__file__).resolve().parents[1] / "lightx2v" / "infer.py"
        tree = ast.parse(path.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        code = compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec")
        for value in (None, "0", "1"):
            with self.subTest(profiler=value):
                runner = MagicMock()
                config = {"parallel": False, "warmup": True}
                profiler_module = types.ModuleType("torch.profiler")
                profiler_module.ProfilerActivity = types.SimpleNamespace(CPU="cpu", CUDA="cuda")
                profiler_module.profile = MagicMock()
                profiler_module.record_function = MagicMock(side_effect=lambda _: nullcontext())
                profiler = profiler_module.profile.return_value.__enter__.return_value
                namespace = {
                    "argparse": argparse,
                    "os": os,
                    "OMNI_VISION_SUBTASK_CHOICES": (),
                    "WAN_ANIMATE2_MODEL_ID": "wan2.2_animate2",
                    "seed_all": MagicMock(),
                    "set_config": MagicMock(return_value=config),
                    "init_empty_input_info": MagicMock(return_value=object()),
                    "print_config": MagicMock(),
                    "validate_config_paths": MagicMock(),
                    "ProfilingContext4DebugL1": lambda _: nullcontext(),
                    "init_runner": MagicMock(return_value=runner),
                    "update_input_info_from_dict": MagicMock(),
                    "dist": types.SimpleNamespace(is_initialized=lambda: False),
                    "logger": MagicMock(),
                }
                env = {"LIGHTX2V_TORCH_PROFILER_FULL_TRACE": "/tmp/lightx2v-test/trace.json"}
                if value is not None:
                    env["LIGHTX2V_TORCH_PROFILER_FULL"] = value
                argv = ["infer", "--model_cls", "wan2.2_moe", "--model_path", "/models/wan", "--config_json", "thor.json"]
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.dict(sys.modules, {"torch.profiler": profiler_module}),
                    patch.object(sys, "argv", argv),
                    patch("os.makedirs") as mkdir,
                    patch("builtins.print"),
                ):
                    exec(code, namespace)  # noqa: S102 - Execute the local CLI function without loading GPU runners.
                    namespace["main"]()
                runner.run_pipeline.assert_called_once()
                self.assertIs(config["warmup"], True)
                if value == "1":
                    profiler_module.profile.assert_called_once()
                    profiler.export_chrome_trace.assert_called_once_with(env["LIGHTX2V_TORCH_PROFILER_FULL_TRACE"])
                    mkdir.assert_called_once_with("/tmp/lightx2v-test", exist_ok=True)
                else:
                    profiler_module.profile.assert_not_called()
                    profiler.export_chrome_trace.assert_not_called()
                    mkdir.assert_not_called()


if __name__ == "__main__":
    unittest.main()
