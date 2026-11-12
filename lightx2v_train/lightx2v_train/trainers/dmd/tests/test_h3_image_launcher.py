"""CPU checks for the eight-GPU H3-Image preprocessing/training handoff."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[5]
SCRIPT = REPO / "lightx2v_train/scripts/run_minimax_h3_image_pdmd_stages.py"
spec = importlib.util.spec_from_file_location("h3_image_stages", SCRIPT)
stages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stages)


class H3ImageLauncherTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        args = stages.arguments([
            "--code-root", str(REPO), "--model-path", str(root / "image"),
            "--pe-model-path", str(root / "pe-model"), "--dataset-dir", str(root / "data"),
            "--output-dir", str(root / "output"), "--count", "2", "--max-iters", "12",
        ])
        files = {
            args.model_path: ["pdd_config.json", "transformer_diffusers/config.json",
                              "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
                              "processor/preprocessor_config.json"],
            args.text_encoder_path: ["config.json"],
            args.pe_model_path: ["config.json", "tokenizer.json", "tokenizer_config.json",
                                 "prompt/t2i/system_prompt.txt", "prompt/t2i/task_prompt.txt",
                                 "prompt/t2i/request_params.json"],
        }
        for folder, names in files.items():
            for name in names:
                path = folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
        for folder, name in (
            (args.model_path / "transformer_diffusers", "diffusion_pytorch_model.safetensors.index.json"),
            (args.text_encoder_path, "model.safetensors.index.json"),
            (args.pe_model_path, "model.safetensors.index.json"),
        ):
            (folder / name).write_text(json.dumps({"weight_map": {"test.weight": "part.safetensors"}}))
            (folder / "part.safetensors").write_bytes(b"test")
        self.write_prompts(args.prompts)
        return args

    @staticmethod
    def write_prompts(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(json.dumps({"id": i, "prompt": f"scene {i}"}) for i in range(2)) + '\n')

    def preprocessing_result(self, args, command, **kwargs):
        self.assertTrue(kwargs["check"])
        self.assertEqual(kwargs["env"]["H3_IMAGE_CACHE"], str(args.cache_dir / "metadata.jsonl"))
        self.assertEqual(kwargs["env"]["H3_IMAGE_MAX_ITERS"], "12")
        if any(value.endswith("expand_prompts.py") for value in command):
            self.write_prompts(args.pe_cache / "prompts_expanded.jsonl")
        elif any(value.endswith("prepare_cache.py") for value in command):
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for i in range(2):
                condition = args.cache_dir / f"{i}.pt"
                condition.write_bytes(b"condition")
                rows.append({"condition_path": str(condition)})
            (args.cache_dir / "metadata.jsonl").write_text('\n'.join(json.dumps(row) for row in rows))

    def test_copied_prompts_without_receipt_are_reused_before_sequential_handoff(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            self.assertFalse(args.prompts.with_suffix(".selection.json").exists())
            with patch.object(stages, "validate_environment"), patch.object(stages.subprocess, "run") as run:
                run.side_effect = lambda command, **kwargs: self.preprocessing_result(args, command, **kwargs)
                stages.run(args)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 3)
            self.assertTrue(any(x.endswith("expand_prompts.py") for x in commands[0]))
            self.assertTrue(any(x.endswith("prepare_cache.py") for x in commands[1]))
            self.assertTrue(any(x.endswith("train.py") for x in commands[2]))
            for command in commands:
                self.assertIn("--standalone", command)
                self.assertIn("--nnodes=1", command)
                self.assertIn("--nproc_per_node=8", command)
                self.assertIn("--max_restarts=0", command)

    def test_failed_pe_never_starts_cache_or_training(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            with patch.object(stages, "validate_environment"), patch.object(stages.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["pe"])) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    stages.run(args)
            self.assertEqual(run.call_count, 1)

    def test_missing_weight_shard_reports_download_before_any_subprocess(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            (args.model_path / "transformer_diffusers/part.safetensors").unlink()
            with patch.object(stages.subprocess, "run") as run:
                with self.assertRaisesRegex(FileNotFoundError, "hf download MiniMaxAI/MiniMax-H3-Image"):
                    stages.run(args)
            run.assert_not_called()

    def test_raw_duplicate_and_missing_condition_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            args.prompts.write_text('{"id":0,"prompt":"Same text"}\n{"id":1,"prompt":"same  TEXT"}\n')
            with self.assertRaisesRegex(ValueError, "Duplicate prompt"):
                stages.validate_prompts(args.prompts, 2, unique=True)
            path = Path(root) / "metadata.jsonl"
            path.write_text('{"condition_path":"missing.pt"}\n')
            with self.assertRaisesRegex(FileNotFoundError, "Cached condition missing"):
                stages.validate_cache(path, 1)

    def test_dry_run_has_no_file_or_dependency_side_effects_and_http_is_forwarded(self):
        with tempfile.TemporaryDirectory() as root:
            args = stages.arguments(["--dataset-dir", root + "/absent-data", "--output-dir", root + "/absent-output",
                                     "--pe-backend", "http", "--pe-endpoint", "http://127.0.0.1:30000", "--dry-run"])
            with patch.object(stages, "validate_models") as models, patch.object(stages, "validate_environment") as environment, patch.object(stages.subprocess, "run") as run:
                stages.run(args)
            models.assert_not_called()
            environment.assert_not_called()
            run.assert_not_called()
            self.assertEqual(list(Path(root).iterdir()), [])
            command = stages.commands(args)["pe"]
            self.assertEqual(command[command.index("--backend") + 1], "http")
            self.assertEqual(command[command.index("--endpoint") + 1], "http://127.0.0.1:30000")

    def test_gan_teacher_cache_is_generated_and_verified_before_training(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            args.method = "pdmd_gan"
            with patch.object(stages, "validate_environment"), patch.object(stages, "validate_gan_cache") as verify, patch.object(stages.subprocess, "run") as run:
                run.side_effect = lambda command, **kwargs: self.preprocessing_result(args, command, **kwargs)
                stages.run(args)
            calls = run.call_args_list
            scripts = [next(value for value in call.args[0] if value.endswith(".py")) for call in calls]
            self.assertEqual([Path(script).name for script in scripts], ["expand_prompts.py", "prepare_cache.py", "prepare_gan_real_cache.py", "train.py"])
            teacher_command = calls[2].args[0]
            self.assertEqual(teacher_command[teacher_command.index("--condition-cache") + 1], str(args.cache_dir / "metadata.jsonl"))
            self.assertNotIn("--prompts", teacher_command)
            verify.assert_called_once_with(args.gan_real_dir / "metadata.jsonl", args.count)
            self.assertEqual(calls[-1].kwargs["env"]["H3_IMAGE_GAN_REAL_CACHE"], str(args.gan_real_dir / "metadata.jsonl"))

    def test_invalid_gan_teacher_cache_prevents_training(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            args.method = "pdmd_gan"
            with patch.object(stages, "validate_environment"), patch.object(stages, "validate_gan_cache", side_effect=ValueError("Missing teacher latents")), patch.object(stages.subprocess, "run") as run:
                run.side_effect = lambda command, **kwargs: self.preprocessing_result(args, command, **kwargs)
                with self.assertRaisesRegex(ValueError, "Missing teacher latents"):
                    stages.run(args)
            self.assertFalse(any(value.endswith("train.py") for call in run.call_args_list for value in call.args[0]))

    def test_gan_selects_its_own_config_and_output(self):
        with patch.dict(stages.os.environ, {}, clear=True):
            args = stages.arguments(["--method", "pdmd_gan"])
        self.assertEqual(args.config.name, "minimax_h3_image_pdmd2_gan_2048_r128_a128_fsdp8.yaml")
        self.assertEqual(args.output_dir.name, "h3_image_pdmd2_gan_10k_2048")
        self.assertEqual(args.max_iters, 15000)


if __name__ == "__main__":
    unittest.main()
