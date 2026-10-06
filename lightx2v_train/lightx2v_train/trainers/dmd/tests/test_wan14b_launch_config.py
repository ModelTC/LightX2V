"""Offline contracts for the fresh Wan 14B-teacher three-way comparison.

Subprocesses, GPU queries and process-group signals are always mocked here.
"""

import copy
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, call, patch

from lightx2v_train.trainers.dmd.runtime import _DmdRuntime

TRAIN_ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("wan14b_launch_config_under_test", TRAIN_ROOT / "scripts/launch_wan14b_comparison.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
PATHS = {
    "WAN_DMD_MODEL": "/fixture/Wan2.1-T2V-1.3B",
    "WAN_DMD_TEACHER": "/fixture/Wan2.1-T2V-14B",
    "WAN_DMD_PROMPTS": "/fixture/prompts.txt",
}
GPU_GROUPS = {"dmd": "0,1", "pdmd": "2,3", "head": "5,6"}


class Wan14bConfigTest(unittest.TestCase):
    def config(self, group="dmd", dtype="fp32", accumulation=16, profile="fp32_sdpa", student_train_type="full"):
        paths = {
            **PATHS,
            "WAN_DMD_GRAD_ACCUM": str(accumulation),
            "WAN_DMD_PRECISION_PROFILE": profile,
            "WAN_DMD_STUDENT_TRAIN_TYPE": student_train_type,
        }
        return launcher.resolved_config(group, dtype, Path("/fixture/output"), paths)

    def test_default_accumulation_remains_sixteen(self):
        with patch.dict(os.environ, PATHS):
            os.environ.pop("WAN_DMD_GRAD_ACCUM", None)
            os.environ.pop("WAN_DMD_STUDENT_TRAIN_TYPE", None)
            config = launcher.resolved_config("head", "fp32", Path("/fixture/output"), PATHS)
        self.assertEqual(config, self.config("head"))

    def test_default_student_mode_keeps_legacy_full_config(self):
        for group in launcher.GROUPS:
            with self.subTest(group=group), patch.dict(os.environ, PATHS):
                os.environ.pop("WAN_DMD_STUDENT_TRAIN_TYPE", None)
                os.environ.pop("WAN_DMD_GRAD_ACCUM", None)
                os.environ.pop("WAN_DMD_PRECISION_PROFILE", None)
                default = launcher.resolved_config(group, "fp32", Path("/fixture/output"), PATHS)
                self.assertEqual(default, self.config(group))
                self.assertEqual(default["training"]["student"]["train_type"], "full")
                self.assertNotIn("lora", default["training"]["student"])

    def test_lora_changes_only_student_training_mode_and_adapter(self):
        expected_lora = {
            "rank": 128,
            "alpha": 8,
            "target_modules": ["q", "k", "v", "o", "ffn.0", "ffn.2"],
        }
        for group in launcher.GROUPS:
            for dtype, profile in (("fp32", "fp32_sdpa"), ("bf16", "fp32_sdpa"), ("bf16", "bf16_fa3")):
                with self.subTest(group=group, dtype=dtype, profile=profile):
                    original = self.config(group, dtype, accumulation=1, profile=profile)
                    lora = self.config(group, dtype, accumulation=1, profile=profile, student_train_type="lora")
                    training = lora["training"]
                    self.assertEqual(training["student"]["train_type"], "lora")
                    self.assertEqual(training["student"].pop("lora"), expected_lora)
                    self.assertEqual(training["fake"]["train_type"], "full")
                    self.assertNotIn("lora", training["fake"])
                    self.assertEqual(training["student"]["optimizer"]["learning_rate"], 5e-5)
                    self.assertEqual(training["student"]["ema"], {"enabled": True, "decay": 0.99, "use_for_inference": True})
                    self.assertEqual(training["dmd"]["update_order"], "student_first" if group == "pdmd" else "fake_first")
                    self.assertEqual(training["dmd"]["fake_update_ratio"], 1 if group == "pdmd" else 5)
                    training["student"]["train_type"] = "full"
                    # Exact equality protects precision, teacher CFG/path,
                    # fake optimizer, head fitting, sampling and EMA behavior.
                    self.assertEqual(original, lora)

    def test_invalid_student_mode_rejected(self):
        for value in ("", "FULL", "LoRA", "peft", "true"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "WAN_DMD_STUDENT_TRAIN_TYPE must be full or lora"):
                self.config(student_train_type=value)

    def test_acc1_changes_only_three_accumulation_and_head_fit_fields(self):
        for group in launcher.GROUPS:
            for dtype in ("fp32", "bf16"):
                with self.subTest(group=group, dtype=dtype):
                    acc16 = self.config(group, dtype)
                    acc1 = self.config(group, dtype, accumulation=1)
                    training = acc1["training"]
                    self.assertEqual(training["gradient_accumulation_iters"], 1)
                    head = training["dmd"]["residual_head"]
                    # One independent fit query, not sixteen fits replaying the
                    # five available fake-update queries in an acc1 iteration.
                    self.assertEqual(head["fit_steps"], 1)
                    self.assertEqual(head["fit_grad_accum_steps"], 1)
                    training["gradient_accumulation_iters"] = 16
                    head["fit_steps"] = head["fit_grad_accum_steps"] = 16
                    self.assertEqual(acc16, acc1)

    def test_three_groups_have_requested_loss_and_cadence_only(self):
        variants = []
        for group, projected, head, order, ratio in (
            ("dmd", False, False, "fake_first", 5),
            ("pdmd", True, False, "student_first", 1),
            ("head", False, True, "fake_first", 5),
        ):
            with self.subTest(group=group):
                config = self.config(group)
                dmd = config["training"]["dmd"]
                self.assertIs(config["model"]["capabilities"]["distribution_matching"].pop("projected_dmd"), projected)
                self.assertIs(dmd["residual_head"].pop("enabled"), head)
                self.assertEqual(dmd.pop("update_order"), order)
                self.assertEqual(dmd.pop("fake_update_ratio"), ratio)
                variants.append(config)
        self.assertEqual(variants[0], variants[1])
        self.assertEqual(variants[0], variants[2])

    def test_full_acc16_cfg5_and_matching_validation(self):
        for group in launcher.GROUPS:
            with self.subTest(group=group):
                config = self.config(group)
                training, inference = config["training"], config["inference"]
                self.assertEqual(training["max_train_iters"], 10000)
                self.assertEqual(training["gradient_accumulation_iters"], 16)
                self.assertEqual(training["student"]["train_type"], "full")
                self.assertEqual(training["fake"]["train_type"], "full")
                self.assertNotIn("lora", training["student"])
                self.assertEqual(training["teacher"], {"guidance_scale": 5.0, "cfg_norm": "none"})
                self.assertEqual(training["dmd"]["num_inference_steps"], 4)
                self.assertEqual(training["dmd"]["generation_shapes"], [{"value": [81, 480, 832]}])
                self.assertEqual(config["data"]["train"]["data_path"], config["data"]["val"]["data_path"])
                self.assertEqual(config["data"]["val"]["max_samples"], 8)
                self.assertEqual(inference["infer_every_iters"], 100)
                self.assertEqual(inference["num_inference_steps"], 4)
                self.assertEqual(config["seed"], inference["seed"])
                self.assertFalse(inference["enable_cfg"])
                self.assertFalse(config["resume"]["auto_resume"])
                self.assertEqual(config["distributed"]["fsdp2"]["size"], 2)
                self.assertFalse(config["distributed"]["sequence_parallel"]["enabled"])
                self.assertEqual(training["dmd"]["residual_head"]["fit_steps"], 16)
                self.assertEqual(training["dmd"]["residual_head"]["fit_grad_accum_steps"], 16)

    def test_fp32_means_dit_and_latents_with_tf32_disabled(self):
        config = self.config()
        model = config["model"]
        self.assertEqual(model["name"], "wan_t2v")
        self.assertEqual(model["teacher"]["name"], "wan_t2v_14b")
        self.assertEqual(model["pretrained_model_name_or_path"], PATHS["WAN_DMD_MODEL"])
        self.assertEqual(model["teacher"]["pretrained_model_name_or_path"], PATHS["WAN_DMD_TEACHER"])
        for role in (model, model["fake"], model["teacher"]):
            self.assertEqual(role["running_dtype"], "fp32")
            self.assertEqual(role["transformer_param_dtype"], "fp32")
        self.assertEqual(config["distributed"]["fsdp2"]["mixed_precision"]["param_dtype"], "fp32")
        self.assertEqual(config["training"]["dmd"]["latent_dtype"], "fp32")
        self.assertEqual(model["attention_backend"], "sdpa")
        self.assertFalse(config["training"]["allow_tf32"])
        self.assertFalse(config["inference"]["allow_tf32"])
        # Frozen text encoding remains BF16; this recipe promises FP32 DiTs.
        self.assertEqual(model["t5_dtype"], "bf16")

    def test_bf16_fallback_changes_only_three_teacher_precision_fields(self):
        for group in launcher.GROUPS:
            with self.subTest(group=group):
                fp32, bf16 = self.config(group), self.config(group, "bf16")
                teacher = bf16["model"]["teacher"]
                self.assertEqual(teacher["running_dtype"], "bf16")
                self.assertEqual(teacher["transformer_param_dtype"], "bf16")
                self.assertEqual(teacher["distributed"]["fsdp2"]["mixed_precision"]["param_dtype"], "bf16")
                teacher["running_dtype"] = teacher["transformer_param_dtype"] = "fp32"
                teacher["distributed"]["fsdp2"]["mixed_precision"]["param_dtype"] = "fp32"
                self.assertEqual(fp32, bf16)

    def test_real_runtime_routes_bf16_fsdp_to_teacher_only(self):
        config = self.config("head", "bf16")
        before = copy.deepcopy(config)
        runtime = _DmdRuntime.__new__(_DmdRuntime)
        runtime.config, runtime.model_config = config, config["model"]
        base = {key: value for key, value in config["model"].items() if key not in {"fake", "teacher"}}
        fake = runtime._build_dmd_role_config("fake", base)
        teacher = runtime._build_dmd_role_config("teacher", base)
        self.assertEqual(fake["distributed"], config["distributed"])
        self.assertEqual(fake["model"]["running_dtype"], "fp32")
        self.assertEqual(fake["model"]["pretrained_model_name_or_path"], PATHS["WAN_DMD_MODEL"])
        self.assertEqual(teacher["distributed"]["fsdp2"]["mixed_precision"]["param_dtype"], "bf16")
        self.assertEqual(teacher["distributed"]["fsdp2"]["mixed_precision"]["reduce_dtype"], "fp32")
        self.assertNotIn("distributed", teacher["model"])
        self.assertEqual(config, before)

    def test_mixed_fa3_changes_only_compute_precision_attention_and_tf32(self):
        for group in launcher.GROUPS:
            with self.subTest(group=group):
                original = self.config(group, accumulation=1)
                mixed = self.config(group, "bf16", accumulation=1, profile="bf16_fa3")
                model = mixed["model"]
                for role in (model, model["fake"], model["teacher"]):
                    self.assertEqual(role["running_dtype"], "bf16")
                    self.assertEqual(role["transformer_param_dtype"], "fp32")
                    role["running_dtype"] = "fp32"
                self.assertEqual(model["attention_backend"], "flash_attention_3")
                model["attention_backend"] = "sdpa"
                for distributed in (mixed["distributed"], model["teacher"]["distributed"]):
                    policy = distributed["fsdp2"]["mixed_precision"]
                    self.assertEqual(policy["param_dtype"], "bf16")
                    self.assertEqual(policy["reduce_dtype"], "fp32")
                    policy["param_dtype"] = "fp32"
                for section in (mixed["training"], mixed["inference"]):
                    self.assertTrue(section["allow_tf32"])
                    section["allow_tf32"] = False
                # Exact equality also protects all optimizer, small head,
                # cadence, model path, data, generation and EMA settings.
                self.assertEqual(original, mixed)

    def test_mixed_fa3_runtime_inherits_attention_and_mixed_precision_all_roles(self):
        config = self.config("head", "bf16", accumulation=1, profile="bf16_fa3")
        runtime = _DmdRuntime.__new__(_DmdRuntime)
        runtime.config, runtime.model_config = config, config["model"]
        base = {key: value for key, value in config["model"].items() if key not in {"fake", "teacher"}}
        for role in ("fake", "teacher"):
            model_config = runtime._build_dmd_role_config(role, base)
            self.assertEqual(model_config["model"]["attention_backend"], "flash_attention_3")
            self.assertEqual(model_config["model"]["running_dtype"], "bf16")
            self.assertEqual(model_config["model"]["transformer_param_dtype"], "fp32")
            self.assertEqual(model_config["distributed"]["fsdp2"]["mixed_precision"]["param_dtype"], "bf16")
        self.assertEqual(config["training"]["dmd"]["residual_head"]["fit_steps"], 1)

    def test_mixed_profile_rejects_fp32_attempt_and_unknown_profile(self):
        with self.assertRaisesRegex(ValueError, "requires BF16 compute"):
            self.config(profile="bf16_fa3")
        with self.assertRaisesRegex(ValueError, "WAN_DMD_PRECISION_PROFILE"):
            self.config(profile="typo")


class Wan14bLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "new_run"
        self.env = dict(
            PATHS, WAN_DMD_GRAD_ACCUM="16", WAN_DMD_PRECISION_PROFILE="fp32_sdpa", WAN_DMD_STUDENT_TRAIN_TYPE="full", WAN_DMD_RUN_ROOT=str(self.root), DMD_GPUS="0,1", PDMD_GPUS="2,3", HEAD_GPUS="5,6"
        )

    def main_context(self, flags=()):
        stack = ExitStack()
        stack.enter_context(patch.dict(os.environ, self.env))
        stack.enter_context(patch.object(sys, "argv", ["launcher"] + list(flags)))
        stack.enter_context(patch.object(launcher.signal, "signal"))
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        return stack

    def test_dry_run_checks_real_config_but_never_probes_writes_or_starts(self):
        with self.main_context(["--dry-run"]) as stack:
            for obj, name in ((launcher, "preflight"), (launcher, "run_attempt"), (launcher, "gpu_query"), (launcher.subprocess, "Popen"), (Path, "mkdir"), (Path, "write_text")):
                stack.enter_context(patch.object(obj, name, side_effect=AssertionError("Forbidden dry-run operation: " + name)))
            self.assertEqual(launcher.main(), 0)
        self.assertFalse(self.root.exists())

    def test_dry_run_accepts_acc1_and_rejects_invalid_accumulation(self):
        self.env["WAN_DMD_GRAD_ACCUM"] = "1"
        with self.main_context(["--dry-run"]):
            self.assertEqual(launcher.main(), 0)
        self.assertFalse(self.root.exists())
        for value in ("0", "-1", "1.5", "true", "", "garbage"):
            with self.subTest(value=value):
                self.env["WAN_DMD_GRAD_ACCUM"] = value
                with self.main_context(["--dry-run"]), self.assertRaises(SystemExit) as raised:
                    launcher.main()
                self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.root.exists())

    def test_mixed_profile_dry_run_and_wrapper_are_read_only(self):
        self.env.update(WAN_DMD_GRAD_ACCUM="1", WAN_DMD_PRECISION_PROFILE="bf16_fa3")
        with self.main_context(["--dry-run"]), patch.object(launcher, "check_fa3", side_effect=AssertionError("GPU operation")):
            self.assertEqual(launcher.main(), 0)
        wrapper = TRAIN_ROOT / "scripts/run_wan21_dmd_pdmd_head_14b_acc1_bf16_fa3_fsdp2.sh"
        environment = {**os.environ, **self.env, "WAN_DMD_PYTHON": sys.executable, "WAN_DMD_GRAD_ACCUM": "16", "WAN_DMD_PRECISION_PROFILE": "fp32_sdpa"}
        result = subprocess.run(["bash", str(wrapper), "--dry-run"], env=environment, capture_output=True, text=True, check=True)
        self.assertIn("accumulation 1", result.stdout)
        self.assertIn('"precision_profile": "bf16_fa3"', result.stdout)
        self.assertIn('"attention_backend": "flash_attention_3"', result.stdout)
        self.assertFalse(self.root.exists())

    def test_invalid_precision_profile_rejected_before_preflight(self):
        self.env["WAN_DMD_PRECISION_PROFILE"] = "auto"
        with self.main_context(["--dry-run"]), self.assertRaises(SystemExit) as raised:
            launcher.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.root.exists())

    def test_lora_wrapper_forces_lora_acc1_mixed_precision_without_launching(self):
        wrapper = TRAIN_ROOT / "scripts/run_wan21_dmd_pdmd_head_14b_lora_acc1_bf16_fa3_fsdp2.sh"
        environment = {**os.environ, **self.env, "WAN_DMD_PYTHON": sys.executable}
        result = subprocess.run(["bash", str(wrapper), "--dry-run"], env=environment, capture_output=True, text=True, check=True)
        self.assertIn("lora student", result.stdout)
        self.assertIn("full fake", result.stdout)
        self.assertIn("accumulation 1", result.stdout)
        self.assertIn('"precision_profile": "bf16_fa3"', result.stdout)
        self.assertIn('"attention_backend": "flash_attention_3"', result.stdout)
        self.assertIn("Dry run: config checked, no GPU checks/writes/training.", result.stdout)
        self.assertFalse(self.root.exists())

    def test_invalid_student_mode_rejected_before_preflight(self):
        self.env["WAN_DMD_STUDENT_TRAIN_TYPE"] = "peft"
        with self.main_context(["--dry-run"]), patch.object(launcher, "preflight", side_effect=AssertionError("GPU operation")), self.assertRaises(SystemExit) as raised:
            launcher.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.root.exists())

    def test_fa3_preflight_requires_interface_and_hopper_on_every_gpu(self):
        torch = Mock()
        torch.cuda.device_count.return_value = 2
        torch.cuda.get_device_capability.side_effect = [(9, 0), (9, 0)]
        interface = Mock(__file__="/fixture/flash_attn_interface.py")
        wan_attention = Mock(FLASH_ATTN_3_AVAILABLE=True, flash_attn_interface=interface)
        # Importing the native Wan package initializes T5's default CUDA
        # device; keep this contract test CPU-only, just like gpu_query tests.
        package = Mock(attention=wan_attention)
        with patch.dict(sys.modules, {"lightx2v_train.model_zoo.native.wan.modules": package}):
            result = launcher.check_fa3(torch)
            self.assertEqual(result["compute_capabilities"], [(9, 0), (9, 0)])
            self.assertEqual(result["module"], "/fixture/flash_attn_interface.py")
            torch.cuda.get_device_capability.side_effect = [(9, 0), (8, 0)]
            with self.assertRaisesRegex(ValueError, "Hopper GPUs"):
                launcher.check_fa3(torch)
            torch.cuda.get_device_capability.side_effect = [(9, 0), (9, 0)]
            interface.flash_attn_varlen_func = None
            with self.assertRaisesRegex(ValueError, "no FA2/SDPA fallback"):
                launcher.check_fa3(torch)
            torch.cuda.get_device_capability.side_effect = [(9, 0), (9, 0)]
            wan_attention.FLASH_ATTN_3_AVAILABLE = False
            with self.assertRaisesRegex(ValueError, "installed flash_attn_interface"):
                launcher.check_fa3(torch)

    def test_dry_run_rejects_overlapping_gpu_groups_and_existing_output(self):
        self.env["HEAD_GPUS"] = "3,6"
        with self.main_context(["--dry-run"]), self.assertRaises(SystemExit) as raised:
            launcher.main()
        self.assertEqual(raised.exception.code, 2)
        self.env["HEAD_GPUS"] = "5,6"
        self.root.mkdir()
        sentinel = self.root / "keep.txt"
        sentinel.write_text("existing user data")
        with self.main_context(["--dry-run"]), self.assertRaises(SystemExit):
            launcher.main()
        self.assertEqual(sentinel.read_text(), "existing user data")

    def test_cuda_oom_patterns_and_non_cuda_failures(self):
        log = Path(self.temp.name) / "failure.log"
        for message, expected in (
            ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB", True),
            ("RuntimeError: CUDA error: out of memory", True),
            ("torch.cuda.OutOfMemoryError: CUDA out of memory.", True),
            ("torch.OutOfMemoryError: DefaultCPUAllocator: can't allocate memory", False),
            ("RuntimeError: DefaultCPUAllocator: not enough memory", False),
            ("MemoryError: host allocation failed", False),
            ("Out of memory: Killed process 123 (python)", False),
            ("torch.distributed.DistNetworkError: connection closed by peer", False),
            ("RuntimeError: NCCL error: unhandled system error", False),
            ("ModuleNotFoundError: No module named 'ftfy'", False),
            ("KeyboardInterrupt", False),
        ):
            with self.subTest(message=message):
                log.write_text(message + "\n")
                self.assertEqual(launcher.log_has_oom(log), expected)

    def test_main_retries_all_groups_once_only_after_oom(self):
        for results, expected_dtypes, expected_code, status in (
            (["complete"], ["fp32"], 0, "complete"),
            (["failed"], ["fp32"], 1, "failed"),
            (["oom", "complete"], ["fp32", "bf16"], 0, "complete"),
            (["oom", "oom"], ["fp32", "bf16"], 1, "oom"),
        ):
            with self.subTest(results=results):
                self.root = Path(self.temp.name) / ("run_" + "_".join(results))
                self.env["WAN_DMD_RUN_ROOT"] = str(self.root)
                with self.main_context() as stack:
                    stack.enter_context(patch.object(launcher, "preflight", return_value={"fixture": True}))
                    attempt = stack.enter_context(patch.object(launcher, "run_attempt", side_effect=results))
                    self.assertEqual(launcher.main(), expected_code)
                self.assertEqual([entry.args[1] for entry in attempt.call_args_list], expected_dtypes)
                self.assertEqual(json.loads((self.root / "status.json").read_text())["status"], status)
                for entry in attempt.call_args_list:
                    self.assertEqual(entry.args[2], GPU_GROUPS)

    def test_mixed_profile_never_retries_another_precision_or_attention(self):
        self.env.update(WAN_DMD_GRAD_ACCUM="1", WAN_DMD_PRECISION_PROFILE="bf16_fa3")
        with self.main_context() as stack:
            stack.enter_context(patch.object(launcher, "preflight", return_value={"fixture": True}))
            attempt = stack.enter_context(patch.object(launcher, "run_attempt", return_value="oom"))
            self.assertEqual(launcher.main(), 1)
        attempt.assert_called_once()
        self.assertEqual(attempt.call_args.args[1], "bf16")
        status = json.loads((self.root / "status.json").read_text())
        self.assertEqual(status["status"], "oom")
        self.assertEqual(status["precision_profile"], "bf16_fa3")
        self.assertEqual(status["teacher_master_dtype"], "fp32")
        self.assertEqual(status["teacher_compute_dtype"], "bf16")
        self.assertEqual(status["attempt"], "mixed_bf16_fa3")

    def test_interrupt_does_not_trigger_bf16_retry(self):
        with self.main_context() as stack:
            stack.enter_context(patch.object(launcher, "preflight", return_value={}))
            attempt = stack.enter_context(patch.object(launcher, "run_attempt", side_effect=KeyboardInterrupt))
            self.assertEqual(launcher.main(), 1)
        self.assertEqual(attempt.call_count, 1)
        self.assertEqual(json.loads((self.root / "status.json").read_text())["status"], "interrupted")

    def test_unexpected_exception_is_recorded_as_failed_not_interrupted(self):
        with self.main_context() as stack:
            stack.enter_context(patch.object(launcher, "preflight", return_value={}))
            stack.enter_context(patch.object(launcher, "run_attempt", side_effect=OSError("fixture launch failure")))
            try:
                result = launcher.main()
            except OSError:
                pass  # Propagating the actual error is also a valid CLI failure.
            else:
                self.assertNotEqual(result, 0)
        self.assertEqual(json.loads((self.root / "status.json").read_text())["status"], "failed")

    def test_attempt_launches_only_owned_gpu_groups_with_isolated_configs(self):
        self.root.mkdir()
        processes = [Mock(pid=100 + index, returncode=0) for index in range(3)]
        for process in processes:
            process.poll.return_value = 0
        with ExitStack() as stack:
            start = stack.enter_context(patch.object(launcher.subprocess, "Popen", side_effect=processes))
            stop = stack.enter_context(patch.object(launcher, "stop_jobs"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(launcher.run_attempt(self.root, "fp32", GPU_GROUPS, PATHS), "complete")
        self.assertEqual(start.call_count, 3)
        self.assertEqual(set(stop.call_args.args[0]), set(launcher.GROUPS))
        for group, entry in zip(launcher.GROUPS, start.call_args_list):
            argv, kwargs = entry.args[0], entry.kwargs
            self.assertIn("--nproc_per_node=2", argv)
            self.assertIn("--max_restarts=0", argv)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], GPU_GROUPS[group])
            self.assertEqual(kwargs["env"]["NVIDIA_TF32_OVERRIDE"], "0")
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            config = json.loads(Path(argv[-1]).read_text())
            self.assertFalse(config["resume"]["auto_resume"])
            self.assertEqual(Path(config["training"]["output_dir"]).parent, self.root / "teacher_fp32" / group)

    def test_partial_start_failure_cleans_already_registered_jobs(self):
        self.root.mkdir()
        process = Mock(pid=123, returncode=-signal.SIGTERM)
        with ExitStack() as stack:
            stack.enter_context(patch.object(launcher.subprocess, "Popen", side_effect=[process, OSError("fixture startup failure")]))
            stop = stack.enter_context(patch.object(launcher, "stop_jobs"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            with self.assertRaises(OSError):
                launcher.run_attempt(self.root, "fp32", GPU_GROUPS, PATHS)
        self.assertEqual(set(stop.call_args.args[0]), {"dmd"})
        self.assertIs(stop.call_args.args[0]["dmd"]["process"], process)

    def test_mixed_attempt_uses_unambiguous_directory_and_enables_tf32(self):
        self.root.mkdir()
        processes = [Mock(pid=100 + index, returncode=0) for index in range(3)]
        for process in processes:
            process.poll.return_value = 0
        paths = {**PATHS, "WAN_DMD_GRAD_ACCUM": "1", "WAN_DMD_PRECISION_PROFILE": "bf16_fa3"}
        with ExitStack() as stack:
            start = stack.enter_context(patch.object(launcher.subprocess, "Popen", side_effect=processes))
            stack.enter_context(patch.object(launcher, "stop_jobs"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(launcher.run_attempt(self.root, "bf16", GPU_GROUPS, paths), "complete")
        for group, entry in zip(launcher.GROUPS, start.call_args_list):
            self.assertEqual(entry.kwargs["env"]["NVIDIA_TF32_OVERRIDE"], "1")
            config = json.loads(Path(entry.args[0][-1]).read_text())
            self.assertEqual(Path(config["training"]["output_dir"]).parent, self.root / "mixed_bf16_fa3" / group)
            self.assertEqual(config["model"]["teacher"]["transformer_param_dtype"], "fp32")
            self.assertEqual(config["model"]["attention_backend"], "flash_attention_3")
        self.assertFalse((self.root / "teacher_fp32").exists())
        self.assertFalse((self.root / "teacher_bf16").exists())

    def test_signal_during_popen_still_registers_child_and_prevents_next_launch(self):
        self.root.mkdir()
        process = Mock(pid=123, returncode=-signal.SIGTERM)

        def start_then_request_stop(*args, **kwargs):
            launcher.STOP_REQUESTED = True
            return process

        with ExitStack() as stack:
            stack.enter_context(patch.object(launcher, "STOP_REQUESTED", False))
            start = stack.enter_context(patch.object(launcher.subprocess, "Popen", side_effect=start_then_request_stop))
            stop = stack.enter_context(patch.object(launcher, "stop_jobs"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            with self.assertRaises(KeyboardInterrupt):
                launcher.run_attempt(self.root, "fp32", GPU_GROUPS, PATHS)
        self.assertEqual(start.call_count, 1)
        self.assertEqual(set(stop.call_args.args[0]), {"dmd"})
        self.assertIs(stop.call_args.args[0]["dmd"]["process"], process)

    def test_attempt_failure_classification_is_based_on_its_own_log(self):
        for index, (message, expected) in enumerate(
            (
                ("torch.OutOfMemoryError: CUDA out of memory.", "oom"),
                ("torch.OutOfMemoryError: DefaultCPUAllocator: can't allocate memory", "failed"),
                ("torch.distributed.DistNetworkError: connection reset by peer", "failed"),
            )
        ):
            with self.subTest(message=message):
                root = Path(self.temp.name) / ("attempt_" + str(index))
                root.mkdir()
                processes = [Mock(pid=100 + i, returncode=(1 if i == 0 else -15)) for i in range(3)]
                processes[0].poll.return_value = 1

                def start(*args, **kwargs):
                    kwargs["stdout"].write(message + "\n")
                    return processes.pop(0)

                with ExitStack() as stack:
                    stack.enter_context(patch.object(launcher.subprocess, "Popen", side_effect=start))
                    stop = stack.enter_context(patch.object(launcher, "stop_jobs"))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    self.assertEqual(launcher.run_attempt(root, "fp32", GPU_GROUPS, PATHS), expected)
                self.assertEqual(set(stop.call_args.args[0]), set(launcher.GROUPS))

    def test_stop_jobs_signals_only_registered_live_process_group(self):
        live, finished = Mock(pid=101), Mock(pid=202)
        live.poll.return_value, finished.poll.return_value = None, 0
        with patch.object(launcher.os, "killpg") as kill:
            launcher.stop_jobs({"dmd": {"process": live}, "pdmd": {"process": finished}})
        self.assertEqual(kill.call_args_list, [call(101, signal.SIGTERM)])
        live.wait.assert_called_once()
        finished.wait.assert_called_once()

    def test_stop_jobs_tolerates_exit_between_timeout_and_sigkill(self):
        process = Mock(pid=101)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 30), 0]
        with patch.object(launcher.os, "killpg", side_effect=[None, ProcessLookupError]):
            launcher.stop_jobs({"dmd": {"process": process}})
        self.assertEqual(process.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
