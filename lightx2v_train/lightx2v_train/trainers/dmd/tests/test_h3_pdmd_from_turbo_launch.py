"""Read-only ACP dry runs catch missing or incompatible Turbo EMA exports."""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

TRAIN_ROOT = Path(__file__).resolve().parents[4]
CONFIG = TRAIN_ROOT / "configs/train/dmd/minimax_h3_fl2av_3task_pdmd8_from_turbo_ema_shift6_3_r128_a128_adaln16_fsdp32.yaml"
LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_fl2av_pdmd8_from_turbo_ema_fsdp32_32gpu_acp.sh"
PREFLIGHT = TRAIN_ROOT / "scripts/minimax_h3_pdmd_from_turbo_preflight.py"


def write_adapter(path, *, output_rank=16, omit_output=False, extra=False):
    shapes = {}
    targets = [("transformer_blocks.0.attn.to_q", 128), ("transformer_blocks.0.adaln_proj.linear", 16)]
    if not omit_output:
        targets.append(("norm_out.linear", output_rank))
    for target, rank in targets:
        shapes[f"transformer.{target}.lora.down.weight"] = [rank, 2]
        shapes[f"transformer.{target}.lora.up.weight"] = [2, rank]
    if extra:
        shapes["transformer.proj_in.weight"] = [2, 2]
    header = {}
    offset = 0
    for key, shape in shapes.items():
        size = shape[0] * shape[1] * 4
        header[key] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


class PdmdFromTurboLaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "model/transformer").mkdir(parents=True)
        (self.root / "model/transformer/config.json").write_text("{}")
        (self.root / "condition.pt").write_bytes(b"CPU preflight must not load this condition file")
        for task in ("t2av", "i2av", "fl2av"):
            for bucket, height, width in (("landscape", 768, 1344), ("portrait", 1344, 768)):
                directory = self.root / "cache" / task / bucket
                directory.mkdir(parents=True)
                row = {"task": task, "target_height": height, "target_width": width, "target_num_frames": 124, "condition_path": str(self.root / "condition.pt")}
                (directory / "metadata.jsonl").write_text((json.dumps(row) + "\n") * 32)
        self.source = self.root / "stable_student_ema"
        self.source.mkdir()
        self.weights = self.source / "pytorch_lora_weights.safetensors"
        write_adapter(self.weights)
        lora = yaml.safe_load(CONFIG.read_text())["training"]["student"]["lora"]
        self.sidecar = {
            "r": 128,
            "lora_alpha": 128,
            "rank_pattern": {"adaln_proj.linear": 16, "norm_out.linear": 16},
            "alpha_pattern": {"adaln_proj.linear": 16, "norm_out.linear": 16},
            "target_modules": lora["target_modules"],
            "use_rslora": False,
            "use_dora": False,
            "bias": "none",
        }
        (self.source / "adapter_config.json").write_text(json.dumps(self.sidecar))
        self.environment = {
            **os.environ,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29999",
            "H3_CODE_ROOT": str(TRAIN_ROOT.parent),
            "H3_PYTHON": sys.executable,
            "H3_MODEL_PATH": str(self.root / "model"),
            "H3_FL2AV_CACHE": str(self.root / "cache"),
            "H3_STUDENT_INIT": str(self.source),
            "H3_FL2AV_PDMD_OUTPUT": str(self.root / "pdmd_output"),
            "H3_FL2AV_PDMD_CONFIG": "/stale/old_pdmd_config.yaml",
            "H3_RDZV_ID": "",
            "LOCAL_KERNELS": "",
            "KERNELS_CACHE": "",
            "H3_KERNEL_SNAPSHOT": "",
        }
        self.environment.pop("H3_PDMD_FROM_TURBO_CONFIG", None)

    def run_launcher(self, **overrides):
        return subprocess.run(["bash", str(LAUNCHER), "--dry-run"], env={**self.environment, **overrides}, capture_output=True, text=True, timeout=30)

    def test_cpu_dry_run_selects_fresh_pdmd_and_is_read_only(self):
        before = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in (
            CONFIG.name,
            "--nnodes=4 --nproc_per_node=8",
            "h3_fl2av_pdmd8_from_turbo_dcm1700_ema_32gpu_20261029",
            "optimizer_updates=15000, student_updates=2500, critic_updates=12500",
            "save_every_iters=300",
            "LoRA tensors=6",
            "block AdaLN and norm_out.linear rank=16 alpha=16",
            "student=source EMA; frozen teacher=H3 base; fake=H3 base; optimizers start fresh",
            "tasks=T2AV/I2AV/FL2AV",
            "Training rollout shifts: video=6.0, audio=3.0",
        ):
            self.assertIn(expected, result.stdout)
        self.assertNotIn("/stale/", result.stdout)
        self.assertFalse((self.root / "pdmd_output").exists())
        after = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_accepts_direct_safetensors_file_and_missing_optional_sidecar(self):
        (self.source / "adapter_config.json").unlink()
        result = self.run_launcher(H3_STUDENT_INIT=str(self.weights))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_export_fails_before_torchrun(self):
        result = self.run_launcher(H3_STUDENT_INIT=str(self.root / "missing"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing student initialization LoRA safetensors", result.stderr)
        self.assertNotIn("--nnodes", result.stdout)

    def test_incompatible_rank_output_target_and_full_weights_fail(self):
        for options, expected in (
            ({"output_rank": 128}, "LoRA rank mismatch"),
            ({"omit_output": True}, "must include regular, block AdaLN, and norm_out.linear"),
            ({"extra": True}, "must contain LoRA-only tensors"),
        ):
            with self.subTest(options=options):
                write_adapter(self.weights, **options)
                result = self.run_launcher()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
                self.assertNotIn("--nnodes", result.stdout)

    def test_truncated_safetensors_fails_without_loading_tensor_payload(self):
        self.weights.write_bytes(self.weights.read_bytes()[:-4])
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Truncated or invalid LoRA tensor data", result.stderr)

    def test_source_scaling_mismatch_fails(self):
        self.sidecar["alpha_pattern"]["norm_out.linear"] = 128
        (self.source / "adapter_config.json").write_text(json.dumps(self.sidecar))
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("adapter_config.json alpha_pattern", result.stderr)

    def test_config_without_output_adaln_cannot_silently_drop_source_weights(self):
        config = yaml.safe_load(CONFIG.read_text())
        lora = config["training"]["student"]["lora"]
        lora["target_modules"].remove("norm_out.linear")
        del lora["rank_pattern"]["norm_out.linear"]
        del lora["alpha_pattern"]["norm_out.linear"]
        custom = self.root / "wrong_targets.yaml"
        custom.write_text(yaml.safe_dump(config))
        result = self.run_launcher(H3_PDMD_FROM_TURBO_CONFIG=str(custom))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("block AdaLN AND norm_out.linear", result.stderr)

    def test_preflight_import_does_not_import_torch(self):
        code = "import importlib.util, sys; spec=importlib.util.spec_from_file_location('preflight',sys.argv[1]); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code, str(PREFLIGHT)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_shell_syntax(self):
        result = subprocess.run(["bash", "-n", str(LAUNCHER)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
