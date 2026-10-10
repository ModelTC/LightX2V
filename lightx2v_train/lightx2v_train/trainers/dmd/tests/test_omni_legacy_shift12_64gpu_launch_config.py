"""CPU-only checks for the independent 8 x 8 GPU legacy DMD8 recipe."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import yaml

TRAIN_ROOT = Path(__file__).resolve().parents[4]
CONFIG32 = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_fsdp32.yaml"
CONFIG64 = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_fsdp64.yaml"
LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_fsdp64_64gpu_acp.sh"
OVERRIDE = "H3_LEGACY_SHIFT12_FSDP64_CONFIG"


class LegacyShift12Recipe64Tests(unittest.TestCase):
    def test_only_topology_and_opt_in_lora_dtype_differ_from_32_gpu(self):
        original = yaml.safe_load(CONFIG32.read_text())
        expanded = yaml.safe_load(CONFIG64.read_text())
        self.assertEqual(expanded["distributed"]["fsdp2"]["size"], 64)
        self.assertEqual(expanded["training"]["student"]["lora"].pop("param_dtype"), "fp32")
        expanded["distributed"]["fsdp2"]["size"] = 32
        self.assertEqual(expanded, original)
        self.assertNotIn("param_dtype", original["training"]["student"]["lora"])

    def test_shell_syntax_and_no_dependency_on_32_gpu_launcher(self):
        result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("exec bash", LAUNCHER.read_text())


class LegacyShift12Launcher64Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        transformer = self.root / "model/transformer_ref"
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text("{}")
        self.manifest = self.root / "metadata.jsonl"
        self.manifest.write_text("{}\n")
        self.receipt_path = self.manifest.with_suffix(".complete.json")
        self.receipt = {
            "completed_count": 1,
            "failed_count": 0,
            "input_total_rows": 1,
            "preprocess_fingerprint": "fixture",
            "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            "reference_image_counts": {"6": 1},
        }
        self.receipt_path.write_text(json.dumps(self.receipt))
        self.env = {
            **os.environ,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29599",
            "H3_CODE_ROOT": str(TRAIN_ROOT.parent),
            "H3_PYTHON": sys.executable,
            "H3_MODEL_PATH": str(self.root / "model"),
            "H3_REF2AV_CACHE": str(self.manifest),
            "H3_REF2AV_DMD_OUTPUT": str(self.root / "output"),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            # Stale 32-GPU / PDMD environment variables must not select a recipe.
            "H3_CONFIG_PATH": "/stale/pdmd.yaml",
            "H3_LEGACY_SHIFT12_CONFIG": str(CONFIG32),
            "H3_PDMD": "true",
            "H3_REF2AV_EXPECTED_ROWS": "13553",
        }
        for name in (
            OVERRIDE,
            "H3_RDZV_ID",
            "H3_KERNEL_SNAPSHOT",
            "KERNELS_CACHE",
            "LOCAL_RANK",
            "NODE_RANK",
            "GROUP_RANK",
            "ACP_NODE_RANK",
            "NNODES",
            "NPROC_PER_NODE",
        ):
            self.env.pop(name, None)

    def run_launcher(self, overrides=None, config=None):
        environment = {**self.env, **(overrides or {})}
        if config is not None:
            path = self.root / "custom.yaml"
            path.write_text(yaml.safe_dump(config))
            environment[OVERRIDE] = str(path)
        return subprocess.run(["bash", str(LAUNCHER), "--dry-run"], env=environment, capture_output=True, text=True, timeout=30)

    def test_dry_run_has_64_gpu_topology_and_independent_output(self):
        result = self.run_launcher({"H3_REF2AV_DMD_OUTPUT": ""})
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in (
            str(CONFIG64),
            "--nnodes=8 --nproc_per_node=8",
            "FSDP64/DP64/SP1",
            "plain DMD, steps=8, iters=100000",
            "fake_update_ratio=5",
            "model_mode=legacy_train",
            "legacy_numerics=True",
            "64-row global microbatch",
            "Random orientation with no landscape/portrait quota",
            "precision student: transformer_param_dtype=bf16",
            "precision fake: transformer_param_dtype=fp32",
            "precision teacher: transformer_param_dtype=bf16",
            "student LoRA param_dtype=fp32",
            '"reduce_dtype": "fp32"',
            "Verified cache: completed=1",
            "legacy_shift12_fsdp64_lora_fp32_100k",
            "--rdzv_id=h3_ref2av_omni_imageonly_dmd8_legacy_shift12_fsdp64_lora_fp32_100k",
        ):
            self.assertIn(expected, result.stdout)
        self.assertNotIn("--node_rank", result.stdout)
        self.assertFalse((self.root / "output").exists())

    def test_explicit_node_rank_aliases_and_rendezvous_override(self):
        for name in ("NODE_RANK", "GROUP_RANK", "ACP_NODE_RANK"):
            with self.subTest(alias=name):
                result = self.run_launcher({name: "7", "H3_RDZV_ID": "fixture-64"})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--node_rank=7", result.stdout)
                self.assertIn("--rdzv_id=fixture-64", result.stdout)
        result = self.run_launcher({"RANK": "63"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--node_rank", result.stdout)

    def test_invalid_acp_launch_environment_is_rejected(self):
        for overrides, message in (
            ({"NODE_RANK": "8"}, "0..7"),
            ({"ACP_NODE_RANK": "-1"}, "0..7"),
            ({"GROUP_RANK": "not-an-integer"}, "0..7"),
            ({"NODE_RANK": "1", "GROUP_RANK": "2"}, "Conflicting ACP node ranks"),
            ({"NNODES": "4"}, "NNODES must be 8"),
            ({"NPROC_PER_NODE": "4"}, "NPROC_PER_NODE must be 8"),
            ({"MASTER_PORT": "65536"}, "1..65535"),
            ({"MASTER_PORT": "0"}, "1..65535"),
            ({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, "eight distinct GPUs"),
            ({"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,6"}, "eight distinct GPUs"),
        ):
            with self.subTest(overrides=overrides):
                result = self.run_launcher(overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_dtype_and_training_hyperparameter_overrides_are_not_pinned(self):
        for dtype in (None, "bf16", "omitted"):
            with self.subTest(dtype=dtype):
                config = yaml.safe_load(CONFIG64.read_text())
                lora = config["training"]["student"]["lora"]
                if dtype == "omitted":
                    lora.pop("param_dtype")
                else:
                    lora["param_dtype"] = dtype
                lora.update(rank=64, alpha=16)
                config["training"]["max_train_iters"] = 1000
                config["training"]["student"]["optimizer"]["learning_rate"] = 1e-4
                config["model"]["teacher"]["distributed"] = {"fsdp2": {"mixed_precision": {"param_dtype": "fp32"}}}
                result = self.run_launcher(config=config)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"student LoRA param_dtype={None if dtype == 'omitted' else dtype}", result.stdout)
                self.assertIn('"rank": 64', result.stdout)
                self.assertIn('"param_dtype": "fp32"', result.stdout)

    def test_topology_and_algorithm_overrides_are_rejected(self):
        result = self.run_launcher({OVERRIDE: str(CONFIG32)})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires FSDP64", result.stderr)
        mutations = (
            lambda c: c["distributed"]["sequence_parallel"].update(size=2),
            lambda c: c["distributed"].update(dp={"enabled": True}),
            lambda c: c["training"]["dmd"].update(num_inference_steps=4),
            lambda c: c["model"]["capabilities"]["distribution_matching"].update(projected_dmd=True),
            lambda c: c["model"]["capabilities"]["distribution_matching"].update(official_pdmd=True),
            lambda c: c["training"]["dmd"].update(official_pdmd=True),
        )
        for mutate in mutations:
            config = yaml.safe_load(CONFIG64.read_text())
            mutate(config)
            result = self.run_launcher(config=config)
            self.assertNotEqual(result.returncode, 0)

    def test_model_kernel_and_cache_receipt_checks_remain_required(self):
        for overrides, message in (
            ({"H3_MODEL_PATH": str(self.root / "missing")}, "Missing Ref2AV transformer config"),
            ({"H3_KERNEL_SNAPSHOT": str(self.root / "missing")}, "Missing FlashAttention-3 snapshot"),
        ):
            result = self.run_launcher(overrides)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)
        for change, message in (
            ({"input_total_rows": 2}, "Invalid completed/failed/source accounting"),
            ({"preprocess_fingerprint": ""}, "Missing preprocessing fingerprint"),
            ({"manifest_sha256": "bad"}, "no longer matches"),
        ):
            self.receipt_path.write_text(json.dumps({**self.receipt, **change}))
            result = self.run_launcher()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)
        self.receipt_path.unlink()
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing completed cache merge receipt", result.stderr)


class LegacyShift12Sampling64Tests(unittest.TestCase):
    def test_64_ranks_use_one_count_natural_orientation_and_resume_exactly(self):
        from lightx2v_train.data.minimax_h3_cache_dataset import MiniMaxH3ReferenceCostSampler

        config = yaml.safe_load(CONFIG64.read_text())
        options = config["data"]["train"]["reference_cost_sampler"]
        rows = []
        for count in range(1, 7):
            for orientation in ("landscape", "portrait"):
                for offset in range((128 if orientation == "landscape" else 32) + count):
                    rows.append(
                        {
                            "condition_path": f"/unused/{len(rows)}.pt",
                            "target_orientation": orientation,
                            "reference_image_count": count,
                            "reference_video_count": 0,
                            "reference_audio_count": 0,
                            "packed_sequence_tokens_124": count * 10000 + offset,
                        }
                    )

        class Dataset:
            samples = [{"type": "metadata", "row": row} for row in rows]

        def build(rank, start=0):
            sampler = MiniMaxH3ReferenceCostSampler(Dataset(), num_replicas=64, rank=rank, **options)
            sampler.configure(start_iteration=start, gradient_accumulation_iters=1, fake_update_ratio=5)
            return sampler

        samplers = [build(rank) for rank in range(64)]
        self.assertEqual(samplers[0].samples_per_outer_iteration, 6)
        self.assertEqual(options["batch_mode"], "count_random")
        self.assertIs(options["balance_orientation"], False)
        counts = Counter()
        orientations = Counter()
        batch_orientation_counts = []
        for ordinal in range(samplers[0].num_global_batches):
            batch = [sampler.sample_index(ordinal) for sampler in samplers]
            self.assertEqual(len(set(batch)), 64)
            batch_rows = [rows[index] for index in batch]
            batch_counts = {row["reference_image_count"] for row in batch_rows}
            self.assertEqual(len(batch_counts), 1)
            counts.update(batch_counts)
            observed = Counter(row["target_orientation"] for row in batch_rows)
            batch_orientation_counts.append(observed)
            orientations.update(observed)
        self.assertEqual(len(counts), 6)
        self.assertEqual(len(set(counts.values())), 1)
        self.assertGreater(orientations["landscape"], orientations["portrait"])
        self.assertTrue(any(observed != {"landscape": 32, "portrait": 32} for observed in batch_orientation_counts))
        for rank, sampler in enumerate(samplers):
            resumed = build(rank, start=1)
            self.assertEqual(list(resumed), [sampler.sample_index(index) for index in range(6, 6 + len(resumed))])


if __name__ == "__main__":
    unittest.main()
