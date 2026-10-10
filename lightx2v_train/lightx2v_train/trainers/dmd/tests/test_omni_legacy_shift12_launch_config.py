"""Legacy H3 shift12 hyperparameters + new Omni data, without a GPU dependency.

Config/shell checks need only PyYAML. Actual launcher dry-runs additionally
need the normal training environment (torch, OmegaConf, loguru), but no GPUs.
"""

import hashlib
import importlib.util
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
CONFIG = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_fsdp32.yaml"
LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_fsdp32_32gpu_acp.sh"
HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("omegaconf", "torch", "loguru"))


class LegacyShift12RecipeTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_plain_dmd_legacy_cadence_and_training_mode(self):
        config = self.config
        training = config["training"]
        dmd = training["dmd"]
        self.assertEqual(training["method"], "dmd")
        self.assertEqual(dmd["num_inference_steps"], 8)
        self.assertEqual(dmd["update_order"], "student_first")
        self.assertEqual(dmd["fake_update_ratio"], 5)
        self.assertEqual(dmd["model_mode"], "legacy_train")
        self.assertIs(dmd["residual_head"]["enabled"], False)
        self.assertIs(training["student"]["ema"]["enabled"], False)
        self.assertIs(config["model"]["capabilities"]["distribution_matching"]["projected_dmd"], False)
        # Literal settings, not an H3_PDMD-dependent recipe selection.
        self.assertNotIn("oc.env:H3_PDMD", CONFIG.read_text(encoding="utf-8"))

    def test_original_optimizer_lora_and_training_duration(self):
        training = self.config["training"]
        for key, expected in {
            "max_train_iters": 100000,
            "gradient_accumulation_iters": 1,
            "gradient_checkpointing": True,
            "max_grad_norm": 1.0,
            "lr_scheduler": "constant",
            "lr_warmup_iters": 0,
            "save_every_iters": 50,
            "save_total_limit": 5,
        }.items():
            self.assertEqual(training[key], expected, key)
        self.assertEqual(training["student"]["train_type"], "lora")
        self.assertEqual(
            training["student"]["lora"],
            {
                "rank": 128,
                "alpha": 8,
                "target_modules": ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
            },
        )
        self.assertEqual(training["fake"]["train_type"], "full")
        for role, learning_rate in (("student", 5e-5), ("fake", 4e-7)):
            self.assertEqual(
                training[role]["optimizer"],
                {
                    "learning_rate": learning_rate,
                    "adam_beta1": 0.0,
                    "adam_beta2": 0.999,
                    "weight_decay": 0.01,
                    "adam_epsilon": 1e-8,
                },
            )
        self.assertEqual(training["teacher"], {"guidance_scale": 1.0, "cfg_norm": "none"})

    def test_shifted_noise_geometry_and_normalization(self):
        matching = self.config["model"]["capabilities"]["distribution_matching"]
        self.assertIs(matching["legacy_numerics"], True)
        self.assertEqual(matching["video_flow_shift"], 12.0)
        self.assertEqual(matching["audio_flow_shift"], 3.0)
        for key in ("video_loss_weight", "audio_loss_weight", "audio_dmd_loss_weight"):
            self.assertEqual(matching[key], 1.0)
        self.assertIs(matching["dmd_normalization"], True)
        self.assertEqual(matching["dmd_normalization_epsilon"], 0.0)
        self.assertEqual(matching["dmd_reduction"], "mean")
        self.assertIs(matching["geometry_from_metadata"], True)
        self.assertEqual(matching["allowed_resolutions"], [[768, 1344], [1344, 768]])
        dmd = self.config["training"]["dmd"]
        self.assertEqual(dmd["latent_dtype"], "fp32")
        self.assertEqual(dmd["generation_shapes"], [{"value": [124, 768, 1344]}])
        self.assertEqual(
            dmd["score_sampling"],
            {
                "type": "h3_shifted_uniform",
                "legacy_numerics": "${model.capabilities.distribution_matching.legacy_numerics}",
                "video_flow_shift": "${model.capabilities.distribution_matching.video_flow_shift}",
                "audio_flow_shift": "${model.capabilities.distribution_matching.audio_flow_shift}",
                "discrete_samples": 1000,
                "min_sigma": 0.02,
                "max_sigma": 1.0,
            },
        )

    def test_legacy_role_precision_and_32_gpu_topology(self):
        model = self.config["model"]
        self.assertEqual(model["transformer_param_dtype"], "bf16")
        self.assertEqual(model["fake"]["transformer_param_dtype"], "fp32")
        self.assertEqual(model["teacher"]["transformer_param_dtype"], "bf16")
        self.assertEqual(model["running_dtype"], "bf16")
        self.assertIs(model["use_autocast"], False)
        self.assertEqual(model["attention_backend"], "_flash_3_hub")
        distributed = self.config["distributed"]
        self.assertEqual(distributed["sequence_parallel"], {"enabled": False, "size": 1})
        fsdp = distributed["fsdp2"]
        self.assertIs(fsdp["enabled"], True)
        self.assertEqual(fsdp["size"], 32)
        self.assertIs(fsdp["stream_load_pretrained"], True)
        self.assertEqual(fsdp["reshard_after_forward"], {"root_reshard": False, "block_reshard": True})
        self.assertEqual(
            fsdp["mixed_precision"],
            {
                "param_dtype": "bf16",
                "reduce_dtype": "fp32",
                "output_dtype": None,
                "cast_forward_inputs": False,
            },
        )

    def test_count_random_sampling_uses_new_dataset_including_six_images(self):
        data = self.config["data"]["train"]
        self.assertEqual(data["name"], "minimax_h3_ref_cache_dataset")
        self.assertEqual(data["batch_size"], 1)
        self.assertEqual(data["num_workers"], 0)
        self.assertIs(data["pin_memory"], False)
        self.assertIs(data["shuffle"], False)
        self.assertIs(data["drop_last"], True)
        sampler = data["reference_cost_sampler"]
        self.assertEqual(sampler["batch_mode"], "count_random")
        self.assertIs(sampler["require_image_only"], True)
        self.assertEqual(sampler["image_counts"], list(range(1, 7)))
        self.assertIs(sampler["require_all_image_counts"], False)
        self.assertIs(sampler["balance_image_counts"], True)
        self.assertIs(sampler["balance_orientation"], False)
        self.assertIs(sampler["strict_full_epoch"], False)
        self.assertEqual(sampler["remainder_policy"], "rotating_drop")
        self.assertEqual(sampler["cost_key"], "packed_sequence_tokens_124")
        self.assertEqual(sampler["seed"], 42)
        script = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl", script)
        self.assertIn("unset H3_REF2AV_EXPECTED_ROWS", script)

    def test_shell_syntax(self):
        result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(HAS_RUNTIME, "Launcher dry-run needs torch, OmegaConf and loguru (no GPU needed)")
class LegacyShift12LauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        transformer = self.root / "model/transformer_ref"
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text("{}", encoding="utf-8")
        self.manifest = self.root / "metadata.jsonl"
        self.manifest.write_text("{}\n", encoding="utf-8")
        self.receipt = self.manifest.with_suffix(".complete.json")
        self.receipt.write_text(
            json.dumps(
                {
                    "completed_count": 1,
                    "failed_count": 0,
                    "input_total_rows": 1,
                    "preprocess_fingerprint": "test",
                    "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                    "reference_image_counts": {"6": 1},
                }
            ),
            encoding="utf-8",
        )
        self.env = {
            **os.environ,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29599",
            "H3_CODE_ROOT": str(TRAIN_ROOT.parent),
            "H3_PYTHON": sys.executable,
            "H3_MODEL_PATH": str(self.root / "model"),
            "H3_REF2AV_CACHE": str(self.manifest),
            "H3_REF2AV_DMD_OUTPUT": str(self.root / "output"),
            # Neither stale value may hijack this standalone recipe.
            "H3_CONFIG_PATH": "/stale/pdmd.yaml",
            "H3_PDMD": "true",
            "H3_REF2AV_EXPECTED_ROWS": "13553",
        }
        for name in ("H3_LEGACY_SHIFT12_CONFIG", "H3_RDZV_ID", "H3_KERNEL_SNAPSHOT", "KERNELS_CACHE"):
            self.env.pop(name, None)

    def run_launcher(self):
        return subprocess.run(["bash", str(LAUNCHER), "--dry-run"], env=self.env, capture_output=True, text=True, timeout=30)

    def test_dry_run_is_isolated_and_reports_actual_settings(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        for value in (
            str(CONFIG),
            "Verified cache: completed=1",
            "plain DMD, steps=8, iters=100000",
            "legacy_numerics=True",
            "x0 reconstruction remains FP32",
            "model_mode=legacy_train, update_order=student_first, fake_update_ratio=5",
            "PDMD=false",
            "--nnodes=4 --nproc_per_node=8",
            "--rdzv_id=h3_ref2av_omni_imageonly_dmd8_legacy_shift12_100k",
            "precision student: transformer_param_dtype=bf16",
            "precision fake: transformer_param_dtype=fp32",
            "precision teacher: transformer_param_dtype=bf16",
            '"batch_mode": "count_random"',
            "one of the observed 1..6 image counts per 32-row global microbatch",
            "orientations sampled naturally within that count",
        ):
            self.assertIn(value, result.stdout)
        self.assertNotIn("16 landscape + 16 portrait", result.stdout)
        self.assertNotIn("count/orientation cell", result.stdout)
        self.assertFalse((self.root / "output").exists())

    def test_rejects_changed_cache_receipt(self):
        self.manifest.write_text("{}\n{}\n", encoding="utf-8")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no longer matches", result.stderr)

    def test_default_output_is_distinct_from_pdmd_head(self):
        self.env.pop("H3_REF2AV_DMD_OUTPUT")
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("outputs/minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_100k", result.stdout)


@unittest.skipUnless(HAS_RUNTIME, "Sampler integration test needs the training runtime (no GPU needed)")
class LegacyShift12SamplingTests(unittest.TestCase):
    def test_32_ranks_use_single_count_natural_orientations_and_resume_exactly(self):
        from lightx2v_train.data.minimax_h3_cache_dataset import MiniMaxH3ReferenceCostSampler

        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        sampler_options = config["data"]["train"]["reference_cost_sampler"]
        rows = []
        # Unequal, non-divisible count buckets exercise rotating-drop. Count
        # one is all landscape; others have fewer than 16 portrait rows, so
        # successful sampling must not require fixed orientation quotas.
        for image_count in range(1, 7):
            for offset in range(65 + 32 * (image_count - 1)):
                rows.append(
                    {
                        "condition_path": f"/unused/condition_{len(rows):04d}.pt",
                        "target_orientation": "portrait" if image_count > 1 and offset < image_count else "landscape",
                        "reference_image_count": image_count,
                        "reference_video_count": 0,
                        "reference_audio_count": 0,
                        "packed_sequence_tokens_124": image_count * 10000 + offset,
                    }
                )

        class Dataset:
            samples = [{"type": "metadata", "row": row} for row in rows]

        def build(rank, start_iteration=0):
            sampler = MiniMaxH3ReferenceCostSampler(Dataset(), num_replicas=32, rank=rank, **sampler_options)
            sampler.configure(
                start_iteration=start_iteration,
                gradient_accumulation_iters=config["training"]["gradient_accumulation_iters"],
                fake_update_ratio=config["training"]["dmd"]["fake_update_ratio"],
            )
            return sampler

        samplers = [build(rank) for rank in range(32)]
        first = samplers[0]
        self.assertIsNone(first.rows_per_image_orientation_cell)
        self.assertIsNone(first.rows_per_orientation)
        self.assertEqual(first.rows_per_image_count, dict.fromkeys(range(1, 7), 64))
        self.assertEqual(first.checkpoint_metadata()["orientation_sampling"], "natural_within_count")
        self.assertEqual(first.num_global_batches, 12)
        self.assertEqual(first.samples_per_outer_iteration, 6)  # student + 5 fake
        covered = set()
        for epoch in range(4):
            selected = []
            counts = Counter()
            for step in range(first.num_global_batches):
                ordinal = epoch * first.num_global_batches + step
                batch = [sampler.sample_index(ordinal) for sampler in samplers]
                batch_rows = [rows[index] for index in batch]
                self.assertEqual(len(set(batch)), 32)
                batch_counts = {row["reference_image_count"] for row in batch_rows}
                self.assertEqual(len(batch_counts), 1)
                self.assertLessEqual(sum(row["target_orientation"] == "portrait" for row in batch_rows), 6)
                counts.update(batch_counts)
                selected.extend(batch)
            self.assertEqual(counts, {image_count: 2 for image_count in range(1, 7)})
            self.assertEqual(len(selected), len(set(selected)))
            covered.update(selected)
        self.assertEqual(covered, set(range(len(rows))))

        # Resume partway through a data epoch: the next student/fake draws
        # must be identical on every rank, not restart the count schedule.
        offset = 3 * first.samples_per_outer_iteration
        for rank, sampler in enumerate(samplers):
            expected = [sampler.sample_index(i) for i in range(offset, offset + len(sampler))]
            self.assertEqual(list(build(rank, start_iteration=3)), expected)


if __name__ == "__main__":
    unittest.main()
