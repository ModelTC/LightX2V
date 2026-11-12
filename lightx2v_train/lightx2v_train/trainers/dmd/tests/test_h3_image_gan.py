"""Image GAN: own-caption pairing, scalar noising, role isolation and resume."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from lightx2v_train.model_zoo.minimax_h3_image.capability import MiniMaxH3ImageDistributionMatchingCapability
from lightx2v_train.trainers.dmd.dmd2_gan import Dmd2GanObjective, gan_sigmas, validate_gan_config
from lightx2v_train.trainers.dmd.dmd2_gan_data import DMD2GanRealStream
from lightx2v_train.trainers.dmd.dmd2_gan_head import H3GanHead, frozen_dmd2_gan_critic
from lightx2v_train.trainers.dmd.official_pdmd_core import student_objective
from lightx2v_train.trainers.dmd.official_pdmd_training import OfficialH3PdmdTraining
from lightx2v_train.trainers.dmd.tests.test_h3_image_official_pdmd import image_fixture


def make_image_manifest(root, count=5):
    rows = []
    for index in range(count):
        row = {
            "id": index,
            "task": "t2i",
            "prompt": f"teacher image caption {index}",
            "num_frames": 1,
            "target_height": 64,
            "target_width": 64,
            "latent_shape": [24, 1, 4, 4],
            "condition_path": f"condition-{index}.pt",
            "cache_fingerprint": f"text-cache-{index}",
            "image_latent_path": f"image-{index}.pt",
            "real_sample_key": f"sample-{index}",
            "real_cache_fingerprint": f"target-cache-{index}",
            "vae_normalized": True,
            "source": "teacher_generated",
        }
        torch.save(
            {
                "cache_schema_version": 1,
                "cache_fingerprint": row["cache_fingerprint"],
                "inputs": {},
                "conditioning": {
                    "prompt": row["prompt"],
                    "active": "positive",
                    "positive": {"prompt_embeds": torch.full((1, 2, 5120), float(index), dtype=torch.bfloat16), "text_token_tags": torch.ones(2, dtype=torch.long)},
                },
                "meta": {key: row[key] for key in ("id", "task", "num_frames", "target_height", "target_width", "latent_shape")},
            },
            root / row["condition_path"],
        )
        torch.save(
            {"latents": torch.full((24, 1, 4, 4), float(index)), "real_sample_key": row["real_sample_key"], "real_cache_fingerprint": row["real_cache_fingerprint"], "vae_normalized": True},
            root / row["image_latent_path"],
        )
        rows.append(row)
    manifest = root / "metadata.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest, rows


def gan_fixture(output="unused"):
    engine, trainer, _ = image_fixture(output)
    trainer.dmd_config["gan"] = {"feature_block": 0, "generator_weight": 0.005, "discriminator_weight": 0.01}
    trainer.dmd2_gan_head = H3GanHead(2, hidden_dim=4, temporal_bins=2)
    trainer.dmd2_gan_trainable_params = list(trainer.dmd2_gan_head.parameters())
    trainer.dmd2_gan_optimizer = torch.optim.AdamW(trainer.dmd2_gan_trainable_params, lr=0.001, weight_decay=0)
    trainer.dmd2_gan_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.dmd2_gan_optimizer, lambda _: 1)
    calls = []

    def features(state, sigma, condition, feature_block):
        calls.append((sigma.detach().clone(), condition))
        values = trainer.fake.model.predict_denoiser_output(state, sigma, condition).flatten(2).transpose(1, 2)
        return {"image": torch.cat((values, values.square()), dim=-1)}

    trainer.fake.predict_dmad_features = features
    trajectory = engine._trajectory(0, 0, 0)
    real = SimpleNamespace(load=Mock(return_value=(trajectory.states[-1], {"bias": 0.31, "id": "own-real-caption"})))
    engine.gan = Dmd2GanObjective(trainer, real)
    return engine, trainer, real, calls


class ImageGanLossTests(unittest.TestCase):
    def test_scalar_sigmas_preserve_av_random_draws(self):
        for seed in range(20):
            image = gan_sigmas(torch.Generator().manual_seed(seed), "cpu", image=True)
            av = gan_sigmas(torch.Generator().manual_seed(seed), "cpu")
            self.assertEqual(image.ndim, 0)
            torch.testing.assert_close(image.expand(2), av, atol=0, rtol=0)

    def test_generator_reaches_student_through_frozen_fake_without_parameter_gradients(self):
        engine, trainer, _, calls = gan_fixture()
        trajectory = engine._trajectory(1, 0, 0)
        with frozen_dmd2_gan_critic(trainer):
            _, _, clean = student_objective(trainer.student, trainer.fake, trainer.teacher, trajectory, 1, 0.3, return_clean=True)
            loss, _ = engine.gan.generator(clean, trajectory.condition, 5, 0)
            loss.backward()
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in trainer.trainable_params), 0)
        self.assertTrue(all(parameter.grad is None for parameter in [*trainer.fake_trainable_params, *trainer.dmd2_gan_trainable_params, *trainer.teacher.denoiser().parameters()]))
        self.assertEqual(calls[-1][0].ndim, 0)

    def test_fake5_student1_keeps_backbone_shared_and_uses_real_own_caption(self):
        engine, trainer, real, calls = gan_fixture()
        before_student = [parameter.detach().clone() for parameter in trainer.trainable_params]
        before_fake = [parameter.detach().clone() for parameter in trainer.fake_trainable_params]
        before_head = [parameter.detach().clone() for parameter in trainer.dmd2_gan_trainable_params]
        for iteration in range(5):
            role, metrics = engine._update(iter([iteration]), iteration)
            self.assertEqual(role, "fake")
            self.assertIn("gan_d_real", metrics)
        for before, current in zip(before_student, trainer.trainable_params):
            torch.testing.assert_close(before, current, atol=0, rtol=0)
        self.assertTrue(any(not torch.equal(before, current) for before, current in zip(before_fake, trainer.fake_trainable_params)))
        self.assertTrue(any(not torch.equal(before, current) for before, current in zip(before_head, trainer.dmd2_gan_trainable_params)))
        self.assertEqual([call.args[1] for call in real.load.call_args_list], list(range(5)))
        self.assertEqual([condition["id"] for _, condition in calls[1::2]], ["own-real-caption"] * 5)
        frozen = [parameter.detach().clone() for parameter in [*trainer.fake_trainable_params, *trainer.dmd2_gan_trainable_params]]
        role, metrics = engine._update(iter(()), 5)
        self.assertEqual(role, "student")
        self.assertIn("gan_g", metrics)
        self.assertTrue(any(not torch.equal(before, current) for before, current in zip(before_student, trainer.trainable_params)))
        for before, current in zip(frozen, [*trainer.fake_trainable_params, *trainer.dmd2_gan_trainable_params]):
            torch.testing.assert_close(before, current, atol=0, rtol=0)
        self.assertTrue(all(parameter.grad is None for parameter in trainer.teacher.denoiser().parameters()))


class ImageGanDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest, self.rows = make_image_manifest(self.root)

    def stream(self, **kwargs):
        return DMD2GanRealStream(self.manifest, geometry=(64, 64), task="t2i", seed=42, **kwargs)

    def test_preflight_stays_stdlib_only_and_defers_latent_loading(self):
        module = Path(__file__).parents[1] / "dmd2_gan_data.py"
        source = "import runpy,sys; m=runpy.run_path(sys.argv[1]); assert 'torch' not in sys.modules; assert len(m['load_image_gan_real_records'](sys.argv[2],geometry=(64,64))[1])==5"
        subprocess.run([sys.executable, "-c", source, str(module), str(self.manifest)], check=True)
        with patch("torch.load", wraps=torch.load) as loader:
            stream = self.stream()
            self.assertEqual(loader.call_count, 0)
            sample = stream.sample(7)
        self.assertEqual(loader.call_count, 1)
        row = stream.rows[stream.index_at(7)]
        self.assertEqual(sample["conditioning"]["prompt"], row["prompt"])
        self.assertEqual(sample["meta"]["image_latent_path"], row["image_latent_path"])
        self.assertTrue(sample["meta"]["vae_normalized"])
        self.assertEqual(sample["inputs"], {})

    def test_image_preflight_rejects_bad_pair_contract(self):
        for update, message in (
            ({"vae_normalized": False}, "normalized"),
            ({"real_sample_key": ""}, "real_sample_key"),
            ({"latent_shape": [24, 2, 4, 4]}, "latent_shape"),
            ({"image_latent_path": "missing.pt"}, "missing"),
            ({"negative_condition_path": "negative.pt"}, "negative"),
        ):
            with self.subTest(update=update):
                self.manifest.write_text(json.dumps({**self.rows[0], **update}) + "\n")
                with self.assertRaisesRegex((ValueError, FileNotFoundError), message):
                    self.stream()

    def test_real_capability_loads_matching_image_and_condition_without_a_vae(self):
        class ImageModel(SimpleNamespace):
            pass

        model = ImageModel(
            config={"model": {"capabilities": {"distribution_matching": {"official_pdmd": True}}}},
            device="cpu",
            running_dtype=torch.bfloat16,
            latent_channels=24,
            vae_scale_factor=16,
            prepare_text_condition=lambda value: value,
        )
        cap = MiniMaxH3ImageDistributionMatchingCapability(model)
        stream = self.stream()
        index = stream.index_at(7)
        clean, condition = stream.load(cap, 7)
        self.assertEqual(tuple(clean.shape), (1, 24, 1, 4, 4))
        self.assertEqual(clean.dtype, torch.float32)
        torch.testing.assert_close(clean, torch.full_like(clean, float(index)))
        torch.testing.assert_close(condition["prompt_embeds"], torch.full((1, 2, 5120), float(index), dtype=torch.bfloat16))
        payload_path = self.root / self.rows[index]["image_latent_path"]
        payload = torch.load(payload_path, weights_only=True)
        payload["real_sample_key"] = "another-image"
        torch.save(payload, payload_path)
        with self.assertRaisesRegex(ValueError, "real_sample_key"):
            stream.load(cap, 7)

    def test_mismatched_caption_and_unconditional_branch_are_rejected(self):
        for update, message in (({"prompt": "wrong caption"}, "disagree"), ({"negative": {"prompt_embeds": torch.zeros(1, 2, 5120)}}, "negative")):
            make_image_manifest(self.root, count=1)
            path = self.root / "condition-0.pt"
            payload = torch.load(path, weights_only=True)
            payload["conditioning"].update(update)
            torch.save(payload, path)
            with self.subTest(update=tuple(update)), self.assertRaisesRegex(ValueError, message):
                self.stream().sample(0)

    def test_rank_partition_and_absolute_cursor_restore(self):
        streams = [self.stream(rank=rank, world_size=3) for rank in range(3)]
        sequence = [stream.index_at(cursor) for cursor in range(20) for stream in streams]
        for offset in range(0, len(sequence), 5):
            self.assertEqual(sorted(sequence[offset : offset + 5]), list(range(5)))
        resumed = self.stream(rank=1, world_size=3)
        self.assertEqual([streams[1].index_at(cursor) for cursor in range(7, 27)], [resumed.index_at(cursor) for cursor in range(7, 27)])

    def test_official_resume_binds_image_real_manifest_and_head_recipe(self):
        engine, trainer, _ = image_fixture(str(self.root))
        trainer.config["training"]["fake"] = {"train_type": "full"}
        trainer.dmd_config.update(generation_shapes=[{"value": [64, 64]}], gan={"enabled": True, "real_cache": str(self.manifest), "head_token_bins": 4})
        validate_gan_config(trainer.config)
        engine = OfficialH3PdmdTraining(trainer)
        self.assertEqual(engine.gan_real_stream.task, "t2i")
        engine._save(0)
        checkpoint = self.root / "checkpoint-000000000"
        resumed = OfficialH3PdmdTraining(trainer)
        resumed._load_cursor(checkpoint, 0)
        self.assertEqual(resumed.consumed_microbatches, 0)
        original = copy.deepcopy(trainer.dmd_config)
        trainer.dmd_config["gan"]["head_token_bins"] = 2
        with self.assertRaisesRegex(RuntimeError, "recipe mismatch"):
            OfficialH3PdmdTraining(trainer)._load_cursor(checkpoint, 0)
        trainer.dmd_config.clear()
        trainer.dmd_config.update(original)
        self.rows[0]["real_cache_fingerprint"] = "new-target"
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        with self.assertRaisesRegex(RuntimeError, "recipe mismatch"):
            OfficialH3PdmdTraining(trainer)._load_cursor(checkpoint, 0)


if __name__ == "__main__":
    unittest.main()
