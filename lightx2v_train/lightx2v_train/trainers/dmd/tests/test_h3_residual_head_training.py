"""CPU checks of joint H3 residual-head training, without loading a DiT."""

import copy
import os
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from lightx2v_train.model_zoo.minimax_h3.capability_adapters.common import (
    MiniMaxH3JointLatents,
    MiniMaxH3LatentShape,
)
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability import (
    MiniMaxH3DistributionMatchingCapability,
)
from lightx2v_train.trainers.dmd.h3_residual_head_training import H3ResidualHeadTraining
from lightx2v_train.trainers.dmd.residual_head import ResidualHeadConfig
from lightx2v_train.trainers.dmd.trainer import DmdTrainer


class TinyH3Model:
    transformer_component = "transformer_ref"
    device = torch.device("cpu")

    def __init__(self):
        self.config = {"training": {"dmd": {"num_inference_steps": 8}}}
        self.transformer = torch.nn.Linear(1, 1, bias=False)

    def denoiser_module(self):
        return self.transformer


class H3ResidualHeadTrainingTest(unittest.TestCase):
    def setUp(self):
        quiet_logs = patch("lightx2v_train.trainers.dmd.residual_head_training.logger")
        quiet_logs.start()
        self.addCleanup(quiet_logs.stop)

    def fixture(self, **config_updates):
        shape = MiniMaxH3LatentShape(124, 1, 2, 2, 3, (1, 2, 3), (1, 3, 2))
        student_parameter = torch.nn.Parameter(torch.tensor(0.2))
        fake_parameter = torch.nn.Parameter(torch.tensor(0.0))
        student = MiniMaxH3DistributionMatchingCapability(TinyH3Model(), {"video_flow_shift": 12.0, "audio_flow_shift": 3.0})

        def joint(value):
            if not torch.is_tensor(value):
                value = torch.tensor(float(value))
            return MiniMaxH3JointLatents(value.expand(shape.video_tokens), value.expand(shape.audio_tokens), shape)

        def features(latents):
            return {
                "video": torch.cat((latents.video, latents.video[..., :1]), dim=-1).detach(),
                "audio": torch.cat((latents.audio, latents.audio), dim=-1).detach(),
            }

        def predict(latents, sigma, condition):
            del sigma, condition
            return joint(fake_parameter), features(latents)

        def modality_sigmas(sigma):
            video, audio = student._modality_sigmas(sigma)
            return {"video": video, "audio": audio}

        fake_model = SimpleNamespace(
            predict_velocity_with_features=Mock(side_effect=predict),
            residual_head_feature_dims={"video": 4, "audio": 4},
            residual_head_output_dims={"video": 3, "audio": 2},
            residual_head_sigmas=modality_sigmas,
        )

        def rollout(condition, dimensions, grad_enabled, xt=None, student_query=False):
            del condition, dimensions, xt, student_query
            generated = joint(student_parameter)
            return generated if grad_enabled else student.detach(generated), 1000, 0

        trainer = SimpleNamespace(
            trainer_name="dmd",
            student=student,
            fake=SimpleNamespace(set_training=Mock()),
            fake_model=fake_model,
            config={"seed": 42},
            scheduler=object(),
            latent_dtype=torch.float32,
            guidance_scale=1.0,
            cfg_norm="none",
            _encode_conditions=Mock(side_effect=lambda sample: ({"id": sample["id"]}, None)),
            _latent_shape=Mock(return_value=shape),
            sample_initial_latents=Mock(side_effect=lambda dimensions: student.initial_latents(dimensions, torch.float32, lambda value: value)),
            _sample_score_sigma=Mock(return_value=torch.tensor([0.25])),
            run_back_simulation=Mock(side_effect=rollout),
        )
        config = ResidualHeadConfig(
            **{
                "enabled": True,
                "gate_mode": "calibrated",
                "hidden_dim": 4,
                "fit_steps": 2,
                "fit_grad_accum_steps": 2,
                "noise_bins": 5,
                "learning_rate": 0.02,
                "min_checks": 1,
                **config_updates,
            }
        )
        runtime = H3ResidualHeadTraining(trainer, config)
        trainer.residual_head = runtime
        return trainer, runtime, student_parameter, fake_parameter, joint, features

    def assertTreeEqual(self, actual, expected):
        if torch.is_tensor(actual):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assertTreeEqual(actual[key], expected[key])
        elif isinstance(actual, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assertTreeEqual(left, right)
        else:
            self.assertEqual(actual, expected)

    @staticmethod
    def set_corrections(runtime, video, audio):
        with torch.no_grad():
            runtime.modalities["video"].module.output_projection.bias.fill_(video)
            runtime.modalities["audio"].module.output_projection.bias.fill_(audio)

    def test_initialization_preserves_rng_and_outputs_zero_typed_joint_correction(self):
        # Fixture model construction uses RNG; isolate head construction itself.
        trainer, _, _, _, joint, _ = self.fixture()
        torch.manual_seed(813)
        before = torch.random.get_rng_state()
        runtime = H3ResidualHeadTraining(trainer, ResidualHeadConfig(enabled=True, gate_mode="calibrated", hidden_dim=4))
        torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
        for modality in runtime.modalities.values():
            self.assertTrue(all(parameter.dtype == torch.float32 for parameter in modality.module.parameters()))
            modality.gate.lambdas.fill_(1.0)
        noised = joint(1.0)
        corrected = runtime.predict_corrected_fake(noised, torch.tensor([0.25]), {})
        self.assertIsInstance(corrected, MiniMaxH3JointLatents)
        self.assertEqual(corrected.shape, noised.shape)
        torch.testing.assert_close(corrected.video, noised.video, rtol=0, atol=0)
        torch.testing.assert_close(corrected.audio, noised.audio, rtol=0, atol=0)

    def test_rejects_projection_sequence_parallel_and_unsupported_trainer(self):
        trainer, runtime, _, _, _, _ = self.fixture()
        config = runtime.config
        trainer.student = MiniMaxH3DistributionMatchingCapability(TinyH3Model(), {"projected_dmd": True})
        with self.assertRaisesRegex(ValueError, "projected_dmd"):
            H3ResidualHeadTraining(trainer, config)
        trainer.student = MiniMaxH3DistributionMatchingCapability(TinyH3Model())
        with patch("lightx2v_train.trainers.dmd.residual_head_training.get_sequence_parallel_world_size", return_value=2):
            with self.assertRaisesRegex(ValueError, "sequence_parallel"):
                H3ResidualHeadTraining(trainer, config)
        trainer.trainer_name = "unsupported"
        with self.assertRaisesRegex(ValueError, "standard dmd"):
            H3ResidualHeadTraining(trainer, config)

    def test_rejects_reference_rows_in_target_features_and_mismatched_latent_geometry(self):
        trainer, runtime, _, _, joint, features = self.fixture()

        def with_reference_rows(latents, sigma, condition):
            del sigma, condition
            output = features(latents)
            output["video"] = torch.cat((torch.zeros(1, 1, 4), output["video"]), dim=1)
            return joint(0), output

        trainer.fake_model.predict_velocity_with_features.side_effect = with_reference_rows
        with self.assertRaisesRegex(ValueError, "only generated rows"):
            runtime.predict_corrected_fake(joint(1), torch.tensor([0.25]), {})
        invalid = joint(1)
        invalid = MiniMaxH3JointLatents(invalid.video[:, :1], invalid.audio, invalid.shape)
        with self.assertRaisesRegex(ValueError, "generated token geometry"):
            runtime.predict_corrected_fake(invalid, torch.tensor([0.25]), {})

    def test_fit_rescores_joint_queries_once_and_updates_only_both_heads(self):
        trainer, runtime, student_parameter, fake_parameter, joint, _ = self.fixture()
        before = {name: copy.deepcopy(modality.module.state_dict()) for name, modality in runtime.modalities.items()}
        runtime.collecting_fit = True
        for index in range(5):
            runtime.remember_fit(joint(student_parameter), joint(1 + index), torch.tensor([0.25]), {"id": index})
        runtime.collecting_fit = False
        with torch.no_grad():
            fake_parameter.fill_(2.0)
        loss = runtime.fit()
        self.assertTrue(torch.isfinite(torch.as_tensor(loss)))
        self.assertEqual(trainer.fake_model.predict_velocity_with_features.call_count, 2)
        self.assertEqual([entry.args[2]["id"] for entry in trainer.fake_model.predict_velocity_with_features.call_args_list], [0, 1])
        for name, modality in runtime.modalities.items():
            self.assertTrue(modality.optimizer.state)
            self.assertEqual(modality._head_optimizer_updates, 1)
            self.assertEqual(modality._head_fit_microbatches, 2)
            self.assertTrue(any(not torch.equal(value, modality.module.state_dict()[key]) for key, value in before[name].items()))
            self.assertTrue(all(parameter.grad is None for parameter in modality.module.parameters()))
        self.assertIsNone(student_parameter.grad)
        self.assertIsNone(fake_parameter.grad)
        # Cache must not survive one fixed-fake fitting phase.
        self.assertEqual(runtime.fit_batches, [])

    def test_correction_uses_cleanward_x0_and_separate_shifted_sigma_gates(self):
        trainer, runtime, _, fake_parameter, joint, _ = self.fixture()
        with torch.no_grad():
            fake_parameter.fill_(2.0)
        self.set_corrections(runtime, 0.4, 0.6)
        runtime.modalities["video"].gate.lambdas.copy_(torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4]))
        runtime.modalities["audio"].gate.lambdas.copy_(torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]))
        base_sigma = torch.tensor([0.25])
        physical = trainer.fake_model.residual_head_sigmas(base_sigma)
        torch.testing.assert_close(physical["video"], torch.tensor([0.8]))
        torch.testing.assert_close(physical["audio"], torch.tensor([0.5]))
        noised = joint(1.0)
        corrected = runtime.predict_corrected_fake(noised, base_sigma, {})
        for name, amplitude in (("video", 0.4), ("audio", 0.6)):
            sigma = physical[name]
            lam = runtime.modalities[name].gate.lambda_for(sigma)
            expected = getattr(noised, name) + sigma.item() * 2.0 - lam.item() * amplitude
            torch.testing.assert_close(getattr(corrected, name), expected)
            self.assertFalse(getattr(corrected, name).requires_grad)
        # Base-sigma indexing would choose bin 1 rather than bins 4/2.
        self.assertNotEqual(runtime.modalities["video"].gate.lambda_for(base_sigma).item(), 0.4)
        self.assertEqual(trainer.fake_model.predict_velocity_with_features.call_count, 1)

    def test_calibration_and_validation_are_distinct_joint_queries_without_model_updates(self):
        trainer, runtime, student_parameter, fake_parameter, _, _ = self.fixture()
        self.set_corrections(runtime, 0.1, 0.2)
        before_heads = {name: copy.deepcopy(modality.module.state_dict()) for name, modality in runtime.modalities.items()}
        before_optimizers = {name: copy.deepcopy(modality.optimizer.state_dict()) for name, modality in runtime.modalities.items()}
        before_rng = torch.random.get_rng_state()
        metrics = runtime.check(iter([{"id": "calibration"}, {"id": "validation"}]), 3)
        torch.testing.assert_close(before_rng, torch.random.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(trainer.run_back_simulation.call_count, 2)
        self.assertEqual(trainer.fake_model.predict_velocity_with_features.call_count, 2)
        self.assertEqual([entry.args[0]["id"] for entry in trainer._encode_conditions.call_args_list], ["calibration", "validation"])
        initial = [entry.kwargs["xt"] for entry in trainer.run_back_simulation.call_args_list]
        self.assertFalse(torch.equal(initial[0].video, initial[1].video))
        self.assertFalse(torch.equal(initial[0].audio, initial[1].audio))
        for name, modality in runtime.modalities.items():
            self.assertTreeEqual(modality.module.state_dict(), before_heads[name])
            self.assertTreeEqual(modality.optimizer.state_dict(), before_optimizers[name])
            self.assertEqual(modality.gate.counts.sum().item(), 1)
            self.assertIsNone(modality._student_query)
            self.assertEqual(modality._student_records, [])
        self.assertTrue(any("heldout" in name for name in metrics))
        self.assertIsNone(student_parameter.grad)
        self.assertIsNone(fake_parameter.grad)

    def test_validation_checks_frozen_small_scale_and_does_not_choose_its_own_optimum(self):
        trainer, runtime, student_parameter, _, _, features = self.fixture()
        with torch.no_grad():
            student_parameter.zero_()
        self.set_corrections(runtime, 10.0, 10.0)

        def predictable_velocity(latents, sigma, condition):
            physical = trainer.fake_model.residual_head_sigmas(sigma)
            # Calibration has optimal lambda=.1 for both modalities. Fresh
            # validation agrees for video but has the opposite audio residual.
            audio_target = -1.0 if condition["id"] == "validation" else 1.0
            velocity = MiniMaxH3JointLatents(
                (1.0 - latents.video) / physical["video"].reshape(1, 1, 1),
                (audio_target - latents.audio) / physical["audio"].reshape(1, 1, 1),
                latents.shape,
            )
            return velocity, features(latents)

        trainer.fake_model.predict_velocity_with_features.side_effect = predictable_velocity
        runtime.check(iter([{"id": "calibration"}, {"id": "validation"}]), 0)
        physical = trainer.fake_model.residual_head_sigmas(torch.tensor([0.25]))
        self.assertAlmostEqual(runtime.modalities["video"].gate.lambda_for(physical["video"]).item(), 0.1, places=6)
        self.assertEqual(runtime.modalities["audio"].gate.lambda_for(physical["audio"]).item(), 0.0)
        self.assertEqual(trainer.fake_model.predict_velocity_with_features.call_count, 2)

    def test_student_forward_uses_joint_head_and_keeps_gradients_on_student_only(self):
        trainer, runtime, student_parameter, fake_parameter, joint, _ = self.fixture()
        self.set_corrections(runtime, 0.4, 0.6)
        for modality in runtime.modalities.values():
            modality.gate.lambdas.fill_(0.5)
        teacher_parameter = torch.nn.Parameter(torch.tensor(-0.1))
        trainer.teacher = SimpleNamespace(set_training=Mock(), predict_guided_velocity=Mock(side_effect=lambda *args: joint(teacher_parameter)))
        result = DmdTrainer.forward_loss(trainer, joint(0).shape, ({"id": "student"}, None), "student")
        result["loss"].backward()
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertIsNotNone(student_parameter.grad)
        self.assertIsNone(fake_parameter.grad)
        self.assertIsNone(teacher_parameter.grad)
        self.assertEqual(trainer.fake_model.predict_velocity_with_features.call_count, 1)
        self.assertEqual(trainer.teacher.predict_guided_velocity.call_count, 1)
        for modality in runtime.modalities.values():
            self.assertTrue(all(parameter.grad is None for parameter in modality.module.parameters()))
            self.assertIsNone(modality._student_query)
            self.assertEqual(len(modality._student_records), 1)
            self.assertEqual(modality._student_records[0]["lambda_actual"], 0.5)

    def test_checkpoint_roundtrip_restores_both_heads_optimizers_gates_and_usage(self):
        _, runtime, _, _, joint, _ = self.fixture()
        runtime.collecting_fit = True
        runtime.remember_fit(joint(0), joint(1), torch.tensor([0.25]), {})
        runtime.collecting_fit = False
        runtime.fit()
        self.set_corrections(runtime, 0.4, 0.6)
        for name, modality in runtime.modalities.items():
            modality.gate.lambdas.fill_(0.25 if name == "video" else 0.5)
        runtime.predict_corrected_fake(joint(1), torch.tensor([0.25]), {})
        runtime.log_student_query(joint(0), joint(-1))
        runtime._finish_student_iteration(0)
        saved = copy.deepcopy(runtime.state_dict())
        _, restored, _, _, _, _ = self.fixture()
        self.assertEqual(runtime.checkpoint_metadata(), restored.checkpoint_metadata())
        restored.load_state_dict(saved)
        self.assertTreeEqual(restored.state_dict(), saved)
        for modality in restored.modalities.values():
            self.assertEqual(modality._student_updates, 1)
            self.assertTrue(modality.optimizer.state)
        # A single-modality or Wan state must not silently initialize audio.
        with self.assertRaises((RuntimeError, ValueError, KeyError)):
            restored.load_state_dict(runtime.modalities["video"].state_dict())

    def test_outer_iteration_fits_frozen_fake_then_checks_then_uses_fresh_student(self):
        trainer, runtime, student_parameter, fake_parameter, joint, _ = self.fixture(fit_steps=1, fit_grad_accum_steps=1)
        trainer.fake_update_ratio = 5
        stage_order, query_fake_snapshots = [], []
        original_predict = trainer.fake_model.predict_velocity_with_features.side_effect

        def predict(latents, sigma, condition):
            query_fake_snapshots.append((condition["id"], fake_parameter.item()))
            return original_predict(latents, sigma, condition)

        trainer.fake_model.predict_velocity_with_features.side_effect = predict
        trainer.teacher = SimpleNamespace(set_training=Mock(), predict_guided_velocity=Mock(side_effect=lambda *args: joint(-0.1)))

        def stage(samples, *, stage, grad_accum_iters, outer_iteration, **kwargs):
            self.assertEqual(grad_accum_iters, 1)
            self.assertEqual(outer_iteration, 0)
            sample = next(samples)
            condition = trainer._encode_conditions(sample)[0]
            stage_order.append((stage, sample["id"]))
            if stage == "fake":
                runtime.remember_fit(joint(student_parameter), joint(1.0), torch.tensor([0.25]), condition)
                with torch.no_grad():
                    fake_parameter.add_(1.0)
                return {"loss": 2.0, "fake_real": 0.0}
            result = DmdTrainer.forward_loss(trainer, joint(0).shape, (condition, None), "student")
            result["loss"].backward()
            return result

        trainer._train_one_stage = stage
        result, fake_loss, fake_real_loss = runtime.train_iteration(iter({"id": index} for index in range(8)), 1, 0)
        self.assertEqual(stage_order, [("fake", index) for index in range(5)] + [("student", 7)])
        self.assertEqual(query_fake_snapshots, [(0, 5.0), (5, 5.0), (6, 5.0), (7, 5.0)])
        self.assertEqual(trainer.fake_model.predict_velocity_with_features.call_count, 4)
        self.assertEqual((fake_loss, fake_real_loss), (2.0, 0.0))
        self.assertIn("head_video_fit_loss", result)
        self.assertIn("head_audio_fit_loss", result)
        self.assertIsNotNone(student_parameter.grad)
        self.assertIsNone(fake_parameter.grad)
        for modality in runtime.modalities.values():
            self.assertEqual(modality._student_updates, 1)
            self.assertIsNone(modality._student_query)
            self.assertEqual(modality._student_records, [])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU gloo backend is unavailable")
    def test_two_rank_joint_heads_and_gates_stay_synchronized(self):
        with tempfile.TemporaryDirectory(prefix="h3_residual_head_gloo_") as temporary_directory:
            context = mp.spawn(_joint_head_worker, args=(str(Path(temporary_directory) / "init"),), nprocs=2, join=False)
            deadline = time.monotonic() + 45
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.fail("Two-rank H3 residual-head test exceeded its 45-second timeout")
                    if context.join(timeout=remaining):
                        break
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=1)


def _joint_head_worker(rank, init_file):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=25))
    try:
        with patch("lightx2v_train.trainers.dmd.residual_head_training.logger"):
            helper = H3ResidualHeadTrainingTest()
            trainer, runtime, _, _, joint, _ = helper.fixture()
            runtime.collecting_fit = True
            for index in range(2):
                runtime.remember_fit(joint(0), joint(1 + index + rank), torch.tensor([0.1 + rank * 0.4]), {"id": index})
            runtime.collecting_fit = False
            runtime.fit()
            trainer._sample_score_sigma.return_value = torch.tensor([0.1 + rank * 0.4])
            runtime.check(iter([{"id": "calibration"}, {"id": "validation"}]), 0)
            runtime.predict_corrected_fake(joint(1 + rank), trainer._sample_score_sigma.return_value, {})
            runtime.log_student_query(joint(0), joint(-1))
            runtime._finish_student_iteration(0)
            for modality in runtime.modalities.values():
                assert isinstance(modality.head, DistributedDataParallel)
                values = [torch.cat([parameter.detach().flatten() for parameter in modality.module.parameters()])]
                values.extend(modality.gate.state_dict().values())
                values.append(modality._usage_counts)
                for value in values:
                    copies = [torch.empty_like(value) for _ in range(2)]
                    dist.all_gather(copies, value.contiguous())
                    torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
                assert modality._usage_counts[0].sum().item() == 2
                assert modality._head_optimizer_updates == 1
            assert trainer.fake_model.predict_velocity_with_features.call_count == 5
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
