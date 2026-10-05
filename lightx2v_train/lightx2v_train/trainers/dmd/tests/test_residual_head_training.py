import copy
import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from lightx2v_train.trainers.dmd.checkpoint import DmdCheckpointManager
from lightx2v_train.trainers.dmd.math import dmd_loss, expand_sigma
from lightx2v_train.trainers.dmd.residual_head import ResidualHeadConfig
from lightx2v_train.trainers.dmd.residual_head_training import ResidualHeadTraining, _risk_summary_by_bin
from lightx2v_train.trainers.dmd.trainer import DmdTrainer


class ResidualHeadTrainingTest(unittest.TestCase):
    def fixture(self, **config_updates):
        shape = (1, 2, 1, 2, 2)
        student_parameter = torch.nn.Parameter(torch.tensor(0.2))
        fake_parameter = torch.nn.Parameter(torch.tensor(0.0))
        student = SimpleNamespace(
            device=torch.device("cpu"),
            projected_dmd=False,
            x0_from_velocity=lambda latent, velocity, sigma: latent - expand_sigma(sigma, latent.ndim) * velocity,
            latent_hw=lambda dimensions: dimensions[-2:],
            random_noise_like=lambda generated, dtype, broadcast: broadcast(torch.randn_like(generated).to(dtype)),
            add_noise=lambda scheduler, generated, noise, sigma: (1 - expand_sigma(sigma, generated.ndim)) * generated + expand_sigma(sigma, generated.ndim) * noise,
        )
        fake = SimpleNamespace(set_training=Mock())

        def predict(latent, sigma, condition):
            del sigma, condition
            tokens = latent.flatten(2).transpose(1, 2)
            features = torch.cat((tokens, tokens), dim=-1)
            return fake_parameter * torch.ones_like(latent), features.detach()

        fake_model = SimpleNamespace(
            transformer=SimpleNamespace(dim=4),
            _latent_channels=lambda: 2,
            patch_size=(1, 1, 1),
            predict_velocity_with_features=Mock(side_effect=predict),
        )
        trainer = SimpleNamespace(
            trainer_name="dmd",
            student=student,
            fake=fake,
            fake_model=fake_model,
            config={"seed": 42},
            scheduler=object(),
            latent_dtype=torch.float32,
            _encode_conditions=Mock(side_effect=lambda sample: ({"prompt": sample["id"]}, None)),
            _latent_shape=Mock(return_value=shape),
            sample_initial_latents=Mock(side_effect=lambda dimensions: torch.randn(dimensions)),
            _sample_score_sigma=Mock(return_value=torch.tensor([0.25])),
            run_back_simulation=Mock(side_effect=lambda condition, dimensions, grad_enabled, xt, student_query=False: (student_parameter.expand(dimensions).detach(), 1000, 0)),
        )
        config = ResidualHeadConfig(
            **{
                "enabled": True,
                "hidden_dim": 4,
                "fit_steps": 5,
                "noise_bins": 1,
                "learning_rate": 0.02,
                "min_checks": 3,
                **config_updates,
            }
        )
        runtime = ResidualHeadTraining(trainer, config)
        return trainer, runtime, student_parameter, fake_parameter

    def test_head_initialization_preserves_rng_and_zero_correction(self):
        torch.manual_seed(123)
        before = torch.random.get_rng_state()
        _, runtime, _, _ = self.fixture()
        torch.testing.assert_close(before, torch.random.get_rng_state())
        features = torch.randn(1, 4, 4)
        correction = runtime.module(features, torch.tensor([0.25]), (1, 2, 1, 2, 2))
        torch.testing.assert_close(correction, torch.zeros_like(correction), rtol=0, atol=0)

    def test_fit_recomputes_one_fixed_snapshot_per_batch_without_model_grads(self):
        trainer, runtime, student_parameter, fake_parameter = self.fixture()
        generated = student_parameter.expand(1, 2, 1, 2, 2)
        renoised = torch.ones_like(generated)
        runtime.collecting_fit = True
        for index in range(5):
            runtime.remember_fit(generated, renoised, torch.tensor([0.25]), {"index": index})
        runtime.collecting_fit = False
        # Cache collected under the old fake is scored using the final fake.
        fake_parameter.data.fill_(2.0)
        loss = runtime.fit()
        self.assertLessEqual(loss, 0.3**2 + 1e-6)
        self.assertEqual(trainer.fake_model.predict_velocity_with_features.call_count, 5)
        self.assertEqual(runtime.fit_batches, [])
        self.assertIsNone(student_parameter.grad)
        self.assertIsNone(fake_parameter.grad)
        self.assertTrue(runtime.optimizer.state)

    def test_heldout_new_rollout_uses_isolated_rng_and_only_gates_after_three_rounds(self):
        trainer, runtime, _, _ = self.fixture()
        with torch.no_grad():
            runtime.module.output_projection.bias.fill_(0.1)
        torch.manual_seed(24)
        before = torch.random.get_rng_state()
        samples = iter([{"id": "heldout-a"}, {"id": "heldout-b"}, {"id": "heldout-c"}])
        for index in range(3):
            metrics = runtime.check(samples, index)
            self.assertIn("head_heldout_fake_mse", metrics)
            self.assertEqual(runtime.gate.counts.item(), index + 1)
            if index < 2:
                self.assertEqual(runtime.gate.lambdas.item(), 0)
            torch.testing.assert_close(before, torch.random.get_rng_state())
        self.assertEqual(trainer.run_back_simulation.call_count, 3)
        noises = [entry.kwargs["xt"] for entry in trainer.run_back_simulation.call_args_list]
        self.assertFalse(torch.equal(noises[0], noises[1]))
        self.assertEqual([entry.args[0]["id"] for entry in trainer._encode_conditions.call_args_list], ["heldout-a", "heldout-b", "heldout-c"])
        self.assertEqual(runtime.fit_batches, [])

    def test_fit_accumulates_distinct_batches_and_normalizes_partial_group(self):
        trainer, runtime, student_parameter, fake_parameter = self.fixture(fit_grad_accum_steps=4, max_grad_norm=1000)

        class ScalarHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.1))

            def forward(self, features, sigma, shape):
                return self.value.expand(shape)

        module = ScalarHead()
        sync_states = []

        class SyncWrapper(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
                self.sync = True
                self.no_sync_calls = 0

            @contextmanager
            def no_sync(self):
                self.no_sync_calls += 1
                self.sync = False
                try:
                    yield
                finally:
                    self.sync = True

            def forward(self, *args):
                sync_states.append(self.sync)
                return self.wrapped(*args)

        runtime.module = module
        runtime.head = SyncWrapper(module)
        runtime.optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        reference = torch.nn.Parameter(torch.tensor(0.1))
        reference_optimizer = torch.optim.SGD([reference], lr=0.1)
        targets = []
        runtime.collecting_fit = True
        for value, batch_size in zip(range(1, 6), (1, 3, 2, 1, 2)):
            generated = student_parameter.expand(batch_size, 2, 1, 2, 2)
            renoised = torch.full_like(generated, value + student_parameter.item())
            targets.append(renoised.detach() - generated.detach())
            runtime.remember_fit(generated, renoised, torch.tensor([0.25]), {})
        runtime.collecting_fit = False
        for group in (targets[:4], targets[4:]):
            reference_optimizer.zero_grad()
            target = torch.cat(group)
            (reference.expand_as(target) - target).square().mean().backward()
            reference_optimizer.step()
        with patch("lightx2v_train.trainers.dmd.residual_head_training.logger") as logged:
            runtime.fit()
        torch.testing.assert_close(module.value, reference)
        self.assertEqual(sync_states, [False, False, False, True, True])
        self.assertEqual(runtime.head.no_sync_calls, 3)
        self.assertEqual(trainer.fake_model.predict_velocity_with_features.call_count, 5)
        self.assertEqual(runtime._head_optimizer_updates, 2)
        self.assertEqual(runtime._head_fit_microbatches, 5)
        self.assertIsNone(student_parameter.grad)
        self.assertIsNone(fake_parameter.grad)
        self.assertIsNone(module.value.grad)
        report = json.loads(logged.info.call_args.args[1])
        self.assertEqual(report["group_microbatches"], [4, 1])
        self.assertEqual(report["effective_batch_samples_global"], [7, 2])
        self.assertEqual(report["unique_fit_samples_global"], 9)
        self.assertEqual(report["replayed_microbatch_passes_per_rank"], 0)
        self.assertEqual(report["cross_outer_iteration_lag"], 0)

    def test_fit_does_not_present_cached_replay_as_independent_accumulation(self):
        for count, groups in ((1, [1] * 5), (2, [2, 2, 1]), (4, [4, 1])):
            with self.subTest(cached_queries=count):
                trainer, runtime, _, _ = self.fixture(fit_grad_accum_steps=4)
                runtime.collecting_fit = True
                for index in range(count):
                    runtime.remember_fit(torch.zeros(1, 2, 1, 2, 2), torch.ones(1, 2, 1, 2, 2), torch.tensor([0.25]), {"index": index})
                runtime.collecting_fit = False
                with patch("lightx2v_train.trainers.dmd.residual_head_training.logger") as logged:
                    runtime.fit()
                report = json.loads(logged.info.call_args.args[1])
                self.assertEqual(report["group_microbatches"], groups)
                self.assertEqual(report["unique_fit_samples_global"], count)
                self.assertEqual(report["replayed_microbatch_passes_per_rank"], 5 - count)
                self.assertEqual(trainer.fake_model.predict_velocity_with_features.call_count, count)
                self.assertEqual(runtime._head_fit_microbatches, 5)
                self.assertEqual(runtime._head_optimizer_updates, len(groups))

    def test_many_fake_microbatches_only_cache_and_rescore_first_fit_steps(self):
        trainer, runtime, student_parameter, fake_parameter = self.fixture(fit_grad_accum_steps=4)
        _, reference, reference_student, reference_fake = self.fixture(fit_grad_accum_steps=4)
        runtime.collecting_fit = reference.collecting_fit = True
        for fake_update in range(5):
            for microbatch in range(16):
                index = fake_update * 16 + microbatch
                generated = student_parameter.expand(1, 2, 1, 2, 2)
                renoised = torch.full_like(generated, 1 + index / 100)
                runtime.remember_fit(generated, renoised, torch.tensor([0.25]), {"index": index})
                self.assertEqual(len(runtime.fit_batches), min(index + 1, 5))
                if index < 5:
                    reference.remember_fit(reference_student.expand_as(generated), renoised, torch.tensor([0.25]), {"index": index})
        self.assertEqual([batch.condition["index"] for batch in runtime.fit_batches], list(range(5)))
        runtime.collecting_fit = reference.collecting_fit = False
        # Both rescoring and fitting still use the final frozen fake snapshot.
        with torch.no_grad():
            fake_parameter.fill_(2.0)
            reference_fake.fill_(2.0)
        with patch("lightx2v_train.trainers.dmd.residual_head_training.logger") as logged:
            actual_loss = runtime.fit()
            report = json.loads(logged.info.call_args.args[1])
            expected_loss = reference.fit()
        self.assertEqual(actual_loss, expected_loss)
        for name, value in runtime.module.state_dict().items():
            torch.testing.assert_close(value, reference.module.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(trainer.fake_model.predict_velocity_with_features.call_count, 5)
        self.assertEqual([entry.args[2]["index"] for entry in trainer.fake_model.predict_velocity_with_features.call_args_list], list(range(5)))
        self.assertEqual(runtime.fit_batches, [])
        self.assertEqual(report["group_microbatches"], [4, 1])
        self.assertEqual(report["unique_cached_microbatches_per_rank"], 5)
        self.assertEqual(report["replayed_microbatch_passes_per_rank"], 0)
        self.assertEqual(runtime._head_optimizer_updates, 2)
        self.assertIsNone(student_parameter.grad)
        self.assertIsNone(fake_parameter.grad)

    def test_sixteen_unique_fit_batches_match_one_manual_mean_gradient_step(self):
        trainer, runtime, student_parameter, fake_parameter = self.fixture(fit_steps=16, fit_grad_accum_steps=16, max_grad_norm=1000)

        class ScalarHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.1))
                self.forward_count = 0

            def forward(self, features, sigma, shape):
                self.forward_count += 1
                return self.value.expand(shape)

        runtime.module = runtime.head = ScalarHead()
        runtime.optimizer = torch.optim.SGD(runtime.module.parameters(), lr=0.1)
        retained_queries = []
        runtime.collecting_fit = True
        for fake_update in range(5):
            for microbatch in range(16):
                index = fake_update * 16 + microbatch
                generated = student_parameter.expand(1, 2, 1, 2, 2)
                renoised = torch.full_like(generated, 1 + index / 16)
                sigma = torch.tensor([0.25])
                runtime.remember_fit(generated, renoised, sigma, {"index": index})
                self.assertEqual(len(runtime.fit_batches), min(index + 1, 16))
                if index < 16:
                    retained_queries.append((generated.detach(), renoised.detach(), sigma))
        runtime.collecting_fit = False
        self.assertEqual([batch.condition["index"] for batch in runtime.fit_batches], list(range(16)))

        # Rescore against the final fake snapshot, then average 16 distinct
        # batch losses by hand, independently of fit()'s accumulation loop.
        with torch.no_grad():
            fake_parameter.fill_(2.0)
        targets = [renoised - expand_sigma(sigma, renoised.ndim) * fake_parameter.detach() - generated for generated, renoised, sigma in retained_queries]
        reference = torch.nn.Parameter(torch.tensor(0.1))
        reference_optimizer = torch.optim.SGD([reference], lr=0.1)
        expected_loss = torch.stack([(reference.expand_as(target) - target).square().mean() for target in targets]).mean()
        expected_loss.backward()
        expected_gradient = reference.grad.detach().clone()
        reference_optimizer.step()

        observed_gradients = []
        original_step = runtime.optimizer.step

        def step():
            observed_gradients.append(runtime.module.value.grad.detach().clone())
            return original_step()

        with patch.object(runtime.optimizer, "step", side_effect=step) as optimizer_step, patch("lightx2v_train.trainers.dmd.residual_head_training.logger") as logged:
            actual_loss = runtime.fit()
        optimizer_step.assert_called_once()
        torch.testing.assert_close(observed_gradients[0], expected_gradient)
        torch.testing.assert_close(runtime.module.value, reference)
        self.assertAlmostEqual(actual_loss, expected_loss.item(), places=6)
        self.assertEqual(runtime.module.forward_count, 16)
        self.assertEqual(trainer.fake_model.predict_velocity_with_features.call_count, 16)
        self.assertEqual([entry.args[2]["index"] for entry in trainer.fake_model.predict_velocity_with_features.call_args_list], list(range(16)))
        self.assertEqual(runtime.fit_batches, [])
        self.assertEqual(runtime._head_fit_microbatches, 16)
        self.assertEqual(runtime._head_optimizer_updates, 1)
        report = json.loads(logged.info.call_args.args[1])
        self.assertEqual(report["group_microbatches"], [16])
        self.assertEqual(report["effective_batch_samples_global"], [16])
        self.assertEqual(report["unique_fit_samples_global"], 16)
        self.assertEqual(report["unique_cached_microbatches_per_rank"], 16)
        self.assertEqual(report["replayed_microbatch_passes_per_rank"], 0)
        self.assertEqual(report["optimizer_updates"], 1)
        self.assertIsNone(student_parameter.grad)
        self.assertIsNone(fake_parameter.grad)
        self.assertIsNone(runtime.module.value.grad)

    def test_calibrated_check_uses_two_new_rng_streams_and_freezes_models(self):
        trainer, runtime, student_parameter, fake_parameter = self.fixture(gate_mode="calibrated", min_checks=1)
        with torch.no_grad():
            runtime.module.output_projection.bias.fill_(0.1)
        parameters_before = copy.deepcopy(runtime.module.state_dict())
        fake_before = fake_parameter.detach().clone()
        student_before = student_parameter.detach().clone()
        optimizer_before = copy.deepcopy(runtime.optimizer.state_dict())
        torch.manual_seed(29)
        before = torch.random.get_rng_state()
        with patch("lightx2v_train.trainers.dmd.residual_head_training.logger") as logged:
            metrics = runtime.check(iter([{"id": "calibration"}, {"id": "validation"}]), 3)
        torch.testing.assert_close(before, torch.random.get_rng_state())
        self.assertEqual(trainer.run_back_simulation.call_count, 2)
        initial = [entry.kwargs["xt"] for entry in trainer.run_back_simulation.call_args_list]
        self.assertFalse(torch.equal(initial[0], initial[1]))
        self.assertEqual([entry.args[0]["id"] for entry in trainer._encode_conditions.call_args_list], ["calibration", "validation"])
        self.assertEqual(trainer.fake_model.predict_velocity_with_features.call_count, 2)
        self.assertEqual(runtime.fit_batches, [])
        self.assertEqual(runtime._student_records, [])
        self.assertIsNone(runtime._student_query)
        self.assertIsNone(student_parameter.grad)
        self.assertIsNone(fake_parameter.grad)
        torch.testing.assert_close(student_parameter, student_before)
        torch.testing.assert_close(fake_parameter, fake_before)
        self.assertEqual(runtime.optimizer.state_dict(), optimizer_before)
        for name, value in parameters_before.items():
            torch.testing.assert_close(value, runtime.module.state_dict()[name])
        self.assertTrue(all(parameter.grad is None for parameter in runtime.module.parameters()))
        reports = [json.loads(entry.args[1]) for entry in logged.info.call_args_list]
        self.assertEqual([report["evaluation"] for report in reports], ["gate_calibration", "gate_validation"])
        self.assertEqual(reports[0]["selected_candidate_lambdas"], reports[1]["selected_candidate_lambdas"])
        self.assertIn("head_heldout_full_corrected_mse", metrics)

    def test_calibrated_validation_scores_exact_small_scale_not_full_correction(self):
        trainer, runtime, _, _ = self.fixture(gate_mode="calibrated", min_checks=1)
        fake_x0 = torch.ones(1, 2, 1, 2, 2)
        runtime._fake_x0_features = Mock(return_value=(fake_x0, torch.zeros(1, 4, 4)))
        trainer.run_back_simulation = Mock(return_value=(torch.zeros_like(fake_x0), 1000, 0))
        with torch.no_grad():
            runtime.module.output_projection.bias.fill_(10)
        with patch("lightx2v_train.trainers.dmd.residual_head_training.logger") as logged:
            metrics = runtime.check(iter([{"id": "calibration"}, {"id": "validation"}]), 0)
        validation = json.loads(logged.info.call_args_list[1].args[1])
        self.assertAlmostEqual(validation["samples"][0]["lambda_actual"], 0.1)
        self.assertAlmostEqual(metrics["head_heldout_corrected_mse"], 0.0)
        self.assertAlmostEqual(metrics["head_heldout_full_corrected_mse"], 81.0)
        self.assertAlmostEqual(metrics["head_heldout_delta"], 1.0)
        self.assertAlmostEqual(runtime.gate.lambdas.item(), 0.1)

    def test_validation_cannot_reselect_its_own_optimal_lambda(self):
        trainer, runtime, _, _ = self.fixture(gate_mode="calibrated", min_checks=1)
        latent = torch.ones(1, 2, 1, 2, 2)
        features = torch.zeros(1, 4, 4)
        runtime._fake_x0_features = Mock(side_effect=[(latent, features), (-latent, features)])
        trainer.run_back_simulation = Mock(return_value=(torch.zeros_like(latent), 1000, 0))
        with torch.no_grad():
            runtime.module.output_projection.bias.fill_(10)
        with patch("lightx2v_train.trainers.dmd.residual_head_training.logger") as logged:
            metrics = runtime.check(iter([{"id": "calibration"}, {"id": "validation"}]), 0)
        validation = json.loads(logged.info.call_args_list[1].args[1])
        self.assertAlmostEqual(validation["samples"][0]["lambda_actual"], 0.1)
        self.assertEqual(validation["samples"][0]["optimal_lambda"], 0.0)
        self.assertAlmostEqual(metrics["head_heldout_corrected_mse"], 4.0)
        self.assertAlmostEqual(metrics["head_heldout_delta"], -3.0)
        self.assertEqual(runtime.gate.lambdas.item(), 0)

    def test_student_correction_uses_bin_lambda_not_sample_residual(self):
        _, runtime, student_parameter, fake_parameter = self.fixture()
        with torch.no_grad():
            runtime.module.output_projection.bias.fill_(0.4)
            runtime.gate.lambdas.fill_(0.5)
        sigma = torch.tensor([0.25])
        latent = torch.ones(1, 2, 1, 2, 2)
        corrected = runtime.predict_corrected_fake(latent, sigma, {})
        torch.testing.assert_close(corrected, torch.full_like(latent, 0.8))
        self.assertFalse(corrected.requires_grad)
        generated = student_parameter.expand_as(latent)
        dmd_loss(generated, corrected, torch.zeros_like(corrected)).backward()
        self.assertIsNotNone(student_parameter.grad)
        self.assertIsNone(fake_parameter.grad)
        self.assertTrue(all(parameter.grad is None for parameter in runtime.module.parameters()))

    def test_student_logs_exact_cached_lambda_and_fresh_risk_without_more_queries(self):
        trainer, runtime, student_parameter, fake_parameter = self.fixture()
        with torch.no_grad():
            runtime.module.output_projection.bias.fill_(0.4)
            runtime.gate.lambdas.fill_(0.5)
        latent = torch.ones(1, 2, 1, 2, 2)
        generated = student_parameter.expand_as(latent)
        sigma = torch.tensor([0.25])
        before_rng = torch.random.get_rng_state()
        corrected = runtime.predict_corrected_fake(latent, sigma, {})
        # Changing a future gate state must not rewrite the query's choice.
        runtime.gate.lambdas.zero_()
        metrics = runtime.log_student_query(generated, torch.zeros_like(latent))
        torch.testing.assert_close(before_rng, torch.random.get_rng_state())
        self.assertIsNone(runtime._student_query)
        self.assertEqual(trainer.fake_model.predict_velocity_with_features.call_count, 1)
        record = runtime._student_records[0]
        self.assertEqual((record["rank"], record["sample_index"], record["query_index"], record["bin"]), (0, 0, 0, 0))
        self.assertEqual(record["sigma"], 0.25)
        self.assertEqual(record["lambda_actual"], 0.5)
        self.assertAlmostEqual(record["fake_mse"], 0.64, places=6)
        self.assertAlmostEqual(record["selected_lambda_mse"], 0.36, places=6)
        self.assertAlmostEqual(record["full_corrected_mse"], 0.16, places=6)
        self.assertAlmostEqual(record["residual_head_dot"], 0.32, places=6)
        self.assertAlmostEqual(record["head_energy"], 0.16, places=6)
        self.assertAlmostEqual(record["correction_direction_norm_ratio"], 0.2, places=6)
        self.assertAlmostEqual(metrics["head_student_correction_rms"], 0.4, places=6)
        self.assertAlmostEqual(metrics["head_student_applied_correction_rms"], 0.2, places=6)
        self.assertTrue(record["direction_cosine_valid"])
        dmd_loss(generated, corrected, torch.zeros_like(corrected)).backward()
        self.assertIsNotNone(student_parameter.grad)
        self.assertIsNone(fake_parameter.grad)
        self.assertTrue(all(parameter.grad is None for parameter in runtime.module.parameters()))
        with self.assertRaisesRegex(RuntimeError, "current corrected-fake query"):
            runtime.log_student_query(generated, torch.zeros_like(latent))

    def test_usage_counters_distinguish_positive_lambda_from_nonzero_correction_and_resume(self):
        _, runtime, _, _ = self.fixture()
        runtime.gate.lambdas.fill_(0.5)
        latent = torch.ones(1, 2, 1, 2, 2)
        runtime.predict_corrected_fake(latent, torch.tensor([0.25]), {})
        runtime.log_student_query(torch.zeros_like(latent), torch.zeros_like(latent))
        with patch("lightx2v_train.trainers.dmd.residual_head_training.logger") as logged:
            metrics = runtime._finish_student_iteration(0)
        report = json.loads(logged.info.call_args.args[1])
        self.assertEqual(report["positive_lambda_sample_count"], 1)
        self.assertEqual(report["nonzero_correction_sample_count"], 0)
        self.assertEqual(report["cumulative_any_rank_positive_lambda_updates"], 1)
        self.assertEqual(report["cumulative_any_rank_nonzero_correction_updates"], 0)
        self.assertEqual(report["risk_by_bin"][0]["applied_correction_delta"]["independent_query_count"], 1)
        self.assertIsNone(report["risk_by_bin"][0]["applied_correction_delta"]["stderr"])
        self.assertEqual(metrics["head_student_positive_lambda_samples_cumulative"], 1)
        self.assertEqual(runtime._student_records, [])
        _, restored, _, _ = self.fixture()
        restored.load_state_dict(copy.deepcopy(runtime.state_dict()))
        torch.testing.assert_close(runtime._usage_counts, restored._usage_counts)
        self.assertEqual(restored._student_updates, 1)
        self.assertEqual(restored._positive_lambda_updates, 1)
        self.assertEqual(restored._nonzero_correction_updates, 0)
        self.assertTrue(restored._usage_history_complete)
        legacy = copy.deepcopy(runtime.state_dict())
        del legacy["usage"]
        restored.load_state_dict(legacy)
        self.assertEqual(restored._usage_counts.sum().item(), 0)
        self.assertEqual(restored._positive_lambda_updates, 0)
        self.assertFalse(restored._usage_history_complete)
        self.assertFalse(restored.state_dict()["usage"]["history_complete"])

    def test_check_reports_pre_gate_policy_and_keeps_full_correction_gate_target(self):
        trainer, runtime, _, _ = self.fixture()
        fake_x0 = torch.ones(1, 2, 1, 2, 2)
        runtime._fake_x0_features = Mock(return_value=(fake_x0, torch.zeros(1, 4, 4)))
        trainer.run_back_simulation = Mock(return_value=(torch.zeros_like(fake_x0), 1000, 0))
        with torch.no_grad():
            runtime.module.output_projection.bias.fill_(0.4)
            runtime.gate.counts.fill_(3)
            runtime.gate.lambdas.fill_(0.5)
        with patch("lightx2v_train.trainers.dmd.residual_head_training.logger") as logged:
            metrics = runtime.check(iter([{"id": "heldout"}]), 7)
        report = json.loads(logged.info.call_args.args[1])
        record = report["samples"][0]
        self.assertEqual(report["evaluation"], "gate_train_check")
        self.assertEqual(report["applied_policy"], "lambda_snapshot_before_this_gate_update")
        self.assertEqual(record["lambda_actual"], 0.5)
        self.assertAlmostEqual(record["selected_lambda_mse"], 0.64, places=6)
        self.assertAlmostEqual(record["full_corrected_mse"], 0.36, places=6)
        self.assertEqual(report["lambda_after_gate_update"], [0.75])
        self.assertAlmostEqual(metrics["head_heldout_pre_gate_selected_lambda_mse"], 0.64, places=6)
        self.assertAlmostEqual(metrics["head_heldout_delta"], 0.64, places=6)
        self.assertEqual(runtime.gate.counts.item(), 4)
        self.assertEqual(runtime._student_records, [])

    def test_bin_uncertainty_counts_rank_queries_not_pixels_or_batch_items(self):
        _, runtime, _, _ = self.fixture()
        latent = torch.ones(2, 2, 1, 2, 2)
        runtime.predict_corrected_fake(latent, torch.tensor([0.25]), {})
        runtime.log_student_query(torch.zeros_like(latent), torch.zeros_like(latent))
        first = runtime._student_records
        summary = _risk_summary_by_bin(first, 1)[0]
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["full_correction_delta"]["independent_query_count"], 1)
        self.assertFalse(summary["full_correction_delta"]["uncertainty_defined"])
        other_rank = [{**record, "rank": 1, "full_corrected_mse": 0.5} for record in first]
        summary = _risk_summary_by_bin(first + other_rank, 1)[0]
        self.assertEqual(summary["sample_count"], 4)
        self.assertEqual(summary["full_correction_delta"]["independent_query_count"], 2)
        self.assertTrue(summary["full_correction_delta"]["uncertainty_defined"])
        self.assertAlmostEqual(summary["full_correction_delta"]["std"], 0.5 / 2**0.5)
        self.assertAlmostEqual(summary["full_correction_delta"]["stderr"], 0.25)

    def test_usage_counters_pool_all_ranks_once_per_iteration(self):
        _, runtime, _, _ = self.fixture()
        latent = torch.ones(1, 2, 1, 2, 2)
        runtime.predict_corrected_fake(latent, torch.tensor([0.25]), {})
        runtime.log_student_query(torch.zeros_like(latent), torch.zeros_like(latent))
        local = runtime._student_records[0]
        remote = {**local, "rank": 1, "lambda_actual": 0.25, "applied_correction_rms": 0.1}

        def gathered(output, records):
            output[0] = records
            output[1] = [remote]

        def reduced(counts, op):
            counts.add_(torch.tensor([[1], [1], [1]], dtype=torch.int64))

        prefix = "lightx2v_train.trainers.dmd.residual_head_training."
        with (
            patch(prefix + "dist.is_initialized", return_value=True),
            patch(prefix + "dist.get_world_size", return_value=2),
            patch(prefix + "dist.get_rank", return_value=0),
            patch(prefix + "dist.all_gather_object", side_effect=gathered) as gather,
            patch(prefix + "dist.all_reduce", side_effect=reduced) as reduce,
            patch(prefix + "logger") as logged,
        ):
            metrics = runtime._finish_student_iteration(0)
        self.assertEqual(reduce.call_count, 1)
        self.assertEqual(gather.call_count, 1)
        torch.testing.assert_close(runtime._usage_counts, torch.tensor([[2], [1], [1]], dtype=torch.int64))
        report = json.loads(logged.info.call_args.args[1])
        self.assertEqual([record["rank"] for record in report["samples"]], [0, 1])
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["positive_lambda_sample_count"], 1)
        self.assertEqual(report["cumulative_any_rank_positive_lambda_updates"], 1)
        self.assertEqual(metrics["head_student_positive_lambda_fraction"], 0.5)

    def test_pooled_optimal_lambda_uses_ratio_of_pooled_energies_only_for_diagnostics(self):
        _, runtime, _, _ = self.fixture()
        latent = torch.ones(1, 2, 1, 2, 2)
        runtime.predict_corrected_fake(latent, torch.tensor([0.25]), {})
        runtime.log_student_query(torch.zeros_like(latent), torch.zeros_like(latent))
        source = runtime._student_records[0]
        records = [
            {**source, "rank": 0, "fake_mse": 1.0, "residual_head_dot": 0.5, "head_energy": 1.0, "optimal_lambda": 0.5},
            {**source, "rank": 1, "fake_mse": 4.0, "residual_head_dot": 4.0, "head_energy": 4.0, "optimal_lambda": 1.0},
        ]
        summary = _risk_summary_by_bin(records, 1)[0]
        self.assertEqual(summary["optimal_lambda"], 0.75)
        self.assertAlmostEqual(summary["pooled_optimal_lambda"], 0.9)
        self.assertAlmostEqual(summary["pooled_optimal_lambda_mse"], 0.475)
        self.assertTrue(summary["pooled_optimal_lambda_valid"])
        self.assertEqual(runtime.gate.lambdas.item(), 0)

    def test_global_direction_scalar_mean_ignores_invalid_rank_placeholders(self):
        _, runtime, _, _ = self.fixture()
        latent = torch.ones(1, 2, 1, 2, 2)
        runtime.predict_corrected_fake(latent, torch.tensor([0.25]), {})
        runtime.log_student_query(torch.zeros_like(latent), torch.zeros_like(latent))
        record = runtime._student_records[0]
        remote = {**record, "rank": 1, "direction_cosine": 0.0, "direction_angle_degrees": 0.0, "direction_cosine_valid": False}
        with patch("lightx2v_train.trainers.dmd.residual_head_training._gather_records", return_value=[record, remote]), patch("lightx2v_train.trainers.dmd.residual_head_training.logger"):
            metrics = runtime._finish_student_iteration(0)
        self.assertEqual(metrics["head_student_direction_cosine"], 1)
        self.assertEqual(metrics["head_student_direction_valid_fraction"], 0.5)

    def test_head_checkpoint_roundtrip_is_strict_and_restores_gate_optimizer(self):
        _, runtime, _, _ = self.fixture()
        runtime.collecting_fit = True
        runtime.remember_fit(torch.zeros(1, 2, 1, 2, 2), torch.ones(1, 2, 1, 2, 2), torch.tensor([0.25]), {})
        runtime.collecting_fit = False
        runtime.fit()
        runtime.gate.update([(torch.tensor([0.25]), torch.tensor([1.0]), torch.tensor([0.5]))])
        saved = copy.deepcopy(runtime.state_dict())
        _, restored, _, _ = self.fixture()
        restored.load_state_dict(saved)
        for name, value in runtime.module.state_dict().items():
            torch.testing.assert_close(value, restored.module.state_dict()[name])
        torch.testing.assert_close(runtime.gate.counts, restored.gate.counts)
        self.assertEqual(len(runtime.optimizer.state), len(restored.optimizer.state))
        self.assertEqual(restored._head_optimizer_updates, 5)
        self.assertEqual(restored._head_fit_microbatches, 5)
        legacy = copy.deepcopy(saved)
        del legacy["fit_progress"]
        restored.load_state_dict(legacy)
        self.assertEqual(restored._head_optimizer_updates, 0)
        self.assertFalse(restored._fit_history_complete)
        invalid_progress = copy.deepcopy(saved)
        invalid_progress["fit_progress"]["optimizer_updates"] = True
        with self.assertRaisesRegex(RuntimeError, "fit progress counters"):
            restored.load_state_dict(invalid_progress)
        with self.assertRaisesRegex(RuntimeError, "optimizer"):
            restored.load_state_dict({"head": saved["head"], "gate": saved["gate"]})

    def test_head_checkpoint_metadata_cannot_silently_transition(self):
        trainer, runtime, _, _ = self.fixture()
        trainer.residual_head = runtime
        trainer.config["resume"] = {"allow_distribution_matching_transition": True}
        manager = DmdCheckpointManager(trainer)
        metadata = {"residual_head_enabled": True, "residual_head_config": runtime.checkpoint_metadata(), "residual_head_state": runtime.state_dict()}
        manager._validate_residual_head_state(metadata, "trainer_state.pt")
        with self.assertRaisesRegex(RuntimeError, "residual_head_enabled"):
            manager._validate_residual_head_state({}, "trainer_state.pt")
        changed = copy.deepcopy(metadata)
        changed["residual_head_config"]["hidden_dim"] += 1
        with self.assertRaisesRegex(RuntimeError, "residual_head_config"):
            manager._validate_residual_head_state(changed, "trainer_state.pt")
        trainer.residual_head = None
        with self.assertRaisesRegex(RuntimeError, "residual_head_enabled"):
            manager._validate_residual_head_state(metadata, "trainer_state.pt")
        manager._validate_residual_head_state({}, "legacy_baseline.pt")

    def test_enabled_iteration_orders_fake_fit_head_check_then_student(self):
        trainer, runtime, _, _ = self.fixture()
        trainer.fake_update_ratio = 5
        events = []

        def stage(samples, stage, **kwargs):
            sample = next(samples)
            events.append((stage, sample))
            return {"loss": 2.0, "fake_real": 0.0} if stage == "fake" else {"loss": 1.0, "dmd": 1.0}

        trainer._train_one_stage = stage
        runtime.fit = Mock(side_effect=lambda: events.append(("head_fit", None)) or 0.1)
        runtime.check = Mock(side_effect=lambda samples, iteration: events.append(("heldout", next(samples))) or {"head_heldout_delta": 0.2})
        result, fake_loss, _ = runtime.train_iteration(iter(range(7)), 1, 0)
        self.assertEqual([entry[0] for entry in events], ["fake"] * 5 + ["head_fit", "heldout", "student"])
        self.assertEqual(events[-2][1], 5)
        self.assertEqual(events[-1][1], 6)
        self.assertEqual(result["head_fit_loss"], 0.1)
        self.assertEqual(fake_loss, 2.0)

    def test_calibrated_accumulated_iteration_uses_fixed_fake_then_three_distinct_queries(self):
        trainer, runtime, student_parameter, fake_parameter = self.fixture(gate_mode="calibrated", fit_grad_accum_steps=4, min_checks=1)
        trainer.fake_update_ratio = 5
        events = []
        fake_snapshots = []
        predict = runtime._fake_x0_features

        def scored(*args):
            fake_snapshots.append(fake_parameter.item())
            return predict(*args)

        runtime._fake_x0_features = scored
        original_fit = runtime.fit
        runtime.fit = Mock(side_effect=lambda: events.append("fit") or original_fit())
        original_query = runtime._fresh_gate_query

        def query(samples, iteration, stream):
            events.append(stream)
            return original_query(samples, iteration, stream)

        runtime._fresh_gate_query = query
        student_samples = []

        def stage(samples, stage, **kwargs):
            sample = next(samples)
            events.append(stage)
            generated = student_parameter.expand(1, 2, 1, 2, 2)
            latent = torch.ones_like(generated)
            sigma = torch.tensor([0.25])
            if stage == "fake":
                runtime.remember_fit(generated, latent, sigma, {"id": sample["id"]})
                with torch.no_grad():
                    fake_parameter.add_(0.1)
                return {"loss": 2.0, "fake_real": 0.0}
            student_samples.append(sample["id"])
            corrected = runtime.predict_corrected_fake(latent, sigma, {})
            runtime.log_student_query(generated, torch.zeros_like(latent))
            dmd_loss(generated, corrected, torch.zeros_like(latent)).backward()
            return {"loss": 1.0, "dmd": 1.0}

        trainer._train_one_stage = stage
        samples = iter({"id": index} for index in range(8))
        with patch("lightx2v_train.trainers.dmd.residual_head_training.logger"):
            result, _, _ = runtime.train_iteration(samples, 1, 0)
        self.assertEqual(events, ["fake"] * 5 + ["fit", "calibration", "validation", "student"])
        self.assertEqual([entry.args[0]["id"] for entry in trainer._encode_conditions.call_args_list], [5, 6])
        self.assertEqual(student_samples, [7])
        self.assertEqual(fake_snapshots, [0.5] * 8)
        self.assertEqual(result["head_fit_optimizer_updates"], 2)
        self.assertEqual(result["head_fit_cross_outer_iteration_lag"], 0)
        self.assertIsNotNone(student_parameter.grad)
        self.assertIsNone(fake_parameter.grad)
        self.assertTrue(all(parameter.grad is None for parameter in runtime.module.parameters()))

    def test_disabled_iteration_keeps_original_student_then_fake_order(self):
        trainer = DmdTrainer.__new__(DmdTrainer)
        trainer.residual_head = None
        trainer.fake_update_ratio = 2
        trainer._train_one_stage = Mock(side_effect=[{"dmd": 1.0}, {"loss": 2.0, "fake_real": 0.0}, {"loss": 4.0, "fake_real": 0.0}])
        samples = iter(())
        result, fake, fake_real = trainer._train_iteration(samples, 2, 7)
        self.assertEqual(result, {"dmd": 1.0})
        self.assertEqual((fake, fake_real), (3.0, 0.0))
        self.assertEqual(
            trainer._train_one_stage.call_args_list,
            [
                call(samples, stage="student", grad_accum_iters=2, outer_iteration=7),
                call(samples, stage="fake", grad_accum_iters=2, outer_iteration=7, fake_update_index=0),
                call(samples, stage="fake", grad_accum_iters=2, outer_iteration=7, fake_update_index=1),
            ],
        )

    def test_explicit_fake_first_orders_baseline_like_head(self):
        trainer = DmdTrainer.__new__(DmdTrainer)
        trainer.residual_head = None
        trainer.dmd_update_order = "fake_first"
        trainer.fake_update_ratio = 2
        trainer._train_one_stage = Mock(side_effect=[{"loss": 2.0, "fake_real": 0.0}, {"loss": 4.0, "fake_real": 0.0}, {"dmd": 1.0}])
        result, fake, _ = trainer._train_iteration(iter(()), 1, 0)
        self.assertEqual(result, {"dmd": 1.0})
        self.assertEqual(fake, 3.0)
        self.assertEqual([entry.kwargs["stage"] for entry in trainer._train_one_stage.call_args_list], ["fake", "fake", "student"])

    def test_student_query_matches_student_exit_sampling_without_autograd(self):
        trainer = DmdTrainer.__new__(DmdTrainer)
        trainer._prepare_sampling_schedule = Mock()
        trainer._prepare_timestep_lookup = Mock()
        trainer.sample_end_step = Mock(return_value=2)
        trainer._sample_synced_int = Mock(return_value=0)
        trainer._denoised_timestep_window = Mock(return_value=(1000, 0))
        trainer.latent_dtype = torch.float32
        trainer.scheduler = SimpleNamespace(num_inference_steps=4, sigma_at=lambda index, **kwargs: torch.tensor([0.5]))
        trainer.student = SimpleNamespace(
            device=torch.device("cpu"),
            latent_hw=lambda shape: shape[-2:],
            set_training=Mock(),
            step=lambda scheduler, velocity, index, sample: (sample, velocity),
            to_dtype=lambda sample, dtype: sample.to(dtype),
        )
        grad_flags = []
        parameter = torch.nn.Parameter(torch.tensor(1.0))

        def velocity(capability, sample, sigma, condition):
            grad_flags.append(torch.is_grad_enabled())
            return parameter * sample

        trainer._predict_velocity = velocity
        initial = torch.ones(1, 2, 1, 2, 2)
        generated, _, _ = trainer.run_back_simulation({}, initial.shape, False, xt=initial, student_query=True)
        trainer.sample_end_step.assert_called_once()
        trainer._sample_synced_int.assert_not_called()
        self.assertEqual(grad_flags, [False, False, False])
        self.assertFalse(generated.requires_grad)
        grad_flags.clear()
        trainer.sample_end_step.reset_mock()
        trainer.run_back_simulation({}, initial.shape, False, xt=initial)
        trainer.sample_end_step.assert_not_called()
        trainer._sample_synced_int.assert_called_once_with(0, 4)
        self.assertEqual(grad_flags, [False])


if __name__ == "__main__":
    unittest.main()
