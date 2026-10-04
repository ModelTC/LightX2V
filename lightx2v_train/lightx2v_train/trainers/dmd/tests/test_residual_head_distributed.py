"""Real two-rank CPU coverage for the replicated head and held-out gate."""

import copy
import io
import json
import os
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger
from torch.nn.parallel import DistributedDataParallel

from lightx2v_train.trainers.dmd.residual_head import NoiseBinGate, ResidualHeadConfig, TokenResidualHead
from lightx2v_train.trainers.dmd.residual_head_training import ResidualHeadTraining


def _assert_rank_tensors_equal(tensor, world_size):
    copies = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(copies, tensor)
    for copy in copies[1:]:
        torch.testing.assert_close(copy, copies[0], rtol=0, atol=0)


def _assert_state_equal(actual, expected):
    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _parameters_flat(module):
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


def _residual_head_distributed_worker(rank, world_size, init_file):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        # Deliberately different CPU initialization: DDP, not a shared seed,
        # must make all replicated head parameters identical before fitting.
        torch.random.default_generator.manual_seed(100 + rank)
        module = TokenResidualHead(4, 1, (1, 1, 1), hidden_dim=3)
        initial = [torch.empty_like(module.feature_projection.weight) for _ in range(world_size)]
        dist.all_gather(initial, module.feature_projection.weight.detach())
        assert not torch.equal(initial[0], initial[1])
        head = DistributedDataParallel(module, broadcast_buffers=False)
        _assert_rank_tensors_equal(_parameters_flat(module), world_size)

        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2, weight_decay=0)
        generator = torch.Generator(device="cpu").manual_seed(1000 + rank)
        features = torch.randn(2, 4, 4, generator=generator, requires_grad=True)
        sigma = torch.full((2,), 0.2 + 0.6 * rank, requires_grad=True)
        target = torch.full((2, 1, 1, 2, 2), float(rank + 1))
        before_fit = _parameters_flat(module).clone()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            residual = head(features, sigma, tuple(target.shape))
            (residual - target).square().mean().backward()
            assert features.grad is None and sigma.grad is None
            gradients = torch.cat([parameter.grad.reshape(-1) for parameter in module.parameters()])
            _assert_rank_tensors_equal(gradients, world_size)
            optimizer.step()
            _assert_rank_tensors_equal(_parameters_flat(module), world_size)
        assert not torch.equal(before_fit, _parameters_flat(module))

        config = ResidualHeadConfig(enabled=True, noise_bins=4, min_checks=2, ema_decay=0.5)
        gate = NoiseBinGate(config, "cpu")
        # Rank-local checks touch different bins; bin 1 also requires pooling
        # an improvement on rank 0 and a regression on rank 1.
        if rank == 0:
            checks = [(torch.tensor([0.1, 0.3]), torch.tensor([4.0, 4.0]), torch.tensor([2.0, 1.0]))]
        else:
            checks = [(torch.tensor([0.3, 0.9]), torch.tensor([2.0, 2.0]), torch.tensor([3.0, 4.0]))]
        gate.update(checks)
        torch.testing.assert_close(gate.lambdas, torch.zeros(4), rtol=0, atol=0)
        gate.update(checks)
        torch.testing.assert_close(gate.lambdas, torch.tensor([0.25, 0.25, 0.0, 0.0]), rtol=0, atol=0)
        torch.testing.assert_close(gate.counts, torch.tensor([2, 2, 0, 2]), rtol=0, atol=0)
        torch.testing.assert_close(gate.ema_fake_mse, torch.tensor([4.0, 3.0, 0.0, 2.0]), rtol=0, atol=0)
        torch.testing.assert_close(gate.ema_delta, torch.tensor([2.0, 1.0, 0.0, -2.0]), rtol=0, atol=0)
        for value in gate.state_dict().values():
            _assert_rank_tensors_equal(value, world_size)

        # Independent calibration/validation streams pool different rank-local
        # bins. Small lambda .1 helps where lambda 1 would regress; another
        # calibration-selected scale .6 is rejected, not re-fit to the
        # validation optimum .1. Both streams count rounds, not rank samples.
        calibrated_config = ResidualHeadConfig(enabled=True, gate_mode="calibrated", noise_bins=4, min_checks=2, ema_decay=0.5)
        calibrated_gate = NoiseBinGate(calibrated_config, "cpu")
        if rank == 0:
            calibration = [(torch.tensor([0.1, 0.3]), torch.tensor([0.1, 0.8]), torch.ones(2))]
            validation = [(torch.tensor([0.1, 0.3]), torch.tensor([0.04, 1.0]), torch.tensor([0.1, 0.1]), torch.ones(2))]
        else:
            calibration = [(torch.tensor([0.3, 0.9]), torch.tensor([0.4, 0.2]), torch.ones(2))]
            validation = [(torch.tensor([0.3, 0.9]), torch.tensor([1.0, 0.1]), torch.tensor([0.1, -0.1]), torch.ones(2))]
        for round_index in range(2):
            snapshot = calibrated_gate.calibrate(calibration)
            _assert_rank_tensors_equal(snapshot, world_size)
            calibrated_gate.update_calibrated(validation, snapshot)
            if round_index == 0:
                torch.testing.assert_close(calibrated_gate.lambdas, torch.zeros(4), rtol=0, atol=0)
        torch.testing.assert_close(snapshot, torch.tensor([0.1, 0.6, 0.0, 0.2]))
        torch.testing.assert_close(calibrated_gate.lambdas, torch.tensor([0.1, 0.0, 0.0, 0.0]))
        for counters in (calibrated_gate.counts, calibrated_gate.calibration_counts):
            torch.testing.assert_close(counters, torch.tensor([2, 2, 0, 2]), rtol=0, atol=0)
        for value in calibrated_gate.state_dict().values():
            _assert_rank_tensors_equal(value, world_size)
        calibrated_restored = NoiseBinGate(calibrated_config, "cpu")
        calibrated_restored.load_state_dict(calibrated_gate.state_dict())
        _assert_state_equal(calibrated_restored.state_dict(), calibrated_gate.state_dict())

        # Exercise actual torch serialization, including Adam's warmed-up
        # moments and gate EMA/counts/lambdas, with no GPU checkpoint storage.
        buffer = io.BytesIO()
        torch.save(
            {
                "head": module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "gate": gate.state_dict(),
                "config": config.checkpoint_metadata(),
            },
            buffer,
        )
        buffer.seek(0)
        state = torch.load(buffer, map_location="cpu", weights_only=False)
        restored = TokenResidualHead(4, 1, (1, 1, 1), hidden_dim=3)
        restored.load_state_dict(state["head"])
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-2, weight_decay=0)
        restored_optimizer.load_state_dict(state["optimizer"])
        restored_gate = NoiseBinGate(ResidualHeadConfig.from_mapping(state["config"]), "cpu")
        restored_gate.load_state_dict(state["gate"])
        _assert_state_equal(restored.state_dict(), module.state_dict())
        _assert_state_equal(restored_optimizer.state_dict(), optimizer.state_dict())
        _assert_state_equal(restored_gate.state_dict(), gate.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(restored(features, sigma, tuple(target.shape)), module(features, sigma, tuple(target.shape)), rtol=0, atol=0)
        _assert_rank_tensors_equal(_parameters_flat(restored), world_size)

        # Real runtime collectives pool rank-local use (only rank 0's bin is
        # active) and gather both ranks despite rank-zero-only log emission.
        # Use the already trained head; no teacher/model forward is needed.
        runtime = ResidualHeadTraining.__new__(ResidualHeadTraining)
        runtime.device = torch.device("cpu")
        runtime.config = config
        runtime.module, runtime.optimizer, runtime.gate = module, optimizer, gate
        runtime._student_query = None
        runtime._student_records = []
        runtime._student_query_index = 0
        runtime._usage_counts = torch.zeros(3, config.noise_bins, dtype=torch.int64)
        runtime._student_updates = runtime._positive_lambda_updates = runtime._nonzero_correction_updates = 0
        runtime._usage_history_complete = True
        runtime._head_optimizer_updates = runtime._head_fit_microbatches = 0
        runtime._fit_history_complete = True
        runtime._last_fit_metrics = {}
        runtime._fake_x0_features = lambda latent, noise_sigma, condition: (latent.detach(), features[:1].detach())
        local_sigma = torch.tensor([0.1 if rank == 0 else 0.9])
        latent = torch.ones(1, 1, 1, 2, 2)
        gate_before = {key: value.clone() for key, value in gate.state_dict().items()}
        rng_before = torch.random.get_rng_state()
        corrected = runtime.predict_corrected_fake(latent, local_sigma, {})
        runtime.log_student_query(torch.zeros_like(latent), torch.zeros_like(latent))
        torch.testing.assert_close(rng_before, torch.random.get_rng_state(), rtol=0, atol=0)
        assert not corrected.requires_grad and runtime._student_query is None
        emitted = []
        sink = logger.add(lambda message: emitted.append(str(message)), format="{message}", filter=lambda record: record["message"].startswith("[head][student]"))
        try:
            usage_metrics = runtime._finish_student_iteration(0)
        finally:
            logger.remove(sink)
        expected_counts = torch.tensor([[1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.int64)
        torch.testing.assert_close(runtime._usage_counts, expected_counts, rtol=0, atol=0)
        _assert_rank_tensors_equal(runtime._usage_counts, world_size)
        assert usage_metrics["head_student_positive_lambda_fraction"] == 0.5
        assert runtime._positive_lambda_updates == 1 and runtime._nonzero_correction_updates == 1
        assert runtime._student_records == []
        _assert_state_equal(gate.state_dict(), gate_before)
        if rank == 0:
            assert len(emitted) == 1
            report = json.loads(emitted[0].removeprefix("[head][student] "))
            assert [record["rank"] for record in report["samples"]] == [0, 1]
            assert [record["lambda_actual"] for record in report["samples"]] == [0.25, 0.0]
            assert report["cumulative_positive_lambda_sample_count"] == 1
            assert report["cumulative_any_rank_positive_lambda_updates"] == 1
            assert report["risk_by_bin"][0]["full_correction_delta"]["stderr"] is None
            assert report["risk_by_bin"][0]["full_correction_delta"]["uncertainty_defined"] is False
        else:
            assert emitted == []

        usage_buffer = io.BytesIO()
        torch.save(runtime.state_dict(), usage_buffer)
        usage_buffer.seek(0)
        usage_state = torch.load(usage_buffer, map_location="cpu", weights_only=False)
        resumed = ResidualHeadTraining.__new__(ResidualHeadTraining)
        resumed.device, resumed.config = torch.device("cpu"), config
        resumed.module, resumed.optimizer, resumed.gate = restored, restored_optimizer, restored_gate
        resumed._usage_counts = torch.zeros_like(runtime._usage_counts)
        resumed._student_query, resumed._student_records = None, []
        resumed.load_state_dict(usage_state)
        _assert_state_equal(resumed.state_dict(), runtime.state_dict())
        _assert_rank_tensors_equal(resumed._usage_counts, world_size)

        # Exercise the actual fit() no_sync path, not a stand-in wrapper.
        # Compare two accumulated global-batch updates (4+1 unique queries)
        # against an explicitly all-reduced reference optimizer on each rank.
        accumulation_config = ResidualHeadConfig(enabled=True, fit_steps=5, fit_grad_accum_steps=4, max_grad_norm=1000)
        reference_module = copy.deepcopy(module)
        reference_optimizer = torch.optim.SGD(reference_module.parameters(), lr=0.01)
        fit_runtime = ResidualHeadTraining.__new__(ResidualHeadTraining)
        fit_runtime.device, fit_runtime.config = torch.device("cpu"), accumulation_config
        fit_runtime.module, fit_runtime.head = module, head
        fit_runtime.optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        fit_runtime._head_optimizer_updates = fit_runtime._head_fit_microbatches = 0
        fit_runtime._fit_history_complete = True
        fit_runtime._last_fit_metrics = {}
        fit_runtime._fake_x0_features = lambda latent, noise_sigma, condition: (torch.zeros_like(latent), condition["features"])
        fit_runtime.fit_batches = []
        microbatches = []
        for index in range(5):
            fit_features = torch.randn(2, 4, 4, generator=generator)
            fit_sigma = torch.tensor([0.1 + index * 0.1, 0.7 + rank * 0.1])
            fit_target = torch.full((2, 1, 1, 2, 2), float(index + rank + 1))
            microbatches.append((fit_features, fit_sigma, fit_target))
            fit_runtime.fit_batches.append(
                SimpleNamespace(
                    generated=-fit_target,
                    renoised=torch.zeros_like(fit_target),
                    sigma=fit_sigma,
                    condition={"features": fit_features},
                )
            )
        for group in (microbatches[:4], microbatches[4:]):
            reference_optimizer.zero_grad(set_to_none=True)
            for fit_features, fit_sigma, fit_target in group:
                prediction = reference_module(fit_features, fit_sigma, tuple(fit_target.shape))
                ((prediction - fit_target).square().mean() / len(group)).backward()
            for parameter in reference_module.parameters():
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(world_size)
            torch.nn.utils.clip_grad_norm_(reference_module.parameters(), accumulation_config.max_grad_norm)
            reference_optimizer.step()
        with mock.patch.object(head, "no_sync", wraps=head.no_sync) as no_sync:
            fit_runtime.fit()
        assert no_sync.call_count == 3
        assert fit_runtime._head_optimizer_updates == 2 and fit_runtime._head_fit_microbatches == 5
        torch.testing.assert_close(_parameters_flat(module), _parameters_flat(reference_module), rtol=1e-5, atol=1e-6)
        _assert_rank_tensors_equal(_parameters_flat(module), world_size)
        assert not fit_runtime.fit_batches
        assert all(parameter.grad is None for parameter in module.parameters())
    finally:
        dist.destroy_process_group()


class ResidualHeadDistributedTest(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU gloo distributed backend is unavailable")
    def test_two_rank_ddp_gate_and_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory(prefix="residual_head_gloo_") as temporary_directory:
            init_file = str(Path(temporary_directory) / "init")
            context = mp.spawn(_residual_head_distributed_worker, args=(2, init_file), nprocs=2, join=False)
            deadline = time.monotonic() + 30
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.fail("Two-rank residual-head CPU test exceeded its 30-second timeout.")
                    if context.join(timeout=remaining):
                        break
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
