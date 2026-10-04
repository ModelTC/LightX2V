"""Student weight EMA checks on CPU and real uneven DTensor shards."""

import copy
import io
import os
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from lightx2v_train.trainers.dmd.student_ema import StudentWeightEMA


class _Student(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.base = torch.nn.Parameter(torch.full((2, 3), 2.0, dtype=dtype), requires_grad=False)
        self.adapter = torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype))
        self.bias = torch.nn.Parameter(torch.tensor([0.5, -0.5], dtype=dtype))

    def forward(self, inputs):
        return torch.nn.functional.linear(inputs, self.base + self.adapter, self.bias)


class _ReshardingStudent(torch.nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.adapter = torch.nn.Parameter(parameter)
        object.__setattr__(self, "_sharded_adapter", self.adapter)
        self.reshard_calls = 0

    def reshard(self):
        self.reshard_calls += 1
        self.adapter = self._sharded_adapter


def _assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    assert actual["decay"] == expected["decay"]
    assert actual["num_updates"] == expected["num_updates"]
    assert actual["shadow"].keys() == expected["shadow"].keys()
    for name in actual["shadow"]:
        torch.testing.assert_close(actual["shadow"][name], expected["shadow"][name], rtol=0, atol=0)


def _student_ema_distributed_worker(rank, world_size, init_file, checkpoint_directory):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        initial = torch.arange(15, dtype=torch.bfloat16).reshape(5, 3)
        module = torch.nn.Module()
        module.adapter = torch.nn.Parameter(distribute_tensor(initial, mesh, (Shard(0),)))
        module.base = torch.nn.Parameter(distribute_tensor(initial + 50, mesh, (Shard(0),)), requires_grad=False)
        base_before = module.base.full_tensor().clone()
        assert module.adapter.to_local().shape == ((3, 3) if rank == 0 else (2, 3))

        ema = StudentWeightEMA(module)
        assert ema.shadow.keys() == {"adapter"}
        shadow = ema.shadow["adapter"]
        assert isinstance(shadow, DTensor) and shadow.dtype == torch.float32
        assert shadow.device_mesh == mesh and shadow.placements == (Shard(0),)
        assert not shadow.requires_grad and shadow.grad_fn is None
        with torch.no_grad():
            module.adapter.to_local().add_(10 * (rank + 1))
        current = initial.float() + torch.tensor([10, 10, 10, 20, 20]).reshape(5, 1)
        expected = 0.99 * initial.float() + 0.01 * current
        ema.update()
        torch.testing.assert_close(shadow.full_tensor(), expected)
        assert ema.num_updates == 1

        with ema.average_parameters():
            torch.testing.assert_close(module.adapter.full_tensor(), expected.bfloat16(), rtol=0, atol=0)
            torch.testing.assert_close(module.base.full_tensor(), base_before, rtol=0, atol=0)
        torch.testing.assert_close(module.adapter.full_tensor(), current.bfloat16(), rtol=0, atol=0)
        ema.copy_to()
        torch.testing.assert_close(module.adapter.full_tensor(), expected.bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(module.base.full_tensor(), base_before, rtol=0, atol=0)

        dcp.save({"student_ema": ema.state_dict()}, checkpoint_id=checkpoint_directory)
        restored_module = torch.nn.Module()
        restored_module.adapter = torch.nn.Parameter(distribute_tensor(torch.zeros_like(initial), mesh, (Shard(0),)))
        restored_ema = StudentWeightEMA(restored_module)
        restored_state = {"student_ema": restored_ema.state_dict()}
        dcp.load(restored_state, checkpoint_id=checkpoint_directory)
        restored_ema.load_state_dict(restored_state["student_ema"])
        restored_shadow = restored_ema.shadow["adapter"]
        assert isinstance(restored_shadow, DTensor) and restored_shadow.dtype == torch.float32
        assert restored_shadow.device_mesh == mesh and restored_shadow.placements == (Shard(0),)
        assert restored_shadow.to_local().shape == shadow.to_local().shape
        torch.testing.assert_close(restored_shadow.full_tensor(), shadow.full_tensor(), rtol=0, atol=0)
        assert restored_ema.decay == ema.decay and restored_ema.num_updates == ema.num_updates
        restored_ema.copy_to()
        torch.testing.assert_close(restored_module.adapter.full_tensor(), expected.bfloat16(), rtol=0, atol=0)
        with torch.no_grad():
            restored_module.adapter.copy_(module.adapter)
            restored_module.adapter.to_local().add_(2)
            module.adapter.to_local().add_(2)
        ema.update()
        restored_ema.update()
        torch.testing.assert_close(restored_shadow.full_tensor(), shadow.full_tensor(), rtol=0, atol=0)
        assert restored_ema.num_updates == ema.num_updates == 2

        # A root preview may register a full parameter until explicit reshard.
        resharding = _ReshardingStudent(distribute_tensor(initial, mesh, (Shard(0),)))
        with patch("lightx2v_train.trainers.dmd.student_ema.FSDPModule", _ReshardingStudent):
            reshard_ema = StudentWeightEMA(resharding)
            with torch.no_grad():
                resharding.adapter.to_local().add_(4)
            reshard_ema.update()
            original = resharding.adapter.full_tensor().clone()
            calls_before = resharding.reshard_calls
            try:
                with reshard_ema.average_parameters():
                    resharding.adapter = torch.nn.Parameter(resharding.adapter.full_tensor())
                    assert not isinstance(resharding.adapter, DTensor)
                    raise RuntimeError("preview failed")
            except RuntimeError as error:
                assert str(error) == "preview failed"
            assert resharding.reshard_calls > calls_before
            assert isinstance(resharding.adapter, DTensor)
            torch.testing.assert_close(resharding.adapter.full_tensor(), original, rtol=0, atol=0)
            previous_shadow = reshard_ema.shadow["adapter"].full_tensor().clone()
            reshard_ema.update()
            torch.testing.assert_close(
                reshard_ema.shadow["adapter"].full_tensor(),
                0.99 * previous_shadow + 0.01 * original.float(),
            )
            assert reshard_ema.num_updates == 2
    finally:
        dist.destroy_process_group()


class StudentWeightEMATest(unittest.TestCase):
    def test_analytical_update_after_optimizer_step_ignores_frozen_base(self):
        module = _Student()
        ema = StudentWeightEMA(module)
        self.assertEqual(ema.decay, 0.99)
        self.assertEqual(ema.num_updates, 0)
        self.assertEqual(ema.shadow.keys(), {"adapter", "bias"})
        before = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
        optimizer = torch.optim.SGD([module.adapter, module.bias], lr=0.5)
        for parameter in (module.adapter, module.bias):
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        ema.update()
        for name in ema.shadow:
            expected = 0.99 * before[name] + 0.01 * getattr(module, name).detach()
            torch.testing.assert_close(ema.shadow[name], expected)
        self.assertEqual(ema.num_updates, 1)
        torch.testing.assert_close(module.base, before["base"], rtol=0, atol=0)
        with ema.average_parameters():
            torch.testing.assert_close(module.base, before["base"], rtol=0, atol=0)
        ema.copy_to()
        torch.testing.assert_close(module.base, before["base"], rtol=0, atol=0)

    def test_bfloat16_trainable_parameters_have_independent_fp32_shadows(self):
        module = _Student(torch.bfloat16)
        ema = StudentWeightEMA(module)
        before = {name: value.clone() for name, value in ema.shadow.items()}
        for name, shadow in ema.shadow.items():
            parameter = getattr(module, name)
            self.assertEqual(shadow.dtype, torch.float32)
            self.assertFalse(shadow.requires_grad)
            self.assertIsNone(shadow.grad_fn)
            self.assertNotEqual(shadow.data_ptr(), parameter.data_ptr())
        with torch.no_grad():
            module.adapter.fill_(7)
            module.bias.fill_(3)
        ema.update()
        for name in ema.shadow:
            expected = 0.99 * before[name] + 0.01 * getattr(module, name).detach().float()
            torch.testing.assert_close(ema.shadow[name], expected)
        ema.copy_to()
        for name, shadow in ema.shadow.items():
            self.assertEqual(getattr(module, name).dtype, torch.bfloat16)
            torch.testing.assert_close(getattr(module, name), shadow.bfloat16(), rtol=0, atol=0)

    def test_ema_operations_preserve_rng_and_existing_input_and_parameter_gradients(self):
        module = _Student()
        inputs = torch.tensor([[1.0, -2.0, 0.5]], requires_grad=True)
        module(inputs).square().mean().backward()
        input_gradient = inputs.grad.clone()
        gradients = {name: parameter.grad.clone() for name, parameter in module.named_parameters() if parameter.requires_grad}
        rng_before = torch.random.get_rng_state()
        ema = StudentWeightEMA(module)
        with torch.no_grad():
            module.adapter.add_(0.25)
        ema.update()
        with ema.average_parameters():
            module(inputs)
        ema.copy_to()
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
        torch.testing.assert_close(inputs.grad, input_gradient, rtol=0, atol=0)
        for name, gradient in gradients.items():
            torch.testing.assert_close(getattr(module, name).grad, gradient, rtol=0, atol=0)
            self.assertFalse(ema.shadow[name].requires_grad)
            self.assertIsNone(ema.shadow[name].grad_fn)
        self.assertIsNone(module.base.grad)

    def test_average_parameters_restores_original_weights_after_exception(self):
        module = _Student(torch.bfloat16)
        ema = StudentWeightEMA(module)
        with torch.no_grad():
            module.adapter.add_(2)
            module.bias.sub_(1)
        ema.update()
        before = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
        parameter_ids = {name: id(parameter) for name, parameter in module.named_parameters()}
        ema_before = copy.deepcopy(ema.state_dict())
        with self.assertRaisesRegex(RuntimeError, "preview failed"):
            with ema.average_parameters() as averaged:
                self.assertIs(averaged, module)
                for name, shadow in ema.shadow.items():
                    torch.testing.assert_close(getattr(module, name), shadow.bfloat16(), rtol=0, atol=0)
                raise RuntimeError("preview failed")
        for name, parameter in module.named_parameters():
            self.assertEqual(id(parameter), parameter_ids[name])
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        _assert_state_equal(ema.state_dict(), ema_before)

    def test_checkpoint_round_trip_restores_fp32_shadows_and_update_count(self):
        module = _Student(torch.bfloat16)
        ema = StudentWeightEMA(module, decay=0.9)
        for _ in range(2):
            with torch.no_grad():
                module.adapter.add_(1)
                module.bias.sub_(0.5)
            ema.update()
        state = ema.state_dict()
        self.assertEqual(state.keys(), {"decay", "num_updates", "shadow"})
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location="cpu", weights_only=True)
        restored_module = _Student(torch.bfloat16)
        original_weights = {name: parameter.detach().clone() for name, parameter in restored_module.named_parameters()}
        restored = StudentWeightEMA(restored_module, decay=0.9)
        restored.load_state_dict(saved)
        _assert_state_equal(restored.state_dict(), ema.state_dict())
        for name, parameter in restored_module.named_parameters():
            torch.testing.assert_close(parameter, original_weights[name], rtol=0, atol=0)
        restored.copy_to()
        for name, shadow in ema.shadow.items():
            torch.testing.assert_close(getattr(restored_module, name), shadow.bfloat16(), rtol=0, atol=0)

    def test_invalid_checkpoint_names_shapes_and_recipe_fail_before_mutation(self):
        ema = StudentWeightEMA(_Student())
        ema.update()
        before = copy.deepcopy(ema.state_dict())
        for invalid_field in ("names", "shape", "dtype", "decay", "num_updates"):
            with self.subTest(invalid_field=invalid_field):
                invalid = copy.deepcopy(before)
                invalid["shadow"]["adapter"].fill_(100)
                invalid["num_updates"] = 99
                if invalid_field == "names":
                    invalid["shadow"]["wrong_bias"] = invalid["shadow"].pop("bias")
                elif invalid_field == "shape":
                    invalid["shadow"]["bias"] = torch.ones(3)
                elif invalid_field == "dtype":
                    invalid["shadow"]["bias"] = invalid["shadow"]["bias"].bfloat16()
                elif invalid_field == "decay":
                    invalid["decay"] = 0.5
                else:
                    invalid["num_updates"] = -1
                with self.assertRaises(RuntimeError):
                    ema.load_state_dict(invalid)
                _assert_state_equal(ema.state_dict(), before)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU gloo distributed backend is unavailable")
    def test_two_rank_uneven_dtensor_update_swap_restore_and_dcp_round_trip(self):
        with tempfile.TemporaryDirectory(prefix="student_ema_gloo_") as temporary_directory:
            init_file = str(Path(temporary_directory) / "init")
            checkpoint_directory = str(Path(temporary_directory) / "checkpoint")
            context = mp.spawn(
                _student_ema_distributed_worker,
                args=(2, init_file, checkpoint_directory),
                nprocs=2,
                join=False,
            )
            deadline = time.monotonic() + 60
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.fail("Two-rank student EMA CPU test exceeded its 60-second timeout.")
                    if context.join(timeout=remaining):
                        break
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
