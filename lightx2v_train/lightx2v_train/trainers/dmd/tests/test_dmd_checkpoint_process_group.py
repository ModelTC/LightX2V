"""DCP planning uses a cached CPU group without changing checkpoint contents."""

import copy
import os
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from lightx2v_train.trainers.dmd.checkpoint import DmdCheckpointManager

MODULE = "lightx2v_train.trainers.dmd.checkpoint"


def _fixture(*, fake_real=False, ema=False):
    owner = SimpleNamespace(
        config={},
        dataloader_train=SimpleNamespace(sampler=None),
        student_train_type="full",
        fake_train_type="full",
        student=SimpleNamespace(
            extra_checkpoint_metadata=lambda: {},
            legacy_extra_checkpoint_metadata=lambda: {},
            extra_training_state=lambda: {},
            load_extra_training_state=Mock(),
        ),
        _validate_checkpoint_metadata=Mock(),
    )
    roles = {}
    for role in ("student", "fake", "fake_real") if fake_real else ("student", "fake"):
        prefix = "" if role == "student" else f"{role}_"
        model = torch.nn.Linear(4, 4, bias=False)
        optimizer, scheduler = Mock(), Mock()
        scheduler.state_dict.return_value = {"last_epoch": 7}
        setattr(owner, f"{prefix}model", model)
        setattr(owner, f"{prefix}optimizer", optimizer)
        setattr(owner, f"{prefix}lr_scheduler", scheduler)
        setattr(owner, f"{role}_train_type", "full")
        roles[role] = SimpleNamespace(
            model=model,
            train_type="full",
            optimizer=optimizer,
            scheduler=scheduler,
            spec=SimpleNamespace(optimizer_attribute=f"{prefix}optimizer", scheduler_attribute=f"{prefix}lr_scheduler"),
        )
    owner.parallel = SimpleNamespace(state_module=lambda: owner.model)
    owner.role_registry = SimpleNamespace(runtimes=lambda: roles)
    if ema:
        owner.student_ema_config = {"enabled": True}
        owner.student_ema = SimpleNamespace(
            decay=0.99,
            num_updates=7,
            shadow={"weight": owner.model.weight.detach().clone()},
            load_state_dict=Mock(),
        )
    return owner, DmdCheckpointManager(owner), roles


def _parallel(model):
    return SimpleNamespace(state_module=lambda: model)


def _assert_state_equal(actual, expected):
    if isinstance(actual, DTensor):
        assert isinstance(expected, DTensor)
        assert actual.placements == expected.placements
        assert actual.device_mesh == expected.device_mesh
        torch.testing.assert_close(actual.to_local(), expected.to_local(), rtol=0, atol=0)
    elif torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _checkpoint_round_trip_worker(rank, init_file, checkpoint):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        mesh = init_device_mesh("cpu", (2,))
        owner, manager, roles = _fixture(fake_real=True, ema=True)
        owner.config = {"distributed": {"timeout_minutes": 0.5}}
        expected = {}
        for index, (role, runtime) in enumerate(roles.items()):
            weight = torch.arange(16, dtype=torch.float32).reshape(4, 4) + index
            runtime.model.weight = torch.nn.Parameter(distribute_tensor(weight, mesh, [Shard(0)]))
            runtime.optimizer = torch.optim.Adam(runtime.model.parameters(), lr=0.01)
            # Warm up real sharded Adam moments without any training-model forward.
            runtime.model.weight.grad = torch.full_like(runtime.model.weight, index + 1.0)
            runtime.optimizer.step()
            setattr(owner, runtime.spec.optimizer_attribute, runtime.optimizer)
            expected[role] = (copy.deepcopy(runtime.model.state_dict()), copy.deepcopy(runtime.optimizer.state_dict()))
        owner.student_ema.shadow = {"weight": owner.model.weight.detach().clone() * 0.5}
        expected_ema = copy.deepcopy(owner.student_ema.shadow)
        group = manager._get_checkpoint_process_group()
        assert group is not dist.group.WORLD
        assert dist.get_backend(group) == "gloo"
        assert dist.get_world_size(group) == 2
        with patch.object(DmdCheckpointManager, "_parallel", side_effect=_parallel):
            manager._save_distributed_state(checkpoint, 7)
            for runtime in roles.values():
                with torch.no_grad():
                    runtime.model.weight.zero_()
                for state in runtime.optimizer.state.values():
                    for value in state.values():
                        if torch.is_tensor(value):
                            value.zero_()
            for value in owner.student_ema.shadow.values():
                value.zero_()
            # A fresh manager must reuse the trainer-owned communication group.
            restored_manager = DmdCheckpointManager(owner)
            assert restored_manager._get_checkpoint_process_group() is group
            restored_manager._load_distributed_state(checkpoint)
        for role, runtime in roles.items():
            _assert_state_equal(runtime.model.state_dict(), expected[role][0])
            _assert_state_equal(runtime.optimizer.state_dict(), expected[role][1])
            runtime.scheduler.load_state_dict.assert_called_once_with({"last_epoch": 7})
        _assert_state_equal(owner.student_ema.shadow, expected_ema)
        owner.student_ema.load_state_dict.assert_called_once()
        assert owner.student_ema.load_state_dict.call_args.args[0]["num_updates"] == 7
        dist.barrier()
    finally:
        dist.destroy_process_group()


class DmdCheckpointProcessGroupTest(unittest.TestCase):
    def test_group_is_lazy_cached_on_owner_and_shared_by_managers(self):
        owner = SimpleNamespace(config={})
        with (
            patch(f"{MODULE}.dist.is_available", return_value=True),
            patch(f"{MODULE}.dist.is_initialized", return_value=True),
            patch(f"{MODULE}.dist.is_gloo_available", return_value=True),
            patch(f"{MODULE}.dist.new_group", return_value=sentinel.group) as new_group,
        ):
            manager = DmdCheckpointManager(owner)
            new_group.assert_not_called()
            self.assertIs(manager._get_checkpoint_process_group(), sentinel.group)
            self.assertIs(manager._get_checkpoint_process_group(), sentinel.group)
            self.assertIs(DmdCheckpointManager(owner)._get_checkpoint_process_group(), sentinel.group)
            self.assertIs(owner._checkpoint_process_group, sentinel.group)
        # Omitting ranks is deliberate: every world rank participates.
        new_group.assert_called_once_with(backend="gloo", timeout=timedelta(minutes=10))

    def test_group_honors_distributed_timeout(self):
        owner = SimpleNamespace(config={"distributed": {"timeout_minutes": 17}})
        with (
            patch(f"{MODULE}.dist.is_available", return_value=True),
            patch(f"{MODULE}.dist.is_initialized", return_value=True),
            patch(f"{MODULE}.dist.is_gloo_available", return_value=True),
            patch(f"{MODULE}.dist.new_group", return_value=sentinel.group) as new_group,
        ):
            self.assertIs(DmdCheckpointManager(owner)._get_checkpoint_process_group(), sentinel.group)
        new_group.assert_called_once_with(backend="gloo", timeout=timedelta(minutes=17))

    def test_non_distributed_paths_do_not_create_or_cache_a_group(self):
        for available, initialized in ((False, False), (True, False)):
            with self.subTest(available=available, initialized=initialized):
                owner = SimpleNamespace(config={})
                with (
                    patch(f"{MODULE}.dist.is_available", return_value=available),
                    patch(f"{MODULE}.dist.is_initialized", return_value=initialized) as is_initialized,
                    patch(f"{MODULE}.dist.is_gloo_available") as is_gloo_available,
                    patch(f"{MODULE}.dist.new_group") as new_group,
                ):
                    self.assertIsNone(DmdCheckpointManager(owner)._get_checkpoint_process_group())
                new_group.assert_not_called()
                is_gloo_available.assert_not_called()
                self.assertFalse(hasattr(owner, "_checkpoint_process_group"))
                if not available:
                    is_initialized.assert_not_called()

    def test_missing_gloo_fails_without_falling_back_to_training_group(self):
        owner = SimpleNamespace(config={})
        with (
            patch(f"{MODULE}.dist.is_available", return_value=True),
            patch(f"{MODULE}.dist.is_initialized", return_value=True),
            patch(f"{MODULE}.dist.is_gloo_available", return_value=False),
            patch(f"{MODULE}.dist.new_group") as new_group,
        ):
            with self.assertRaisesRegex(RuntimeError, "Gloo"):
                DmdCheckpointManager(owner)._get_checkpoint_process_group()
        new_group.assert_not_called()

    def test_every_save_and_load_forwards_group_without_changing_role_layout(self):
        for extra_roles in (False, True):
            for group in (None, sentinel.group):
                with self.subTest(extra_roles=extra_roles, group=group):
                    owner, manager, roles = _fixture(fake_real=extra_roles, ema=extra_roles)
                    state_dicts = {runtime.model: ({"weight": role}, {"state": role}) for role, runtime in roles.items()}

                    def save_state(state, *, checkpoint_id, process_group):
                        Path(checkpoint_id).mkdir(parents=True, exist_ok=True)

                    with (
                        tempfile.TemporaryDirectory() as directory,
                        patch.object(DmdCheckpointManager, "_parallel", side_effect=_parallel),
                        patch.object(DmdCheckpointManager, "_get_checkpoint_process_group", return_value=group),
                        patch(f"{MODULE}.get_state_dict", side_effect=lambda model, optimizer, **kwargs: state_dicts[model]),
                        patch(f"{MODULE}.set_state_dict") as set_state,
                        patch(f"{MODULE}.dcp.save", side_effect=save_state) as save,
                        patch(f"{MODULE}.dcp.load") as load,
                    ):
                        manager._save_distributed_state(directory, 7)
                        manager._load_distributed_state(directory)
                        state_path = str(Path(directory) / "dist_state")
                        expected_keys = {"student_model", "student_optimizer", "fake_model", "fake_optimizer"}
                        if extra_roles:
                            expected_keys.add("student_ema")
                        for operation in (save, load):
                            self.assertEqual(operation.call_count, 2 if extra_roles else 1)
                            combined = operation.call_args_list[0]
                            self.assertEqual(combined.kwargs, {"checkpoint_id": state_path, "process_group": group})
                            self.assertEqual(set(combined.args[0]), expected_keys)
                            for role in ("student", "fake"):
                                self.assertIs(combined.args[0][f"{role}_model"], state_dicts[roles[role].model][0])
                                self.assertIs(combined.args[0][f"{role}_optimizer"], state_dicts[roles[role].model][1])
                            if extra_roles:
                                self.assertIs(combined.args[0]["student_ema"], owner.student_ema.shadow)
                                extra = operation.call_args_list[1]
                                self.assertEqual(extra.kwargs, {"checkpoint_id": str(Path(state_path) / "fake_real"), "process_group": group})
                                self.assertEqual(extra.args[0], {"model": state_dicts[owner.fake_real_model][0], "optimizer": state_dicts[owner.fake_real_model][1]})
                        self.assertEqual(set_state.call_count, len(roles))
                        trainer_state = torch.load(Path(directory) / "trainer_state.pt", weights_only=False)
                        self.assertEqual(trainer_state["dmd_checkpoint_version"], 2)
                        self.assertNotIn("process_group", trainer_state)
                        self.assertNotIn("_checkpoint_process_group", trainer_state)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU Gloo distributed backend is unavailable")
    def test_two_rank_sharded_model_adam_ema_and_extra_role_round_trip(self):
        with tempfile.TemporaryDirectory(prefix="dmd_checkpoint_gloo_") as directory:
            init_file = str(Path(directory) / "init")
            checkpoint = str(Path(directory) / "checkpoint-000000007")
            Path(checkpoint).mkdir()
            context = mp.spawn(_checkpoint_round_trip_worker, args=(init_file, checkpoint), nprocs=2, join=False)
            deadline = time.monotonic() + 45
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.fail("Two-rank DMD checkpoint round trip exceeded its 45-second timeout.")
                    if context.join(timeout=min(remaining, 5)):
                        break
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
