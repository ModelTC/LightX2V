"""PDD regression contracts and numerical parity with the pinned FastGen source.

Set FASTGEN_REFERENCE_SOURCE to a checkout of commit
898e757836dc244192533fe6ea797906799dd745 to run source-based parity tests.
"""

import ast
import os
import unittest
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from lightx2v_train.trainers.pdd.core import (
    advance_blocks,
    block_weights,
    map_tensors,
    modalities,
    pdd_loss_from_prediction,
    sample_supervision_indices,
    teacher_target,
)


@dataclass
class Joint:
    video: torch.Tensor
    audio: torch.Tensor
    shape: object


def shifted(t, shift):
    return shift * t / (1 + (shift - 1) * t)


def av_schedule(t):
    return {"video": t, "audio": shifted(t, 0.25)}


class PDDCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.grid = torch.tensor([1, 0.8, 0.5, 0.2, 0], dtype=torch.float64)

    def test_joint_advances_use_each_modality_sigma_and_fused_means(self):
        state = Joint(torch.randn(1, 3), torch.randn(1, 2), ("shape",))
        heads = Joint(torch.randn(1, 4, 3), torch.randn(1, 4, 2), state.shape)
        result = advance_blocks(state, heads, 0, 2, self.grid, sigma_schedule=av_schedule)
        weights, total = block_weights(0, 2, self.grid, sigma_schedule=av_schedule)
        fused = Joint(*(torch.einsum("n,bnc->bc", weights[k].float(), modalities(heads)[k]) for k in ("video", "audio")), state.shape)
        other = advance_blocks(state, fused, 0, 2, self.grid, sigma_schedule=av_schedule, fused=True)
        for key in ("video", "audio"):
            expected = modalities(state)[key] + modalities(fused)[key] * total[key].float()
            torch.testing.assert_close(modalities(result)[key], expected, rtol=0, atol=0)
            torch.testing.assert_close(modalities(other)[key], expected, rtol=0, atol=0)
        self.assertNotEqual(total["video"].item(), total["audio"].item())
        self.assertEqual(result.shape, state.shape)

    def test_loss_stops_teacher_and_prefix_gradients_and_detaches_carry(self):
        state = torch.randn(1, 3, requires_grad=True)
        heads = torch.randn(1, 4, 3, requires_grad=True)
        teacher_parameter = torch.nn.Parameter(torch.tensor(0.7))
        calls = []

        def predict(x, time):
            calls.append((torch.is_grad_enabled(), x.requires_grad, time.clone()))
            return x * teacher_parameter + time[:, None]

        losses, carry = pdd_loss_from_prediction(state, heads, 0, self.grid, predict, supervised_indices=torch.tensor([[2]]), next_idx=3, solver="midpoint")
        self.assertEqual(losses["total_loss"].dtype, torch.float64)
        losses["total_loss"].backward()
        self.assertIsNone(state.grad)
        self.assertIsNone(teacher_parameter.grad)
        self.assertTrue(bool((heads.grad[:, 2] != 0).any()))
        self.assertEqual(heads.grad[:, [0, 1, 3]].abs().sum().item(), 0)
        self.assertFalse(carry.requires_grad)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(not enabled and not needs_grad for enabled, needs_grad, _ in calls))

    def test_head_sampling_is_unique_and_terminal_solver_forwards_do_not_diverge(self):
        starts = torch.tensor([0, 2])
        heads = sample_supervision_indices(starts, 4, 2, 2, batch_size=2, generator=torch.Generator().manual_seed(10))
        self.assertEqual(sorted(heads[0].tolist()), [0, 1])
        self.assertEqual(sorted(heads[1].tolist()), [2, 3])
        with self.assertRaises(ValueError):
            sample_supervision_indices(3, 4, 2, 2)
        calls = []

        def predict(x, time):
            calls.append(time.clone())
            return x + time[:, None]

        state = torch.zeros(2, 2)
        target = teacher_target(state, torch.tensor([0, 3]), self.grid, predict, solver="heun")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][-1].item(), 0)
        torch.testing.assert_close(target[-1], torch.full((2,), 0.2, dtype=torch.float64))

    def test_loss_is_sum_of_modality_means_not_mean_over_all_elements(self):
        state = Joint(torch.zeros(1, 10), torch.zeros(1, 2), None)
        heads = Joint(torch.ones(1, 4, 10, requires_grad=True), torch.full((1, 4, 2), 2.0, requires_grad=True), None)
        losses, _ = pdd_loss_from_prediction(state, heads, 0, self.grid, lambda x, t: map_tensors(torch.zeros_like, x), sigma_schedule=av_schedule, supervised_indices=torch.tensor([[0, 1]]))
        self.assertEqual(losses["video_pdd_loss"].item(), 1)
        self.assertEqual(losses["audio_pdd_loss"].item(), 4)
        self.assertEqual(losses["total_loss"].item(), 5)


def _dispatch(fn, primary, *args, **kwargs):
    if not isinstance(primary, dict):
        return fn(primary, *args, **kwargs)
    return {key: fn(value, *(x[key] if isinstance(x, dict) else x for x in args), **{name: x[key] if isinstance(x, dict) else x for name, x in kwargs.items()}) for key, value in primary.items()}


def _first(value):
    return next(iter(value.values())) if isinstance(value, dict) else value


def _assert_tensor(condition, message, error_type=AssertionError):
    if not bool(condition):
        raise error_type(message)


def _assert_same(**kwargs):
    values = [v for k, v in kwargs.items() if k not in {"shape", "device", "dtype"}]
    for tensors in zip(*(tuple(v.values()) if isinstance(v, dict) else (v,) for v in values)):
        if kwargs.get("shape") is True:
            assert all(x.shape == tensors[0].shape for x in tensors)


def _source_definitions(path, names, class_name=None):
    source = ast.parse(path.read_text())
    if class_name:
        source = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]


def load_reference(root):
    """Execute the actual pinned functions, replacing only framework plumbing."""
    namespace = {
        "torch": torch,
        "partial": partial,
        "per_modality": _dispatch,
        "assert_tensor": _assert_tensor,
        "assert_same": _assert_same,
        "get_device": lambda x: _first(x).device,
        "get_dtype": lambda x: _first(x).dtype,
        "get_batch_size": lambda x: _first(x).shape[0],
        "expand_like": lambda scale, value: scale.reshape(*scale.shape, *((1,) * (value.ndim - scale.ndim))),
        "to": lambda x, **kwargs: _dispatch(lambda t: t.to(**kwargs), x),
        "cast_like": lambda x, y: _dispatch(lambda a, b: a.to(b.dtype), x, y),
        "index": lambda x, indices: _dispatch(lambda t: t[indices], x),
        "to_dict": lambda x, suffix="": {f"{k}_{suffix}": v for k, v in x.items()} if isinstance(x, dict) else {suffix: x},
        "high_precision_dtype": lambda *args: torch.float64,
        "to_high_precision": lambda *args: tuple(t.double() for t in args),
        "einsum_without_autocast": torch.einsum,
        "F": torch.nn.functional,
    }
    network = root / "fastgen/networks/utils/pdd_mixin.py"
    method = root / "fastgen/methods/trajectory_matching/pdd.py"
    source = ast.parse(network.read_text())
    constants = [node for node in source.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "RK_STAGES"]
    definitions = constants + _source_definitions(network, {"_validate_finite_nonzero_tensor"})
    definitions += _source_definitions(root / "fastgen/methods/common_loss.py", {"mse_loss", "mean_with_loss_scale"})
    methods = _source_definitions(
        network,
        {
            "_validate_pdd_indices",
            "pdd_block_weights",
            "pdd_advance_prediction",
            "pdd_advance_blocks",
            "pdd_teacher_target",
            "_pdd_rk_mean_velocity",
        },
        "PDDMixin",
    )
    methods += _source_definitions(method, {"_validate_pdd_end_idx", "_compute_pdd_loss_from_prediction"}, "PDDModelMixin")
    definitions.append(ast.ClassDef(name="Reference", bases=[], keywords=[], body=methods, decorator_list=[]))
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *definitions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(network), "exec"), namespace)
    return namespace["Reference"]


class Schedule:
    def __init__(self, shift=1):
        self.shift = shift
        self.is_unwarped = shift == 1

    def sigma(self, t):
        return shifted(t, self.shift)

    def sigma_prime(self, t):
        return self.shift / (1 + (self.shift - 1) * t).square()

    def determinant(self, start, end):
        # ShiftedRFNoiseSchedule's stable closed form.
        return self.shift * (end - start) / ((1 + (self.shift - 1) * start) * (1 + (self.shift - 1) * end))


@unittest.skipUnless(os.environ.get("FASTGEN_REFERENCE_SOURCE"), "Set FASTGEN_REFERENCE_SOURCE for pinned-source parity")
class FastGenParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Reference = load_reference(Path(os.environ["FASTGEN_REFERENCE_SOURCE"]))

    def reference(self, multimodal):
        ref = self.Reference()
        ref.pdd_steps = 4
        ref.pdd_primary_schedule = Schedule()
        ref.noise_scheduler = SimpleNamespace(schedules={"video": Schedule(), "audio": Schedule(0.25)} if multimodal else Schedule())
        ref.precision_decoding = torch.float32
        ref.net = ref
        ref.net_pred_type = "flow"
        ref.device = torch.device("cpu")
        return ref

    def test_materialized_and_fused_advances_match_official_source(self):
        torch.manual_seed(11)
        grid = torch.tensor([1, 0.83, 0.5, 0.17, 0], dtype=torch.float64)
        for multi in (False, True):
            ref = self.reference(multi)
            state = {"video": torch.randn(2, 3), "audio": torch.randn(2, 2)} if multi else torch.randn(2, 3)
            pred = _dispatch(lambda x: torch.randn(x.shape[0], 4, *x.shape[1:]), state)
            schedule = av_schedule if multi else None
            for start, end, allow in ((0, 3, False), (1, 1, True), (torch.tensor([0, 2]), torch.tensor([2, 4]), False), (torch.tensor([0, 2]), torch.tensor([[0, 2, 4], [2, 3, 4]]), True)):
                expected = ref.pdd_advance_blocks(state, pred, start, end, grid, allow_empty=allow, precision_decoding=torch.float32)
                actual = advance_blocks(state, pred, start, end, grid, sigma_schedule=schedule, allow_empty=allow)
                for key in modalities(actual):
                    torch.testing.assert_close(modalities(actual)[key], modalities(expected)[key], rtol=1e-6, atol=2e-7)
            fused = _dispatch(torch.randn_like, state)
            expected = ref.pdd_advance_blocks(state, fused, 0, 3, grid, fused=True, precision_decoding=torch.float32)
            actual = advance_blocks(state, fused, 0, 3, grid, sigma_schedule=schedule, fused=True)
            for key in modalities(actual):
                torch.testing.assert_close(modalities(actual)[key], modalities(expected)[key], rtol=1e-6, atol=2e-7)

    def test_rk_targets_loss_carry_and_gradients_match_official_source(self):
        torch.manual_seed(3)
        grid = torch.tensor([1, 0.8, 0.5, 0.2, 0], dtype=torch.float64)
        starts, supervised, end = torch.tensor([0, 2]), torch.tensor([[0, 2], [2, 3]]), torch.tensor([3, 4])
        for solver in ("euler", "midpoint", "heun", "kutta4"):
            ref = self.reference(True)
            ref.config = SimpleNamespace(loss_config=SimpleNamespace(type_rk=solver))
            ref._sample_pdd_supervision_idx = lambda *args: supervised

            def native(x, time):
                return {key: value * 0.2 + time[:, None].to(value.dtype).square() for key, value in x.items()}

            def shared_flow(x, time, **kwargs):
                return {key: (value * ref.noise_scheduler.schedules[key].sigma_prime(time).double()[:, None]).to(value.dtype) for key, value in native(x, time).items()}

            ref._guided_teacher = shared_flow
            state = {"video": torch.randn(2, 3), "audio": torch.randn(2, 2)}
            pred = {key: torch.randn(2, 4, value.shape[1], requires_grad=True) for key, value in state.items()}
            ours = {key: value.detach().clone().requires_grad_() for key, value in pred.items()}
            expected, carry = ref._compute_pdd_loss_from_prediction(state, starts, pred, grid, None, None, next_idx=end)
            actual, our_carry = pdd_loss_from_prediction(state, ours, starts, grid, native, sigma_schedule=av_schedule, supervised_indices=supervised, next_idx=end, solver=solver)
            torch.testing.assert_close(actual["total_loss"], expected["total_loss"], rtol=1e-6, atol=1e-7)
            expected["total_loss"].backward()
            actual["total_loss"].backward()
            for key in pred:
                torch.testing.assert_close(ours[key].grad, pred[key].grad, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(our_carry[key], carry[key], rtol=1e-6, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
