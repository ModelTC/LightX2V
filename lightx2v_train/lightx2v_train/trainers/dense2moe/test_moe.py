"""Numerical tests for partitioning, sparse dispatch and router gradients."""

import copy

import pytest
import torch
from diffusers.models.attention import FeedForward

from .loss import FeatureMSE, feature_distillation_loss
from .moe import SparseSwiGLU, SwiGLU


def test_channel_partition():
    torch.manual_seed(7)
    dense = FeedForward(16, inner_dim=64, activation_fn="swiglu", bias=False)
    order = torch.randperm(64)
    pieces = []
    for indices in order.chunk(4):
        part = SwiGLU(16, 16)
        part.copy_dense(dense, indices)
        pieces.append(part)
    x = torch.randn(9, 16)
    torch.testing.assert_close(sum(part(x) for part in pieces), dense(x))


@pytest.mark.parametrize("empty_experts", [False, True])
def test_sparse_dispatch_and_backward(empty_experts):
    torch.manual_seed(8)
    moe = SparseSwiGLU(128, 128, 128, 4, 2, device="cuda", dtype=torch.bfloat16)
    for parameter in moe.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    if empty_experts:
        with torch.no_grad():
            moe.router.weight.zero_()
    x = torch.randn(67, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual = moe(x)
    probs = torch.softmax(x.float() @ moe.router.weight.float().T, dim=-1)
    weights, indices = probs.topk(2, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    expected = moe.shared(x)
    for expert in range(4):
        up, gate = (x @ moe.up_gate[expert]).chunk(2, dim=-1)
        result = (up * torch.nn.functional.silu(gate)) @ moe.down[expert]
        scale = (weights * (indices == expert)).sum(-1).to(result.dtype)
        expected = expected + result * scale[:, None]
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.03)
    variables = [x, *moe.parameters()]
    actual_grad = torch.autograd.grad(actual.float().square().sum(), variables, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.float().square().sum(), variables)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=0.004, rtol=0.06)


def test_identical_features_have_zero_loss():
    x = torch.randn(4, 8, requires_grad=True)
    out = torch.nn.functional.mse_loss(x, x.detach())
    feat = feature_distillation_loss(out, [x], [x.detach()])
    (out + feat).backward()
    assert out == 0 and feat == 0
    assert torch.isfinite(x.grad).all()


def test_feature_loss_matches_float_mse_gradient():
    student = torch.randn(11, 64, dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn_like(student)
    actual = FeatureMSE.apply(student, teacher)
    expected = torch.nn.functional.mse_loss(student.float(), teacher.float())
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, student)[0]
    expected_grad = torch.autograd.grad(expected, student)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_offloaded_feature_target_matches_gpu_loss():
    student = torch.randn(29, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn_like(student)
    target = teacher.cpu().pin_memory()
    actual = FeatureMSE.apply(student, target)
    expected = torch.nn.functional.mse_loss(student.float(), teacher.float())
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, student)[0]
    expected_grad = torch.autograd.grad(expected, student)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_local_router_gradient_matches_explicit_auxiliary_loss():
    torch.manual_seed(19)
    actual = SparseSwiGLU(128, 128, 128, 4, 2, balance_weight=0.05, device="cuda", dtype=torch.bfloat16)
    for parameter in actual.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    expected = copy.deepcopy(actual)
    expected.balance_weight = 0.0
    x = torch.randn(67, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual_loss = actual(x).float().square().mean()
    expected_output = expected(x)
    probabilities = torch.softmax(x.float() @ expected.router.weight.float().T, -1)
    indices = probabilities.topk(2, -1).indices
    frequency = torch.nn.functional.one_hot(indices, 4).float().mean((0, 1))
    balance = 4 * (frequency * probabilities.mean(0)).sum()
    expected_loss = expected_output.float().square().mean() + 0.05 * balance
    actual_grads = torch.autograd.grad(actual_loss, [x, *actual.parameters()])
    expected_grads = torch.autograd.grad(expected_loss, [x, *expected.parameters()])
    assert not actual.balance_loss.requires_grad
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=0.04)
