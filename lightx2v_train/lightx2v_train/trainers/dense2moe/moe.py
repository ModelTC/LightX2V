"""Disjoint SwiGLU channel partitioning and token-choice grouped GEMM."""

import torch
from torch import nn
from torch.nn import functional as F


class AuxiliaryLoss(torch.autograd.Function):
    """Backpropagate a weighted router loss with its own FFN, not a later branch.

    The trainer uses unscaled BF16 loss backward (no GradScaler or accumulation).
    Keeping this edge inside the FSDP module lets it release each layer promptly.
    """

    @staticmethod
    def forward(ctx, output, loss):
        ctx.loss_dtype = loss.dtype
        return output

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient.new_ones((), dtype=ctx.loss_dtype)


class SwiGLU(nn.Module):
    def __init__(self, hidden, width, *, device=None, dtype=None):
        super().__init__()
        self.up_gate = nn.Linear(hidden, 2 * width, bias=False, device=device, dtype=dtype)
        self.down = nn.Linear(width, hidden, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        up, gate = self.up_gate(x).chunk(2, dim=-1)
        return self.down(up * F.silu(gate))

    @torch.no_grad()
    def copy_dense(self, ff, indices):
        up_gate, down = ff.net[0].proj.weight, ff.net[2].weight
        width = down.shape[1]
        self.up_gate.weight.copy_(up_gate[torch.cat((indices, indices + width))])
        self.down.weight.copy_(down[:, indices])


class SparseSwiGLU(nn.Module):
    """Every original intermediate channel belongs to exactly one expert."""

    def __init__(self, hidden, shared_width, expert_width, experts, top_k, *, balance_weight=0.0, device=None, dtype=None):
        super().__init__()
        self.shared = SwiGLU(hidden, shared_width, device=device, dtype=dtype)
        self.router = nn.Linear(hidden, experts, bias=False, device=device, dtype=dtype)
        self.up_gate = nn.Parameter(torch.empty(experts, hidden, 2 * expert_width, device=device, dtype=dtype))
        self.down = nn.Parameter(torch.empty(experts, expert_width, hidden, device=device, dtype=dtype))
        self.experts, self.top_k = experts, top_k
        self.balance_loss = None
        self.balance_weight = balance_weight

    def forward(self, x):
        original_shape = x.shape
        tokens = x.reshape(-1, original_shape[-1])
        probabilities = F.softmax(F.linear(tokens.float(), self.router.weight.float()), dim=-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        if self.training and torch.is_grad_enabled():
            frequency = F.one_hot(indices, self.experts).float().mean((0, 1))
            balance = self.experts * (frequency.detach() * probabilities.mean(0)).sum()
            self.balance_loss = balance.detach()
        order = indices.flatten().argsort(stable=True)
        # Fixed-size histogram stays on GPU; bincount synchronizes to infer its size.
        counts = torch.zeros(self.experts, device=indices.device, dtype=torch.int32)
        counts.scatter_add_(0, indices.flatten(), torch.ones_like(indices.flatten(), dtype=torch.int32))
        # One zero sentinel per expert keeps grouped GEMM backward's reduction
        # nonempty, including experts that receive no tokens. Sentinels do not
        # contribute to the result or parameter gradients.
        offsets = (counts + 1).cumsum(0).to(torch.int32)
        token_ids = order // self.top_k
        rows = torch.arange(order.numel(), device=order.device) + indices.flatten().index_select(0, order)
        selected = tokens.new_zeros((order.numel() + self.experts, tokens.shape[-1]))
        selected = selected.index_copy(0, rows, tokens.index_select(0, token_ids))
        projected = torch._grouped_mm(selected, self.up_gate, offs=offsets)
        up, gate = projected.chunk(2, dim=-1)
        expert_outputs = torch._grouped_mm((up * F.silu(gate)).contiguous(), self.down, offs=offsets)
        routed = expert_outputs.index_select(0, rows) * weights.flatten().index_select(0, order).to(expert_outputs.dtype)[:, None]
        output = self.shared(tokens).index_add(0, token_ids, routed)
        if self.training and torch.is_grad_enabled() and self.balance_weight:
            output = AuxiliaryLoss.apply(output, balance * self.balance_weight)
        return output.reshape(original_shape)

    @torch.no_grad()
    def copy_dense(self, ff, permutation, shared_width):
        self.shared.copy_dense(ff, permutation[:shared_width])
        groups = permutation[shared_width:].reshape(self.experts, -1)
        original_width = ff.net[2].weight.shape[1]
        for expert, indices in enumerate(groups):
            up_gate_rows = torch.cat((indices, indices + original_width))
            self.up_gate[expert].copy_(ff.net[0].proj.weight[up_gate_rows].T)
            self.down[expert].copy_(ff.net[2].weight[:, indices].T)
        # Same deterministic seed on every rank; nonzero logits break routing ties.
        nn.init.normal_(self.router.weight, std=0.01)


def replace_ffns(transformer, permutations, stage, config, shared_state=None, *, balance_weight=0.0):
    hidden, width = transformer.config.hidden_size, transformer.config.ffn_dim
    shared_width = int(width * config["shared_fraction"])
    experts = config["experts"]
    if (width - shared_width) % experts:
        raise ValueError("FFN remainder must divide evenly among experts")
    for index in config["layers"]:
        block = transformer.transformer_blocks[index]
        dense = block.ff
        indices = permutations[str(index)].long()
        if stage == "shared":
            block.ff = SwiGLU(hidden, shared_width, dtype=torch.float32)
            block.ff.copy_dense(dense, indices[:shared_width])
        else:
            block.ff = SparseSwiGLU(hidden, shared_width, (width - shared_width) // experts, experts, config["top_k"], balance_weight=balance_weight / len(config["layers"]), dtype=torch.float32)
            block.ff.copy_dense(dense, indices, shared_width)
            if shared_state is not None:
                block.ff.shared.load_state_dict(shared_state[str(index)])
            block.ff.shared.requires_grad_(False)
        del dense


def taylor_channels(ff):
    """Sum |weight * gradient| over the up, gate and down neuron triplet."""
    scores = []
    for parameter in (ff.net[0].proj.weight, ff.net[2].weight):
        weight = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
        grad = parameter.grad.full_tensor() if hasattr(parameter.grad, "full_tensor") else parameter.grad
        scores.append((weight.float() * grad.float()).abs())
    width = scores[1].shape[1]
    return (scores[0][:width].sum(1) + scores[0][width:].sum(1) + scores[1].sum(0)).detach().cpu()
