"""Shared numerical primitives for linear attention weight modules.

Statistics and affine scans adapted from OpenVDN/vdn-minimax-h3 e02ff077
(Apache-2.0): models/linear_attention/{scan,delta_rule}.py.
"""

import torch

from lightx2v.common.modules.weight_module import WeightModule


class LinearAttentionBase(WeightModule):
    """Model-independent statistics and scans; subclasses define attention."""

    @staticmethod
    def weighted_statistics(keys, values, beta, chunk_blocks=16, use_tf32=False):
        """[blocks,heads,tokens,dim] -> FP32 A=K^T beta K, B=V^T beta K, in bounded chunks."""
        blocks, heads, _, dim = keys.shape
        a = keys.new_empty((blocks, heads, dim, dim), dtype=torch.float32)
        b = torch.empty_like(a)
        with torch.autocast(device_type=keys.device.type, enabled=False):
            for start in range(0, blocks, chunk_blocks):
                stop = min(start + chunk_blocks, blocks)
                k = keys[start:stop].contiguous()
                k32 = k.float()
                weighted_k = (k32 * beta[start:stop, ..., None].float()).contiguous()
                previous = torch.backends.cuda.matmul.allow_tf32
                try:
                    if keys.is_cuda:
                        torch.backends.cuda.matmul.allow_tf32 = use_tf32
                    moment = weighted_k.transpose(-1, -2) @ k32
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = previous
                a[start:stop] = 0.5 * (moment + moment.transpose(-1, -2))
                weighted_v = (values[start:stop] * beta[start:stop, ..., None].to(values.dtype)).contiguous()
                b[start:stop] = (weighted_v.transpose(-1, -2) @ k).float()
        return a, b

    @staticmethod
    def affine_scan(transitions, injections, initial_state, *, reverse=False):
        """FP32 S[i] = injections[i] + S[previous] @ transitions[i].

        Transitions/injections are [blocks, heads, dim, dim], initial_state is
        [heads, dim, dim]. Reverse scans keep results in the original block order.
        """
        states = torch.empty_like(injections)
        blocks = transitions.shape[0]
        indices = range(blocks - 1, -1, -1) if reverse else range(blocks)
        with torch.autocast(device_type=transitions.device.type, enabled=False):
            state = initial_state
            for index in indices:
                torch.baddbmm(injections[index], state, transitions[index], out=states[index])
                state = states[index]
        return states
