"""Shared compute counts: one multiply-add is 2 FLOPs; exclude routing, softmax, quantization, and transfers.

Callers supply actual dimensions/final regions. No session, backend, file, or model state is kept here.
"""


def gemm_flops(m, n, k, batch=1):
    return 2 * batch * m * n * k


def attention_flops(token_pairs, qk_dim, value_dim):
    return 2 * token_pairs * (qk_dim + value_dim)


def selected_token_pairs(mask, q_lengths, k_lengths):
    """Sum binary rectangular block areas with int64 lengths, including actual tails or variable blocks."""
    return int((mask.long() * q_lengths.unsqueeze(-1) * k_lengths.unsqueeze(-2)).sum().item())


def range_token_pairs(q_ranges, k_ranges):
    """Count production-provided, non-overlapping integer [N,2] ranges without reselecting or merging them."""
    q_lengths = q_ranges[:, 1].long() - q_ranges[:, 0].long()
    k_lengths = k_ranges[:, 1].long() - k_ranges[:, 0].long()
    return int((q_lengths * k_lengths).sum().item())


def routed_matmul_flops(routes, weight_shapes):
    """Count routed matmuls for homogeneous experts; weight_shapes gives [output,input] for each FC/gate."""
    return sum(gemm_flops(routes, rows, columns) for rows, columns in weight_shapes)


def convolution_flops(output_elements, weight_shape):
    """Count convolution MACs including padding; the weight input-channel dimension already reflects groups."""
    elements = 1
    for size in weight_shape[1:]:
        elements *= size
    return 2 * output_elements * elements
