# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Vendored from Wan2.2/wan/modules/s2v/s2v_utils.py
import numpy as np
import torch


def rope_precompute(x, grid_sizes, freqs, start=None):
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2

    if type(freqs) is list:
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = torch.view_as_complex(x.detach().reshape(b, s, n, -1, 2).to(torch.float64))
    seq_bucket = [0]
    if not isinstance(grid_sizes, list):
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not isinstance(g, list):
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1, seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(), (-t_f - f_o).item() + 1, seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1, seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1, seq_w).astype(int).tolist()

                    assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                    freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][f_sam].conj()
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1)

                    freqs_i = torch.cat(
                        [
                            freqs_0.expand(seq_f, seq_h, seq_w, -1),
                            freqs[1][h_sam].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                            freqs[2][w_sam].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1),
                        ],
                        dim=-1,
                    ).reshape(seq_len, 1, -1)
                elif t_f < 0:
                    freqs_i = trainable_freqs.unsqueeze(1)
                output[i, seq_bucket[-1] : seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x, precomputed_freqs):
    """Match Wan2.2/wan/modules/s2v/model_s2v.py rope_apply for precomputed freqs."""
    output = []
    for i in range(x.size(0)):
        s = x.size(1)
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(s, x.size(2), -1, 2))
        freqs_i = precomputed_freqs[i, :s]
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        if s < x.size(1):
            x_i = torch.cat([x_i, x[i, s:]])
        output.append(x_i)
    return torch.stack(output).float()


@torch.amp.autocast("cuda", enabled=False)
def apply_precomputed_rope(x, precomputed_freqs):
    return rope_apply(x, precomputed_freqs)
