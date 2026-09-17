"""Native VDN frame statistics, bidirectional state recurrence and readout.

Math adapted from OpenVDN/vdn-minimax-h3 e02ff077, Apache-2.0,
models/linear_attention/{branch,scan,delta_rule,layers,features}.py.
Only the released c1/anchor-both, vdn_solve, alpha-bridge configuration is used.
"""

import torch
import torch.nn.functional as F

from .vdn_kernels import activate, linear_epilogue, temporal_activate


def frame_statistics(keys, values, beta, chunk_frames=16, use_tf32=False):
    """[F,H,S,D] -> FP32 A=K^T beta K, B=V^T beta K, in bounded chunks."""
    frames, heads, _, dim = keys.shape
    a = keys.new_empty((frames, heads, dim, dim), dtype=torch.float32)
    b = torch.empty_like(a)
    with torch.autocast(device_type=keys.device.type, enabled=False):
        for start in range(0, frames, chunk_frames):
            stop = min(start + chunk_frames, frames)
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


def factor_states(alpha, a, b):
    with torch.autocast(device_type=a.device.type, enabled=False):
        eye = torch.eye(a.shape[-1], device=a.device, dtype=torch.float32).expand_as(a)
        chol = torch.linalg.cholesky(a.float() + eye)
        inverse_l = torch.linalg.solve_triangular(chol, eye, upper=False, left=True)
        inverse = inverse_l.transpose(-1, -2) @ inverse_l
        return alpha.unsqueeze(-1) * inverse, b.float() @ inverse


def scan_states(alpha, a, b, text_state):
    transitions, injections = factor_states(alpha, a, b)
    prefix = torch.empty_like(injections)
    suffix = torch.empty_like(injections)
    with torch.autocast(device_type=a.device.type, enabled=False):
        state = text_state
        for frame in range(alpha.shape[0]):
            torch.baddbmm(injections[frame], state, transitions[frame], out=prefix[frame])
            state = prefix[frame]
        state = text_state
        for frame in range(alpha.shape[0] - 1, -1, -1):
            torch.baddbmm(injections[frame], state, transitions[frame], out=suffix[frame])
            state = suffix[frame]
    return prefix, suffix


def gather_states(prefix, suffix, alpha, text_state, bounds):
    count = alpha.shape[0]
    device = alpha.device
    before = torch.tensor([lo - 1 for lo, _ in bounds], device=device)
    after = torch.tensor([hi + 1 for _, hi in bounds], device=device)
    left = torch.where((before >= 0)[:, None, None, None], prefix[before.clamp(min=0)], text_state)
    right = torch.where((after < count)[:, None, None, None], suffix[after.clamp(max=count - 1)], text_state)
    cumulative = torch.cat((torch.zeros_like(alpha[:1]), torch.log(alpha.clamp_min(1e-12)).cumsum(0)))
    frame = torch.arange(count, device=device)
    left_decay = torch.exp(cumulative[frame + 1] - cumulative[(before + 1).clamp(min=0)])
    right_decay = torch.exp(cumulative[after.clamp(max=count)] - cumulative[frame])
    return left * left_decay.unsqueeze(2) + right * right_decay.unsqueeze(2)


def frame_alpha(weights, means, head_start, heads, dim):
    channels = slice(head_start * dim, (head_start + heads) * dim)
    with torch.autocast(device_type=means.device.type, enabled=False):
        # LightX2V MMWeight stores [in_features, out_features].
        delta = means.float() @ weights.alpha_down.weight.float()
        delta = delta @ weights.alpha_up.weight[:, channels].float()
        delta = delta + weights.alpha_dt_bias.weight[channels].float()
        scale = weights.alpha_a_log.weight[head_start : head_start + heads].float().exp()[:, None]
        return torch.exp(-scale * F.softplus(delta.view(-1, heads, dim)))


def _conv_feature(tokens, spatial_weight, temporal_weight, frames, frame_size, head_start, normalize):
    heads, dim = tokens.shape[-2:]
    height, width = frame_size
    channels = heads * dim
    channel_slice = slice(head_start * dim, (head_start + heads) * dim)
    volume = tokens.reshape(frames, height, width, channels).permute(0, 3, 1, 2)
    volume = F.conv2d(volume, spatial_weight[channel_slice], padding=2, groups=channels)
    x = volume.permute(0, 2, 3, 1).reshape(frames, height * width, channels)
    temporal = temporal_weight[channel_slice].squeeze(1).to(x.dtype).contiguous()
    return temporal_activate(x, temporal, heads, dim, normalize)


def linear_readout(weights, raw_q, raw_k, raw_v, beta, gate, means, layout, head_start=0, use_tf32=True):
    """Return all rows for this head shard; non-target and anchor rows remain zero."""
    heads, dim = raw_q.shape[-2:]
    frames, spatial = layout.num_frames, layout.tokens_per_frame
    output = raw_q.new_zeros(raw_q.shape)
    if frames <= 2:
        return output
    text_length = layout.text_length
    # The Qwen prefix includes vision rows. Text uses no spatial/temporal conv.
    text_k = F.normalize(F.silu(raw_k[:text_length]), dim=-1, eps=1e-6).to(raw_k.dtype)
    text_v = F.silu(raw_v[:text_length])
    a_text, b_text = frame_statistics(text_k.permute(1, 0, 2)[None], text_v.permute(1, 0, 2)[None], beta[:text_length].transpose(0, 1)[None])
    _, injection = factor_states(torch.ones((1, heads, dim), device=raw_q.device), a_text, b_text)
    text_state = injection[0] * 0.5
    del text_k, text_v, a_text, b_text, injection

    # Both generated boundary latent frames belong entirely to dense softmax.
    start = layout.video_start + spatial
    stop = layout.video_start + (frames - 1) * spatial
    count = frames - 2
    # Match the contiguous SP1 reduction layout after Ulysses packs Q/K/V.
    q = activate(raw_q[start:stop].contiguous(), True).view(count, spatial, heads, dim).permute(0, 2, 1, 3).contiguous()
    k = _conv_feature(raw_k[start:stop], weights.k_sp.weight, weights.k_tm.weight, count, layout.frame_size, head_start, True)
    v = _conv_feature(raw_v[start:stop], weights.v_sp.weight, weights.v_tm.weight, count, layout.frame_size, head_start, False)
    k = k.view(count, spatial, heads, dim).permute(0, 2, 1, 3)
    v = v.view(count, spatial, heads, dim).permute(0, 2, 1, 3)
    frame_beta = beta[start:stop].view(count, spatial, heads).permute(0, 2, 1)
    # The released VDN inference path uses TF32 only for this video A statistic.
    a, b = frame_statistics(k, v, frame_beta, use_tf32=use_tf32)
    del k, v, frame_beta
    alpha = frame_alpha(weights, means[1:-1], head_start, heads, dim)
    prefix, suffix = scan_states(alpha, a, b, text_state)
    bounds = [(lo - 1, hi - 1) for lo, hi in layout.bounds[1:-1]]
    state = gather_states(prefix, suffix, alpha, text_state, bounds).to(gate.dtype)
    del prefix, suffix, a, b
    readout = q @ state.transpose(-1, -2)
    output[start:stop] = linear_epilogue(readout, weights.norm.weight, gate[start:stop])
    return output
