"""Generation loop and iHC operations for a physical-layer prompt KV cache."""

import torch
import torch.nn.functional as F


def generation_layers(config):
    layers = config["num_hidden_layers"]
    loops = config.get("num_loops", 1)
    split = config.get("loop_split")
    if type(loops) is not int or loops < 1:
        raise ValueError(f"Expected a positive integer num_loops, got {loops!r}")
    if split is None:
        if loops != 1 or config.get("hc_type", "none") != "none":
            raise ValueError("Generation loop or iHC requires loop_split")
        return tuple(range(layers))
    if len(split) != 3 or any(type(x) is not int or x < 0 for x in split) or not split[1] or sum(split) != layers:
        raise ValueError(f"Invalid pre/core/post split {split!r} for {layers} physical layers")
    pre, core, post = split
    if config.get("hc_type", "none") != "none":
        if (config["hc_type"], config.get("hc_scope"), config.get("hc_readout"), config.get("hc_gate_recipe")) != ("ihc", "loop_only", "hy4", "hy4") or post == 0:
            raise ValueError("NeoPP accelerated iHC supports loop_only with HY4 gates, HY4 readout and a nonempty post section")
    return tuple(range(pre)) + tuple(range(pre, pre + core)) * loops + tuple(range(pre + core, layers))


def ihc_read(states, weights):
    # iHC uses the identity residual mixer. Keep the streams around the
    # attention/MLP call; only its inputs and outputs have standard hidden size.
    with torch.autocast(device_type=states.device.type, enabled=False):
        full = states.float()
        flat = full.flatten(-2)
        inv_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5)
        logits = F.linear(flat, weights.proj_rw.tensor.float().T) * inv_rms
        pre_logit, post_logit = logits.chunk(2, dim=-1)
        pre = torch.sigmoid(weights.bias_pre.tensor + weights.alpha_pre.tensor * pre_logit) + 1e-6
        post = 2 * torch.sigmoid(weights.bias_post.tensor + weights.alpha_post.tensor * post_logit) + 1e-6
        return (pre.unsqueeze(-1) * full).sum(-2).to(states.dtype), post


def ihc_write(states, update, post):
    return (states.float() + post.unsqueeze(-1) * update.float().unsqueeze(-2)).to(states.dtype)


def hy4_readout(states, weights):
    with torch.autocast(device_type=states.device.type, enabled=False):
        full = states.float()
        flat = full.flatten(-2)
        gates = F.linear(flat, weights.hc_fn.tensor.float())
        gates = gates * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5)
        pre = torch.sigmoid(gates * weights.hc_scale.tensor.float() + weights.hc_base.tensor.float()) + 1e-6
        return (pre.unsqueeze(-1) * full).sum(-2).to(states.dtype)
