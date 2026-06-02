import torch

from lightx2v.models.networks.wan.infer.s2v.wan_causal_audio_module import CausalAudioEncoder
from lightx2v_platform.base.global_var import AI_DEVICE

_MODULE_CACHE = {}


def mm_weight_fp32(linear, input_2d):
    """FP32 matmul for MMWeight when autocast does not apply (e.g. AdaIN)."""
    inp = input_2d.float()
    weight = linear.weight.float()
    bias = linear.bias.float() if linear.bias is not None else None
    if bias is not None:
        return torch.addmm(bias, inp, weight)
    return torch.mm(inp, weight)


def _sync_casual_audio_weights(module, enc_w):
    module.weights.data.copy_(enc_w.weights.tensor.to(module.weights.dtype))
    dst, src = module.encoder, enc_w.encoder
    dst.conv1_local.conv.weight.data.copy_(src.conv1_local_weight.tensor)
    dst.conv1_local.conv.bias.data.copy_(src.conv1_local_bias.tensor)
    dst.conv2.conv.weight.data.copy_(src.conv2_weight.tensor)
    dst.conv2.conv.bias.data.copy_(src.conv2_bias.tensor)
    dst.conv3.conv.weight.data.copy_(src.conv3_weight.tensor)
    dst.conv3.conv.bias.data.copy_(src.conv3_bias.tensor)
    dst.padding_tokens.data.copy_(src.padding_tokens.tensor)
    if hasattr(dst, "conv1_global"):
        dst.conv1_global.conv.weight.data.copy_(src.conv1_global_weight.tensor)
        dst.conv1_global.conv.bias.data.copy_(src.conv1_global_bias.tensor)
        dst.final_linear.weight.data.copy_(src.final_linear.weight.t())
        dst.final_linear.bias.data.copy_(src.final_linear.bias)


def _get_causal_audio_module(enc_w, audio_dim, out_dim, num_layers, num_token, enable_adain):
    key = (audio_dim, out_dim, num_layers, num_token, enable_adain, str(AI_DEVICE))
    if key not in _MODULE_CACHE:
        module = CausalAudioEncoder(
            dim=audio_dim,
            num_layers=num_layers,
            out_dim=out_dim,
            num_token=num_token,
            need_global=enable_adain,
        ).to(device=AI_DEVICE, dtype=enc_w.weights.tensor.dtype)
        module.eval()
        _MODULE_CACHE[key] = module
    return _MODULE_CACHE[key]


def apply_casual_audio_encoder(
    encoder_weights,
    features,
    num_heads,
    enable_adain,
    audio_dim=1024,
    out_dim=5120,
    num_layers=25,
):
    num_layers = encoder_weights.weights.tensor.shape[1]
    module = _get_causal_audio_module(encoder_weights, audio_dim, out_dim, num_layers, num_heads, enable_adain)
    _sync_casual_audio_weights(module, encoder_weights)
    return module(features)
