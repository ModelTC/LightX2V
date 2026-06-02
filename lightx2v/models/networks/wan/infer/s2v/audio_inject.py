import torch
import torch.nn.functional as F
from einops import rearrange

from lightx2v.models.networks.wan.infer.s2v.wan_ops import cross_attn_forward, mm_weight_autocast_nd
from lightx2v_platform.base.global_var import AI_DEVICE


def _apply_adain(x, temb, adain_linear):
    """Mirror diffusers AdaLayerNorm(chunk_dim=1) used by Wan audio_injector."""
    temb = F.silu(temb)
    emb = mm_weight_autocast_nd(adain_linear, temb)
    shift, scale = emb.chunk(2, dim=-1)
    x = F.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=1e-5)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_audio_inject(block, hidden_states, pre_infer_out, config):
    if not hasattr(block, "audio_inject"):
        return hidden_states

    audio_emb = pre_infer_out.merged_audio_emb
    num_frames = audio_emb.shape[1]
    original_seq_len = pre_infer_out.s2v_extra.get("global_original_seq_len", pre_infer_out.original_seq_len)

    input_hidden_states = hidden_states[:, :original_seq_len].clone()
    input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)

    inj = block.audio_inject
    audio_emb = rearrange(audio_emb, "b t n c -> (b t) n c", t=num_frames)
    n, d = config["num_heads"], config["dim"] // config["num_heads"]
    with torch.amp.autocast(str(AI_DEVICE), dtype=torch.bfloat16):
        if config.get("enable_adain", False) and config.get("adain_mode") == "attn_norm":
            audio_emb_global = rearrange(pre_infer_out.audio_emb_global, "b t n c -> (b t) n c", t=num_frames)
            attn_hidden_states = _apply_adain(input_hidden_states, audio_emb_global[:, 0], inj.adain_linear)
        else:
            attn_hidden_states = F.layer_norm(input_hidden_states, (input_hidden_states.shape[-1],), weight=None, bias=None, eps=1e-6)
        residual_out = cross_attn_forward(inj, attn_hidden_states, audio_emb, None, n, d)
    residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
    hidden_states[:, :original_seq_len] = hidden_states[:, :original_seq_len] + residual_out
    return hidden_states
