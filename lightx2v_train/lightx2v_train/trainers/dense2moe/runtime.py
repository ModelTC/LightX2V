"""H3 model loading, native packing, and FSDP2 for Dense2MoE experiments."""

from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from lightx2v_train.model_zoo.native.minimax_h3 import (
    KEYFRAME_NOISE_AUG,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    load_minimax_h3_transformer,
    video_latent_num_frames,
)
from lightx2v_train.runtime.distributed import get_device_mesh

from .moe import replace_ffns


def load_attention_kernel(config):
    """Load a pinned local Hub kernel without importing an external runner."""
    if not config.get("attention_kernel_path"):
        return
    from diffusers.models.attention_dispatch import _HUB_KERNELS_REGISTRY, AttentionBackendName, _resolve_kernel_attr
    from kernels import get_local_kernel

    kernel = get_local_kernel(Path(config["attention_kernel_path"]))
    backend = _HUB_KERNELS_REGISTRY[AttentionBackendName(config["attention_backend"])]
    backend.kernel_fn = _resolve_kernel_attr(kernel, backend.function_attr)
    backend.wrapped_forward_fn = _resolve_kernel_attr(kernel, backend.wrapped_forward_attr)
    backend.wrapped_backward_fn = _resolve_kernel_attr(kernel, backend.wrapped_backward_attr)


def load_model(config, stage="dense", permutations=None, shared_state=None, sharded=True):
    load_attention_kernel(config)
    # Match PDMD fake-score precision: FP32 student master weights, BF16 FSDP
    # compute, FP32 gradient reduction. No global autocast context.
    dtype = torch.float32 if sharded and stage != "dense" else torch.bfloat16
    model = load_minimax_h3_transformer(config["model_path"], torch_dtype=dtype, attention_backend=config["attention_backend"])
    model.requires_grad_(False)
    if stage in ("shared", "moe"):
        replace_ffns(model, permutations, stage, config["moe"], shared_state, balance_weight=config["balance_weight"])
        for index in config["moe"]["layers"]:
            model.transformer_blocks[index].ff.requires_grad_(True)
            if stage == "moe" and not config["moe"].get("train_shared", False):
                model.transformer_blocks[index].ff.shared.requires_grad_(False)
    elif stage == "calibrate":
        for index in config["moe"]["layers"]:
            model.transformer_blocks[index].ff.float().requires_grad_(True)
    if sharded:
        mixed = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
        mesh = get_device_mesh()
        for block in model.transformer_blocks:
            fully_shard(block.ff, mesh=mesh, mp_policy=mixed)
            fully_shard(block, mesh=mesh, mp_policy=mixed)
        for block in model.token_refiner.refiner_blocks:
            fully_shard(block, mesh=mesh, mp_policy=mixed)
        fully_shard(model, mesh=mesh, mp_policy=mixed, reshard_after_forward=False)
    else:
        # Match FSDP's parameter views without rounding RoPE's FP32 buffers.
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
        model.to(device="cuda")
    model.train(stage != "dense")
    if stage != "dense" and sharded:
        model.enable_gradient_checkpointing()
    return model


class AVSample:
    def __init__(self, condition, record, device, *, keyframe=None):
        frames = video_latent_num_frames(record["num_frames"])
        anchors = ("first",) if record.get("task") == "i2av" else ()
        height, width = record["height"] // 16, record["width"] // 16
        layout = build_packed_sequence(condition["text_token_tags"], frames, height, width, audio_latent_num_frames(record["num_frames"]), keyframe_anchors=anchors)
        self.keyframe = keyframe
        if anchors and keyframe is None:
            from lightx2v_train.model_zoo.native.minimax_h3.packing import keyframe_condition_noise

            rows = condition["condition_video_latents"].reshape(1, -1, 96).to(device=device, dtype=torch.float32)
            noise = keyframe_condition_noise(((1, height, width),), (1, 2, 2), 24, device=device)
            self.keyframe = KEYFRAME_NOISE_AUG * rows + (1 - KEYFRAME_NOISE_AUG) * noise
        self.layout = layout.to(device)
        self.condition = condition["prompt_embeds"].to(device).reshape(1, -1, condition["prompt_embeds"].shape[-1]).detach().clone()
        self.shapes = {"video": (1, len(layout.video_indices) - layout.num_condition_video_rows, 96), "audio": (1, len(layout.audio_indices), 32)}

    def forward(self, model, latents, sigmas):
        times, indices = build_row_timesteps(self.layout, sigmas["video"], sigmas["audio"])
        layout = self.layout
        video, audio = model(
            hidden_states=latents["video"] if self.keyframe is None else torch.cat((self.keyframe, latents["video"]), dim=1),
            audio_hidden_states=latents["audio"],
            encoder_hidden_states=self.condition,
            timestep=times.to(latents["video"].device),
            timestep_indices=indices.to(latents["video"].device),
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
            return_dict=False,
        )
        if self.keyframe is not None:
            video = video[:, layout.num_condition_video_rows:]
        return {"video": video, "audio": audio}


def shift_sigma(base, shift):
    return shift * base / (1 + (shift - 1) * base)


def save_ffns(model, layers, path):
    state = {}
    for index in layers:
        values = {}
        for name, parameter in model.transformer_blocks[index].ff.named_parameters():
            full = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
            if dist.get_rank() == 0:
                values[name] = full.detach().cpu()
        if dist.get_rank() == 0:
            state[str(index)] = values
    if dist.get_rank() == 0:
        torch.save(state, path)
    dist.barrier()
