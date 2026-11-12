"""Manual two-GPU native T2AV/Ref2AV PDD forward/backward/fusion regression.

Run with ``torchrun --standalone --nproc_per_node=2 -m
lightx2v_train.model_zoo.tests.verify_pdd_av_fsdp``.
"""

import gc
import os

import torch
import torch.distributed as dist
from lightx2v_train.model_capabilities import DistributionMatchingCapability
from lightx2v_train.model_zoo.minimax_h3.tests import test_ref2av_capability as ref_fixtures
from lightx2v_train.model_zoo.minimax_h3.tests.test_turbo_jvp import tiny_setup
from lightx2v_train.model_zoo.native.minimax_h3.modeling import _transformer_class
from lightx2v_train.runtime.fsdp import _build_mp_policy, _iter_shard_plan
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


def reference_setup(device):
    model = ref_fixtures.model_wrapper()
    model.device = device
    model.running_dtype = torch.bfloat16
    model.text_dim = 12
    model.video_latent_channels = model.audio_latent_channels = 2
    model.transformer = _transformer_class()(
        num_attention_heads=2,
        attention_head_dim=8,
        hidden_size=16,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=2,
        audio_in_channels=2,
        patch_size=(1, 2, 2),
        text_dim=12,
        freq_dim=4,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        rope_freq_dim=1,
    ).to(device=device, dtype=torch.bfloat16)
    for name in ("proj_in", "audio_proj_in", "proj_out", "audio_proj_out", "time_embedder"):
        getattr(model.transformer, name).float()
    cached = ref_fixtures.reference_condition()
    cached["prompt_embeds"] = torch.randn(1, 4, 12)
    cached["references"][0]["video_latents"] = torch.randn(6, 8)
    cached["references"][1]["audio_latents"] = torch.randn(4, 2)
    capability = model.ensure_capabilities().require(DistributionMatchingCapability)
    sample, condition = ref_fixtures.Ref2AVCapabilityTest().prepare(capability, cached)
    shape = capability.latent_shape(sample, [{"value": [124, 64, 96]}], lambda value: value)
    return model, capability, capability.initial_latents(shape, torch.float32, lambda value: value), None, condition


def run(reference, mesh, rank):
    torch.manual_seed(42)
    device = torch.device("cuda", rank)
    model, _, state, _, condition = reference_setup(device) if reference else tiny_setup(device, torch.bfloat16)
    model.config["training"] = {"method": "pdd", "pdd": {"video_shift": 12, "audio_shift": 3}}
    model.add_lora(4, 4, ["to_q", "to_k", "to_v", "to_out.0", "adaln_proj.linear", "norm_out.linear"])
    model.set_lora_trainable()
    model.set_lora_param_dtype("fp32")
    model.enable_pdd(32)
    model.enable_gradient_checkpointing()
    for modules, reshard, overrides in _iter_shard_plan(model.fsdp2_shard_plan({})):
        precision = {"param_dtype": "bf16", "reduce_dtype": "fp32", "cast_forward_inputs": False, **(overrides or {})}
        fully_shard(modules, mesh=mesh, reshard_after_forward=reshard, mp_policy=_build_mp_policy(precision))
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
    for step in range(2):
        sigma = torch.tensor(0.8 - 0.2 * rank, device=device)
        heads = model.predict_pdd_heads(state, 0, condition, sigma)
        assert heads.video.shape == (1, 32, *state.video.shape[1:])
        assert heads.audio.shape == (1, 32, *state.audio.shape[1:])
        loss = heads.video[:, rank * 31].square().mean() + heads.audio[:, 31 - rank * 31].square().mean()
        loss.backward()
        for name in model.pdd_auxiliary_parameter_names():
            parameter = model.transformer.get_parameter(name)
            assert parameter.dtype == torch.float32 and parameter.grad is not None
            assert torch.isfinite(parameter.grad.to_local()).all()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        grid = torch.linspace(1, 0, 33, device=device, dtype=torch.float64)
        displacement = model.predict_pdd_block(state, rank * 16, (rank + 1) * 16, condition, grid)
        assert torch.isfinite(displacement.video).all() and torch.isfinite(displacement.audio).all()
    print(f"PASS native {'Ref2AV' if reference else 'T2AV'} PDD BF16/FSDP2 rank={rank}, two backward/AdamW steps + fused sampling", flush=True)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl")
    try:
        mesh = init_device_mesh("cuda", (2,))
        run(False, mesh, rank)
        run(True, mesh, rank)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
