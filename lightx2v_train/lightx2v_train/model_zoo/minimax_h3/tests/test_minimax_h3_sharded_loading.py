import json
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from lightx2v_train.model_zoo.base import BaseModel
from lightx2v_train.model_zoo.capability_adapters.common import CommonTrainableCapability
from lightx2v_train.model_zoo.native.minimax_h3.modeling import (
    _transformer_class,
    init_empty_minimax_h3_transformer,
)
from lightx2v_train.model_zoo.native.minimax_h3.packing import build_packed_sequence, build_row_timesteps
from lightx2v_train.model_zoo.native.minimax_h3.sharded_loading import (
    _checkpoint_to_target_keys,
    stream_load_minimax_h3_transformer,
)


def _tiny_h3(transformer_dir):
    cls = _transformer_class()
    original_dtype = cls._set_default_torch_dtype(torch.bfloat16)
    try:
        model = cls(
            num_attention_heads=2,
            attention_head_dim=8,
            hidden_size=16,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=32,
            in_channels=2,
            audio_in_channels=2,
            patch_size=(1, 1, 1),
            text_dim=12,
            freq_dim=4,
            time_embed_hidden_dim=16,
            time_embed_dim=8,
            rope_freq_dim=1,
        )
    finally:
        torch.set_default_dtype(original_dtype)

    for module_name, module in model.named_modules():
        if module_name and any(pattern in module_name for pattern in model._keep_in_fp32_modules):
            module.to(dtype=torch.float32)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters(), start=1):
            parameter.fill_(index / 1000.0)
    model.save_pretrained(
        transformer_dir,
        safe_serialization=True,
        max_shard_size="20KB",
    )


def _full_checkpoint_state(transformer_dir):
    index_path = Path(transformer_dir) / "diffusion_pytorch_model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    state = {}
    for filename in dict.fromkeys(index["weight_map"].values()):
        state.update(load_file(str(Path(transformer_dir) / filename), device="cpu"))
    return state


class _H3TrainingModel(BaseModel):
    """Exercise real LoRA configuration/checkpoint methods without extra components."""

    pipeline_cls = object

    def __init__(self, transformer):
        self.transformer = transformer

    def denoiser_module(self):
        return self.transformer


def _distributed_lora_training_worker(rank, world_size, init_file, transformer_dir):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        lora_config = {
            "rank": 2,
            "alpha": 2,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
            "param_dtype": "fp32",
        }
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )

        def shard(transformer):
            for block in list(transformer.token_refiner.refiner_blocks) + list(transformer.transformer_blocks):
                fully_shard(block, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)
            fully_shard(transformer, mesh=mesh, reshard_after_forward=False, mp_policy=mp_policy)

        transformer = init_empty_minimax_h3_transformer(transformer_dir, torch_dtype=torch.bfloat16)
        original_params = dict(transformer.named_parameters())
        original_dtypes = {name: parameter.dtype for name, parameter in original_params.items()}
        owner = _H3TrainingModel(transformer)
        trainable_capability = CommonTrainableCapability(owner)
        trainable_capability.configure("lora", lora_config)
        for name, parameter in transformer.named_parameters():
            assert parameter.is_meta
            assert parameter.requires_grad == ("lora" in name)
            if "lora" in name:
                assert parameter.dtype == torch.float32
            else:
                base_name = name.replace(".base_layer.", ".")
                assert parameter is original_params[base_name]
                assert parameter.dtype == original_dtypes[base_name]
        del original_params

        # Frozen BF16 base and FP32 trainable LoRA share each FSDP block;
        # the explicit policy changes compute, not the master parameters.
        shard(transformer)
        stream_load_minimax_h3_transformer(transformer, transformer_dir, device=torch.device("cpu"), lora_seed=123)
        selected_params = dict(transformer.named_parameters())
        selected_storage = {name: parameter.to_local().data_ptr() for name, parameter in selected_params.items()}
        trainable_capability.restore("lora")
        for name, parameter in transformer.named_parameters():
            assert parameter is selected_params[name]
            assert parameter.to_local().data_ptr() == selected_storage[name]
            assert not parameter.is_meta
            assert parameter.requires_grad == ("lora" in name)
            expected_dtype = torch.float32 if "lora" in name else original_dtypes[name.replace(".base_layer.", ".")]
            assert parameter.dtype == expected_dtype
        assert selected_params["transformer_blocks.0.attn.to_q.base_layer.weight"].dtype == torch.bfloat16

        compute_dtypes = []

        def record_lora_compute(module, inputs, output):
            compute_dtypes.append((module.weight.dtype, inputs[0].dtype, output.dtype))

        hooks = [module.register_forward_hook(record_lora_compute) for name, module in transformer.named_modules() if ".lora_" in name and isinstance(module, torch.nn.Linear)]
        optimizer = torch.optim.AdamW(trainable_capability.parameters(), lr=1e-3)
        layout = build_packed_sequence(torch.ones(4, dtype=torch.long), 2, 2, 2, 3, patch_size=(1, 1, 1))
        times, time_indices = build_row_timesteps(layout, 0.5, 0.25)
        torch.manual_seed(1234 + rank)
        video, audio = transformer(
            hidden_states=torch.randn(1, 8, 2, dtype=torch.float32),
            audio_hidden_states=torch.randn(1, 6, 2, dtype=torch.float32),
            encoder_hidden_states=torch.randn(1, 4, 12, dtype=torch.bfloat16),
            timestep=times,
            timestep_indices=time_indices,
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
            return_dict=False,
        )
        loss = video.float().square().mean() + audio.float().square().mean()
        assert bool(torch.isfinite(loss))
        loss.backward()
        for hook in hooks:
            hook.remove()
        assert compute_dtypes and all(dtypes == (torch.bfloat16,) * 3 for dtypes in compute_dtypes)
        trainable = list(trainable_capability.parameters())
        assert trainable and all(parameter.dtype == torch.float32 for parameter in trainable)
        assert all(parameter.grad is not None for parameter in trainable)
        gradients = [parameter.grad.to_local() for parameter in trainable]
        assert all(gradient.dtype == torch.float32 for gradient in gradients)
        assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        assert all(parameter.grad is None for parameter in transformer.parameters() if not parameter.requires_grad)
        optimizer.step()
        assert len(optimizer.state) == len(trainable)
        for state in optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                assert state[key].dtype == torch.float32
                assert bool(torch.isfinite(state[key].to_local()).all())

        # Public adapter save/resume: configure the same explicit FP32 option
        # on an eager BF16 base, load the adapter, then shard and compare locally.
        save_dir = str(Path(transformer_dir).parent / "lora_checkpoint")
        owner.save_lora_weights(save_dir)
        dist.barrier()
        saved_state = load_file(str(Path(save_dir) / "pytorch_lora_weights.safetensors"))
        assert saved_state and all(value.dtype == torch.float32 for value in saved_state.values())
        restored = _transformer_class().from_pretrained(transformer_dir, torch_dtype=torch.bfloat16, local_files_only=True)
        restored_owner = _H3TrainingModel(restored)
        restored_capability = CommonTrainableCapability(restored_owner)
        restored_capability.configure("lora", lora_config)
        restored_owner.load_lora_weights_for_resume(save_dir)
        shard(restored)
        restored_capability.restore("lora")
        for name, parameter in restored.named_parameters():
            if "lora" in name:
                assert parameter.dtype == torch.float32 and parameter.requires_grad
                torch.testing.assert_close(parameter.to_local(), selected_params[name].to_local(), rtol=0, atol=0)
            else:
                assert not parameter.requires_grad
                assert parameter.dtype == original_dtypes[name.replace(".base_layer.", ".")]
    finally:
        dist.destroy_process_group()


def _distributed_stream_worker(rank, world_size, init_file, transformer_dir, full_fake=False):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        lora_config = LoraConfig(
            r=2,
            lora_alpha=2,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
        )

        def build_role(*, lora, seed, dtype=torch.bfloat16, mp_policy=None):
            role_model = init_empty_minimax_h3_transformer(
                transformer_dir,
                torch_dtype=dtype,
            )
            if lora:
                role_model = inject_adapter_in_model(
                    lora_config,
                    role_model,
                    adapter_name="default",
                )
            for block in list(role_model.token_refiner.refiner_blocks) + list(role_model.transformer_blocks):
                fully_shard(block, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy or MixedPrecisionPolicy())
            fully_shard(role_model, mesh=mesh, reshard_after_forward=False, mp_policy=mp_policy or MixedPrecisionPolicy())
            stream_load_minimax_h3_transformer(
                role_model,
                transformer_dir,
                device=torch.device("cpu"),
                lora_seed=seed,
            )
            return role_model

        if full_fake:
            fake_model = build_role(
                lora=False,
                seed=124,
                dtype=torch.float32,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    cast_forward_inputs=False,
                ),
            )
            fake_model.train()
            assert all(parameter.dtype == torch.float32 for parameter in fake_model.parameters())
            layout = build_packed_sequence(
                torch.ones(4, dtype=torch.long),
                2,
                2,
                2,
                3,
                patch_size=(1, 1, 1),
            )
            times, time_indices = build_row_timesteps(layout, 0.5, 0.25)
            video, audio = fake_model(
                hidden_states=torch.randn(1, 8, 2, dtype=torch.float32),
                audio_hidden_states=torch.randn(1, 6, 2, dtype=torch.float32),
                encoder_hidden_states=torch.randn(1, 4, 12, dtype=torch.bfloat16),
                timestep=times,
                timestep_indices=time_indices,
                token_tags=layout.token_tags,
                position_ids=layout.position_ids,
                video_indices=layout.video_indices,
                audio_indices=layout.audio_indices,
                text_indices=layout.text_indices,
                return_dict=False,
            )
            loss = video.float().square().mean() + audio.float().square().mean()
            assert bool(torch.isfinite(loss))
            loss.backward()
            assert all(parameter.dtype == torch.float32 for parameter in fake_model.parameters())
            gradients = [parameter.grad.to_local() for parameter in fake_model.parameters() if parameter.grad is not None]
            assert gradients and all(gradient.dtype == torch.float32 for gradient in gradients)
            assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
            return

        # This is the same three-role lifecycle used by DMD: student and fake
        # carry independent LoRA initialization, teacher is frozen/base-only.
        model = build_role(lora=True, seed=123)
        fake_model = build_role(lora=True, seed=124)
        teacher_model = build_role(lora=False, seed=125)

        for role_model in (model, fake_model, teacher_model):
            assert not any(parameter.is_meta for parameter in role_model.parameters())
            assert not any(buffer.is_meta for buffer in role_model.buffers())
            assert role_model.rope.inv_freq.dtype == torch.float32
        local_params = dict(model.named_parameters())
        assert local_params["proj_in.weight"].dtype == torch.float32
        assert local_params["context_embedder.weight"].dtype == torch.bfloat16
        assert local_params["transformer_blocks.0.attn.to_q.base_layer.weight"].dtype == torch.bfloat16

        options = StateDictOptions(full_state_dict=True, cpu_offload=True, strict=False)
        full_state = get_model_state_dict(model, options=options)
        fake_full_state = get_model_state_dict(fake_model, options=options)
        teacher_full_state = get_model_state_dict(teacher_model, options=options)
        if rank == 0:
            checkpoint = _full_checkpoint_state(transformer_dir)
            for role_state in (full_state, fake_full_state, teacher_full_state):
                key_map = _checkpoint_to_target_keys(set(role_state), set(checkpoint))
                for checkpoint_key, expected in checkpoint.items():
                    torch.testing.assert_close(role_state[key_map[checkpoint_key]], expected, rtol=0, atol=0)

            lora_a = [value for key, value in full_state.items() if ".lora_A." in key]
            lora_b = [value for key, value in full_state.items() if ".lora_B." in key]
            fake_lora_a = [value for key, value in fake_full_state.items() if ".lora_A." in key]
            assert lora_a and lora_b
            assert all(value.dtype == torch.bfloat16 for value in lora_a + lora_b)
            assert any(bool(torch.count_nonzero(value)) for value in lora_a)
            assert all(not bool(torch.count_nonzero(value)) for value in lora_b)
            assert any(not torch.equal(left, right) for left, right in zip(lora_a, fake_lora_a))
            assert not any(".lora_" in key for key in teacher_full_state)
    finally:
        dist.destroy_process_group()


class MiniMaxH3ShardedLoadingTest(unittest.TestCase):
    def test_meta_model_matches_checkpoint_mixed_precision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transformer_dir = Path(tmpdir) / "transformer"
            _tiny_h3(transformer_dir)
            model = init_empty_minimax_h3_transformer(
                tmpdir,
                torch_dtype=torch.bfloat16,
            )
            params = dict(model.named_parameters())
            self.assertEqual(params["proj_in.weight"].dtype, torch.float32)
            self.assertEqual(params["time_embedder.linear_1.weight"].dtype, torch.float32)
            self.assertEqual(params["context_embedder.weight"].dtype, torch.bfloat16)
            self.assertEqual(params["transformer_blocks.0.attn.to_q.weight"].dtype, torch.bfloat16)
            self.assertTrue(all(parameter.is_meta for parameter in model.parameters()))
            self.assertTrue(model.rope.inv_freq.is_meta)

    def test_two_rank_meta_fsdp_stream_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transformer_dir = Path(tmpdir) / "transformer"
            _tiny_h3(transformer_dir)
            init_file = Path(tmpdir) / "process_group_init"
            mp.spawn(
                _distributed_stream_worker,
                args=(2, str(init_file), str(transformer_dir)),
                nprocs=2,
                join=True,
            )

    def test_full_fake_fp32_master_bf16_fsdp_forward_backward(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transformer_dir = Path(tmpdir) / "transformer"
            _tiny_h3(transformer_dir)
            mp.spawn(
                _distributed_stream_worker,
                args=(2, str(Path(tmpdir) / "process_group_init"), str(transformer_dir), True),
                nprocs=2,
                join=True,
            )

    def test_opt_in_lora_fp32_master_bf16_fsdp_forward_backward_and_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transformer_dir = Path(tmpdir) / "transformer"
            _tiny_h3(transformer_dir)
            mp.spawn(
                _distributed_lora_training_worker,
                args=(2, str(Path(tmpdir) / "process_group_init"), str(transformer_dir)),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
