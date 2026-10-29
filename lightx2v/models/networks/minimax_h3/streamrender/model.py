"""Native StreamRender-H3 DiT using LightX2V's H3 weights and prefix cache."""

import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

from lightx2v.models.networks.minimax_h3.autoregressive_model import H3AutoregressiveModel
from lightx2v.models.networks.minimax_h3.infer.autoregressive.streamrender.pre_infer import StreamRenderH3PreInfer
from lightx2v.models.networks.minimax_h3.infer.autoregressive.streamrender.prefix_cache import StreamRenderH3PrefixBank
from lightx2v.models.networks.minimax_h3.infer.autoregressive.streamrender.rope import StreamRenderH3RoPE
from lightx2v.models.networks.minimax_h3.infer.autoregressive.streamrender.transformer_infer import StreamRenderH3OffloadTransformerInfer, StreamRenderH3TransformerInfer
from lightx2v.models.networks.minimax_h3.infer.post_infer import MiniMaxH3PostInfer
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
from lightx2v.models.networks.minimax_h3.packing import unpack_audio_tokens, unpatchify_video_tokens
from lightx2v.models.networks.minimax_h3.streamrender.weights import load_streamrender_rope_frequencies, remap_streamrender_weight


class StreamRenderH3Model(H3AutoregressiveModel, MiniMaxH3Model):
    """Native released DiT; request assembly and streaming VAE stay in runner."""

    def __init__(self, model_path, config, device, lora_path=None, lora_strength=1.0, lora_alpha=None):
        config = dict(config)
        architecture = {
            "seq_parallel": False,
            "tensor_parallel": False,
            "num_layers": 50,
            "num_refiner_layers": 2,
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "hidden_size": 5376,
            "ffn_dim": 14336,
            "freq_dim": 256,
            "rope_freq_dim": 16,
            "norm_eps": 1e-5,
            "qk_norm_eps": 1e-5,
            "final_norm_eps": 1e-5,
            "attn_type": "flash_attn3",
            "rms_type": "torch_native",
            "rope_type": "torch_real_rope",
        }
        for name, value in architecture.items():
            config.setdefault(name, value)
        config["allow_online_adaln_offload"] = True
        checkpoint = Path(config.get("dit_original_ckpt") or Path(model_path) / "backbone")
        self.checkpoint_index = checkpoint / "model.safetensors.index.json"
        self._rope_frequencies = load_streamrender_rope_frequencies(self.checkpoint_index)
        config["dit_original_ckpt"] = str(checkpoint)
        super().__init__(model_path, config, device, lora_path=lora_path, lora_strength=lora_strength, lora_alpha=lora_alpha)
        self.pre_infer.rope = StreamRenderH3RoPE(self._rope_frequencies)

    def _init_infer_class(self):
        self.pre_infer_class = StreamRenderH3PreInfer
        self.transformer_infer_class = StreamRenderH3OffloadTransformerInfer if self.config.get("cpu_offload", False) else StreamRenderH3TransformerInfer
        self.post_infer_class = MiniMaxH3PostInfer

    def _load_shared_cpu_weights(self, unified_dtype, sensitive_layer):
        from lightx2v.models.networks.minimax_h3.shared_block_weights import load_h3_shared_weights

        del unified_dtype

        def validate():
            if not self.block_offload:
                raise ValueError("StreamRender shared weights require block CPU offload")
            if (
                any(self.config.get(key, False) for key in ("tensor_parallel", "weight_auto_quant", "lazy_load", "unload_modules", "dummy_model", "lora_dynamic_apply"))
                or self.lora_path
                or self.config.get("lora_configs")
            ):
                raise ValueError("StreamRender shared weights do not support TP, LoRA or lazy loading")
            if self.dit_quantized and self.config.get("dit_quant_scheme") not in ("fp8-f16-accum", "fp8-f32-accum"):
                raise ValueError("StreamRender shared quantized weights require fp8-f16-accum or fp8-f32-accum")

        def transform(name, tensor):
            converted = {name: self._fp8_f16_checkpoint_tensor(name, tensor, sensitive_layer)} if self.dit_quantized else remap_streamrender_weight(name, tensor)
            return {name: tensor for name, tensor in converted.items() if not any(remove in name for remove in self.remove_keys)}

        def include(name):
            prefix = "transformer_blocks." if self.dit_quantized else "dit.blocks."
            return not name.startswith(prefix) or int(name[len(prefix) :].split(".", 1)[0]) < int(self.config["num_layers"])

        path = self.config["dit_quantized_ckpt"] if self.dit_quantized else self.checkpoint_index.parent
        weights = load_h3_shared_weights(
            path,
            self.config,
            "streamrender-dit-fp8-f16" if self.dit_quantized else "streamrender-dit-bf16",
            include=include,
            shared=lambda name: name.startswith("transformer_blocks."),
            validate=validate,
            transform=transform,
            fused_qkv=bool(self.config.get("use_fused_qkv", False)),
        )
        self._record_checkpoint_metadata(weights)
        return weights

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        """Load only indexed shards; the release also includes a duplicate merged file."""
        index = json.loads(self.checkpoint_index.read_text())
        weight_dict = {}
        for shard_name in sorted(set(index["weight_map"].values())):
            shard = self._load_safetensor_to_dict(str(self.checkpoint_index.parent / shard_name), unified_dtype, sensitive_layer)
            duplicate = weight_dict.keys() & shard.keys()
            if duplicate:
                raise ValueError(f"Duplicate StreamRender checkpoint weights in {shard_name}: {sorted(duplicate)[:4]}")
            weight_dict.update(shard)
        return weight_dict

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        """Map a single source shard, then transfer only this TP rank's slices."""
        del unified_dtype, sensitive_layer
        mapped = {}
        load_device = self._checkpoint_load_device()
        with safe_open(file_path, framework="pt", device="cpu") as source:
            for source_name in source.keys():
                if source_name.startswith("dit.blocks.") and int(source_name.split(".")[2]) >= int(self.config["num_layers"]):
                    continue
                converted = remap_streamrender_weight(source_name, source.get_tensor(source_name))
                for name, tensor in converted.items():
                    if any(remove in name for remove in self.remove_keys):
                        continue
                    local = self._select_tensor_parallel_shard(name, tensor)
                    mapped[name] = local.to(load_device)
        self._record_checkpoint_metadata(mapped)
        return mapped

    def _ensure_prefix_bank(self):
        scheduler = self.scheduler
        sequence_ranks = 1
        if self.seq_p_group is not None:
            import torch.distributed as dist

            sequence_ranks = dist.get_world_size(self.seq_p_group)
        num_heads = int(self.config["num_attention_heads"]) // self.tp_size // sequence_ranks
        signature = (
            scheduler.prefix_cache_capacity,
            num_heads,
            int(self.config["attention_head_dim"]),
            bool(self.config.get("ar_config", {}).get("kv_offload", False)),
        )
        previous_signature = getattr(self, "_prefix_bank_signature", None)
        if signature != previous_signature:
            for bank in getattr(self, "_prefix_bank_pool", {}).values():
                bank.clear()
            if previous_signature is not None:
                self.reset_cuda_graph()
            self._prefix_bank_pool = {}
            self._prefix_raw_workspace = None
            self._prefix_bank_signature = signature
        if not scheduler.capture_prefix or scheduler.prefix_bank is not None:
            return
        bank = self._prefix_bank_pool.get(scheduler.prefix_bank_key)
        if bank is None:
            bank = StreamRenderH3PrefixBank(
                num_layers=int(self.config["num_layers"]),
                capacity=scheduler.prefix_cache_capacity,
                num_heads=num_heads,
                head_dim=int(self.config["attention_head_dim"]),
                dtype=torch.bfloat16,
                device=torch.device(self.device),
                offload=signature[-1],
                sage=self.transformer_infer.use_sage_attention,
                segment_candidates=(self.config.get("ar_config") or {}).get("sage_segment_candidates"),
                raw_workspace=getattr(self, "_prefix_raw_workspace", None),
            )
            self._prefix_bank_pool[scheduler.prefix_bank_key] = bank
            if bank.compact_sage:
                self._prefix_raw_workspace = bank.pool
        scheduler.set_prefix_bank(bank)

    def _seq_parallel_pre_process(self, pre):
        # Norm and RoPE are row-local. Rotate each rank's own rows before the
        # QKV exchange so the fused projection/norm/RoPE path stays available.
        rotary = pre.rotary_emb
        pre = super()._seq_parallel_pre_process(pre)
        size, rank = dist.get_world_size(self.seq_p_group), dist.get_rank(self.seq_p_group)
        padding = pre.hidden_states.shape[0] * size - rotary[0].shape[0]
        pre.rotary_emb = tuple(F.pad(tensor, (0, 0, 0, padding)).chunk(size)[rank].contiguous() for tensor in rotary)
        return pre

    @staticmethod
    def _source_x0(source_tokens, source_rows, output_rows, repo_timesteps, velocity):
        source_indices = source_rows.index_select(0, output_rows)
        x_t = source_tokens.index_select(0, source_indices).to(velocity.device)
        h3_t = 1.0 - repo_timesteps.index_select(0, output_rows).to(velocity.device)
        sigma = 1 - h3_t.to(x_t.dtype)
        return x_t + sigma.unsqueeze(-1) * velocity

    @torch.no_grad()
    def infer(self, inputs):
        self._ensure_prefix_bank()
        super().infer(inputs)
        scheduler = self.scheduler
        layout = scheduler.layout
        device = scheduler.video_noise_pred.device
        source_rows = layout.source_rows.to(device)
        video_rows = layout.noisy_video_indices.to(device)
        audio_rows = layout.noisy_audio_indices.to(device)
        row_timesteps = scheduler.row_timesteps.to(device)
        video_x0 = self._source_x0(scheduler.video_tokens.to(device), source_rows, video_rows, row_timesteps, scheduler.video_noise_pred)
        audio_x0 = self._source_x0(scheduler.audio_tokens.to(device), source_rows, audio_rows, row_timesteps, scheduler.audio_noise_pred)
        height, width = scheduler.plan.target_latent_size
        frames = int(scheduler.plan.video_noisy_mask.sum())
        scheduler.video_noise_pred = unpatchify_video_tokens(video_x0, frames, height, width)[0]
        scheduler.audio_noise_pred = unpack_audio_tokens(audio_x0, int(scheduler.plan.audio_noisy_mask.sum()))

    def reset_autoregressive_state(self):
        super().reset_autoregressive_state()
        self.pre_infer.clear_text_condition()
        for bank in getattr(self, "_prefix_bank_pool", {}).values():
            bank.clear()
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None:
            scheduler.prefix_banks.clear()
            scheduler.prefix_bank = None

    def to_cpu(self):
        for bank in getattr(self, "_prefix_bank_pool", {}).values():
            bank.clear()
        self._prefix_bank_pool = {}
        self._prefix_raw_workspace = None
        self._prefix_bank_signature = None
        return super().to_cpu()
