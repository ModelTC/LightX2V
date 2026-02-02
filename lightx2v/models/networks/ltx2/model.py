import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.ltx2.infer.offload.transformer_infer import (
    LTX2OffloadTransformerInfer,
)
from lightx2v.models.networks.ltx2.infer.post_infer import LTX2PostInfer
from lightx2v.models.networks.ltx2.infer.pre_infer import LTX2PreInfer
from lightx2v.models.networks.ltx2.infer.transformer_infer import (
    LTX2TransformerInfer,
)
from lightx2v.models.networks.ltx2.weights.post_weights import LTX2PostWeights
from lightx2v.models.networks.ltx2.weights.pre_weights import LTX2PreWeights
from lightx2v.models.networks.ltx2.weights.transformer_weights import (
    LTX2TransformerWeights,
)
from lightx2v.utils.custom_compiler import compiled_method
from lightx2v.utils.envs import *
from lightx2v.utils.utils import *


class LTX2Model(BaseTransformerModel):
    pre_weight_class = LTX2PreWeights
    transformer_weight_class = LTX2TransformerWeights
    post_weight_class = LTX2PostWeights

    def __init__(self, model_path, config, device, lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, None, lora_path, lora_strength)
        if self.config.get("tensor_parallel", False):
            self.use_tp = True
            self.tp_group = self.config.get("device_mesh").get_group(mesh_dim="tensor_p")
            self.tp_rank = dist.get_rank(self.tp_group)
            self.tp_size = dist.get_world_size(self.tp_group)
        else:
            self.tp_group = None
            self.use_tp = False
            self.tp_rank = 0
            self.tp_size = 1

        self.padding_multiple = self.config.get("padding_multiple", 1)

        # Track original video sequence length before padding (for sequence parallel)
        self.original_video_seq_len = None

        # self.model_type = model_type
        self.remove_keys = ["text_embedding_projection", "audio_vae", "vae", "vocoder", "model.diffusion_model.audio_embeddings_connector", "model.diffusion_model.video_embeddings_connector"]
        if self.lazy_load:
            self.remove_keys.extend(["diffusion_model.transformer_blocks."])
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer_class(self):
        self.pre_infer_class = LTX2PreInfer
        self.post_infer_class = LTX2PostInfer
        self.transformer_infer_class = LTX2TransformerInfer if not self.cpu_offload else LTX2OffloadTransformerInfer

    def _should_load_weights(self):
        """Determine if current rank should load weights from disk."""
        if self.config.get("device_mesh") is None:
            # Single GPU mode
            return True
        elif dist.is_initialized():
            if self.use_tp:
                # Multi-GPU mode, only rank 0 loads
                if dist.get_rank() == 0:
                    logger.info(f"Loading weights from {self.model_path}")
                    return True
            else:
                return True
        return False

    def _load_weights_from_rank0(self, weight_dict, is_weight_loader):
        """
        Load and distribute weights from rank 0 to all ranks.

        Only supports tensor parallel mode with CUDA device.
        CPU offload is not supported.
        """
        # CPU offload is not supported
        if self.cpu_offload:
            raise NotImplementedError("_load_weights_from_rank0 does not support CPU offload. Please set cpu_offload=False.")

        logger.info("Loading distributed weights with tensor parallel (CUDA only)")
        global_src_rank = 0

        if is_weight_loader:
            # Rank 0: prepare weights (split for TP)
            processed_weight_dict = {}
            meta_dict = {}

            for key, tensor in weight_dict.items():
                # Only process .weight keys for TP splitting (bias is handled separately)
                if key.endswith(".weight") and self._is_tp_weight(key):
                    # Split weights for TP: create one entry per rank with original key name
                    split_weights = self._split_weight_for_tp(key, tensor, self.tp_size)
                    # Store all split weights temporarily (will be filtered by rank later)
                    for rank_idx in range(self.tp_size):
                        rank_key = f"{key}__tp_rank_{rank_idx}"
                        processed_weight_dict[rank_key] = split_weights[rank_idx]
                        if rank_idx == 0:  # Use rank 0's shape for meta (all ranks have same shape after split)
                            meta_dict[key] = {"shape": split_weights[rank_idx].shape, "dtype": split_weights[rank_idx].dtype, "is_tp": True}

                    # Also handle bias if it exists (for column split weights)
                    bias_key = key.replace(".weight", ".bias")
                    if bias_key in weight_dict and self._get_split_type(key) == "col":
                        # Column split: bias also needs to be split
                        bias_tensor = weight_dict[bias_key]
                        assert bias_tensor.shape[0] % self.tp_size == 0, f"bias dimension ({bias_tensor.shape[0]}) must be divisible by tp_size ({self.tp_size}) for {bias_key}"
                        chunk_size = bias_tensor.shape[0] // self.tp_size
                        for rank_idx in range(self.tp_size):
                            rank_bias_key = f"{bias_key}__tp_rank_{rank_idx}"
                            start_idx = rank_idx * chunk_size
                            end_idx = start_idx + chunk_size
                            processed_weight_dict[rank_bias_key] = bias_tensor[start_idx:end_idx]
                            if rank_idx == 0:
                                meta_dict[bias_key] = {"shape": bias_tensor[start_idx:end_idx].shape, "dtype": bias_tensor.dtype, "is_tp": True}
                        # For row split weights, bias is not split (added after all-reduce)
                else:
                    # Non-TP weights or bias (bias is handled above if it's a TP weight's bias)
                    # Skip bias keys that are already processed above
                    if not (key.endswith(".bias") and key.replace(".bias", ".weight") in processed_weight_dict):
                        processed_weight_dict[key] = tensor
                        meta_dict[key] = {"shape": tensor.shape, "dtype": tensor.dtype, "is_tp": False}

            obj_list = [meta_dict]
            dist.broadcast_object_list(obj_list, src=global_src_rank)
            synced_meta_dict = obj_list[0]
            weight_dict = processed_weight_dict  # Use processed weights
        else:
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=global_src_rank)
            synced_meta_dict = obj_list[0]

        # Allocate tensors on CUDA
        distributed_weight_dict = {}
        for key, meta in synced_meta_dict.items():
            is_tp = meta.get("is_tp", False)
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            if is_tp:
                # TP weight: each rank gets its own slice
                distributed_weight_dict[key] = torch.empty(meta["shape"], dtype=meta["dtype"], device=device)
            else:
                # Non-TP weight: all ranks get full weight
                distributed_weight_dict[key] = torch.empty(meta["shape"], dtype=meta["dtype"], device=device)

        dist.barrier()

        # Distribute weights
        for key in sorted(synced_meta_dict.keys()):
            meta = synced_meta_dict[key]
            is_tp = meta.get("is_tp", False)

            if is_tp:
                # TP weight: rank 0 sends different slices to each rank
                # Use send/recv to ensure each rank gets its own slice
                dist.barrier(group=self.tp_group)

                if is_weight_loader:
                    # Rank 0: send each rank's slice in order
                    for rank_idx in range(self.tp_size):
                        rank_key = f"{key}__tp_rank_{rank_idx}"
                        if rank_key in weight_dict:
                            if rank_idx == self.tp_rank:
                                # Copy to my own buffer
                                distributed_weight_dict[key].copy_(weight_dict[rank_key], non_blocking=True)
                            else:
                                # Send to other ranks
                                dist.send(weight_dict[rank_key].contiguous(), dst=rank_idx, group=self.tp_group)
                else:
                    # Other ranks: receive from rank 0
                    dist.recv(distributed_weight_dict[key], src=global_src_rank, group=self.tp_group)
            else:
                # Non-TP weight: broadcast to all ranks
                if is_weight_loader:
                    distributed_weight_dict[key].copy_(weight_dict[key], non_blocking=True)

                dist.broadcast(distributed_weight_dict[key], src=global_src_rank)

        torch.cuda.synchronize()

        logger.info(f"Weights distributed across {dist.get_world_size()} devices on CUDA")

        return distributed_weight_dict

    def _is_tp_weight(self, key):
        """Check if a weight key needs TP splitting.

        TP weights include:
        - Attention layers: to_q, to_k, to_v, to_out.0, q_norm, k_norm
        - FFN layers: net.0.proj, net.2
        """
        # Generic patterns that apply to all attention and FFN layers
        tp_patterns = [
            ".to_q.",
            ".to_k.",
            ".to_v.",
            ".to_out.0.",
            ".q_norm.",
            ".k_norm.",
            ".net.0.proj.",
            ".net.2.",
        ]
        return any(pattern in key for pattern in tp_patterns)

    def _get_split_type(self, key):
        """Determine the split type for a weight key.

        Returns:
            "col": Column split (to_q, to_k, to_v, net.0.proj)
            "row": Row split (to_out.0, net.2)
            "norm": Norm split (q_norm, k_norm)
            None: No split needed
        """
        if ".q_norm." in key or ".k_norm." in key:
            return "norm"
        elif ".to_q." in key or ".to_k." in key or ".to_v." in key or ".net.0.proj." in key:
            return "col"
        elif ".to_out.0." in key or ".net.2." in key:
            return "row"
        return None

    def _split_weight_for_tp(self, key, weight, tp_size):
        """
        Split a weight tensor for tensor parallel.
        Returns a list of split weights, one for each rank.
        """
        split_type = self._get_split_type(key)
        if split_type is None:
            # Unknown pattern, don't split
            return [weight] * tp_size

        if split_type == "norm":
            # 1D weights (norm weights): [hidden_dim] -> [hidden_dim/tp_size] per rank
            assert weight.dim() == 1, f"Norm weight should be 1D, got {weight.dim()}D for {key}"
            assert weight.shape[0] % tp_size == 0, f"hidden_dim ({weight.shape[0]}) must be divisible by tp_size ({tp_size}) for {key}"
            chunk_size = weight.shape[0] // tp_size
            return [weight[rank_idx * chunk_size : (rank_idx + 1) * chunk_size] for rank_idx in range(tp_size)]

        # 2D weights (linear layer weights)
        assert weight.dim() == 2, f"Linear weight should be 2D, got {weight.dim()}D for {key}"

        # Transpose to [in_dim, out_dim] format for easier splitting
        weight_t = weight.t()  # [in_dim, out_dim]

        if split_type == "col":
            # Column split: [out_dim, in_dim] -> [out_dim/tp_size, in_dim] per rank
            # Split along out_dim dimension
            assert weight_t.shape[1] % tp_size == 0, f"out_dim ({weight_t.shape[1]}) must be divisible by tp_size ({tp_size}) for {key}"
            chunk_size = weight_t.shape[1] // tp_size
            split_weights = []
            for rank_idx in range(tp_size):
                split_weight = weight_t[:, rank_idx * chunk_size : (rank_idx + 1) * chunk_size].t()  # Back to [out_dim/tp_size, in_dim]
                split_weights.append(split_weight)
        else:  # split_type == "row"
            # Row split: [out_dim, in_dim] -> [out_dim, in_dim/tp_size] per rank
            # Split along in_dim dimension
            assert weight_t.shape[0] % tp_size == 0, f"in_dim ({weight_t.shape[0]}) must be divisible by tp_size ({tp_size}) for {key}"
            chunk_size = weight_t.shape[0] // tp_size
            split_weights = []
            for rank_idx in range(tp_size):
                split_weight = weight_t[rank_idx * chunk_size : (rank_idx + 1) * chunk_size, :].t()  # Back to [out_dim, in_dim/tp_size]
                split_weights.append(split_weight)

        return split_weights

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()

    @compiled_method()
    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        self.scheduler.infer_condition = infer_condition
        pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs)

        # Apply sequence parallel pre-processing (only for video)
        if self.config.get("seq_parallel", False):
            pre_infer_out = self._seq_parallel_pre_process(pre_infer_out)

        # Apply tensor parallel pre-processing (split positional embeddings)
        if self.config.get("tensor_parallel", False):
            pre_infer_out = self._tensor_parallel_pre_process(pre_infer_out)

        vx, ax, video_embedded_timestep, audio_embedded_timestep = self.transformer_infer.infer(self.transformer_weights, pre_infer_out)

        # Apply sequence parallel post-processing (only for video)
        if self.config.get("seq_parallel", False):
            vx = self._seq_parallel_post_process(vx, self.original_video_seq_len)
            video_embedded_timestep = self._seq_parallel_post_process(video_embedded_timestep, self.original_video_seq_len)
            # Audio remains global, no gather needed

        vx, ax = self.post_infer.infer(self.post_weight, vx, ax, video_embedded_timestep, audio_embedded_timestep)
        return vx, ax

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        """
        Pre-process for sequence parallel: only split video sequences across ranks.
        Audio remains global (not split) as it has fewer tokens.
        """
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)

        # Only process video args for sequence parallel
        if pre_infer_out.video_args is not None:
            # Split x (latent)
            vx = pre_infer_out.video_args.x
            self.original_video_seq_len = vx.shape[0]  # Record original length before padding
            multiple = world_size * self.padding_multiple
            padding_size = (multiple - (vx.shape[0] % multiple)) % multiple
            if padding_size > 0:
                vx = F.pad(vx, (0, 0, 0, padding_size))
            pre_infer_out.video_args.x = torch.chunk(vx, world_size, dim=0)[cur_rank]

            # Split positional embeddings (cos_freqs, sin_freqs) for video self-attention
            if pre_infer_out.video_args.positional_embeddings is not None:
                v_cos, v_sin = pre_infer_out.video_args.positional_embeddings
                # For 4D tensors: [batch, num_heads, seq_len, head_dim], seq_len is at dim=2
                if v_cos.dim() == 4:
                    seq_dim = 2
                elif v_cos.dim() == 2:
                    seq_dim = 0
                else:
                    seq_dim = 1

                seq_len = v_cos.shape[seq_dim]
                padding_size = (multiple - (seq_len % multiple)) % multiple
                if padding_size > 0:
                    pad_spec = [0, 0] * (v_cos.dim() - seq_dim - 1) + [0, padding_size] + [0, 0] * seq_dim
                    v_cos = F.pad(v_cos, pad_spec)
                    v_sin = F.pad(v_sin, pad_spec)

                pre_infer_out.video_args.positional_embeddings = (torch.chunk(v_cos, world_size, dim=seq_dim)[cur_rank], torch.chunk(v_sin, world_size, dim=seq_dim)[cur_rank])

            # Split cross-attention positional embeddings for cross-modal attention
            if pre_infer_out.video_args.cross_positional_embeddings is not None:
                v_cross_cos, v_cross_sin = pre_infer_out.video_args.cross_positional_embeddings
                if v_cross_cos.dim() == 4:
                    seq_dim = 2
                elif v_cross_cos.dim() == 2:
                    seq_dim = 0
                else:
                    seq_dim = 1

                seq_len = v_cross_cos.shape[seq_dim]
                padding_size = (multiple - (seq_len % multiple)) % multiple
                if padding_size > 0:
                    pad_spec = [0, 0] * (v_cross_cos.dim() - seq_dim - 1) + [0, padding_size] + [0, 0] * seq_dim
                    v_cross_cos = F.pad(v_cross_cos, pad_spec)
                    v_cross_sin = F.pad(v_cross_sin, pad_spec)

                pre_infer_out.video_args.cross_positional_embeddings = (torch.chunk(v_cross_cos, world_size, dim=seq_dim)[cur_rank], torch.chunk(v_cross_sin, world_size, dim=seq_dim)[cur_rank])

            # Split timestep embeddings (sequence-length dependent)
            if pre_infer_out.video_args.timesteps is not None:
                v_timesteps = pre_infer_out.video_args.timesteps
                padding_size = (multiple - (v_timesteps.shape[0] % multiple)) % multiple
                if padding_size > 0:
                    v_timesteps = F.pad(v_timesteps, (0, 0, 0, padding_size))
                pre_infer_out.video_args.timesteps = torch.chunk(v_timesteps, world_size, dim=0)[cur_rank]

            if pre_infer_out.video_args.embedded_timestep is not None:
                v_embedded_timestep = pre_infer_out.video_args.embedded_timestep
                padding_size = (multiple - (v_embedded_timestep.shape[0] % multiple)) % multiple
                if padding_size > 0:
                    v_embedded_timestep = F.pad(v_embedded_timestep, (0, 0, 0, padding_size))
                pre_infer_out.video_args.embedded_timestep = torch.chunk(v_embedded_timestep, world_size, dim=0)[cur_rank]

            if pre_infer_out.video_args.cross_scale_shift_timestep is not None:
                v_cross_ss = pre_infer_out.video_args.cross_scale_shift_timestep
                padding_size = (multiple - (v_cross_ss.shape[0] % multiple)) % multiple
                if padding_size > 0:
                    v_cross_ss = F.pad(v_cross_ss, (0, 0, 0, padding_size))
                pre_infer_out.video_args.cross_scale_shift_timestep = torch.chunk(v_cross_ss, world_size, dim=0)[cur_rank]

            if pre_infer_out.video_args.cross_gate_timestep is not None:
                v_cross_gate = pre_infer_out.video_args.cross_gate_timestep
                padding_size = (multiple - (v_cross_gate.shape[0] % multiple)) % multiple
                if padding_size > 0:
                    v_cross_gate = F.pad(v_cross_gate, (0, 0, 0, padding_size))
                pre_infer_out.video_args.cross_gate_timestep = torch.chunk(v_cross_gate, world_size, dim=0)[cur_rank]

        # Audio remains global - no splitting needed
        # Audio has fewer tokens, so we keep it on all ranks

        return pre_infer_out

    @torch.no_grad()
    def _tensor_parallel_pre_process(self, pre_infer_out):
        """
        Pre-process for tensor parallel: split positional embeddings along head dimension.

        In tensor parallel, QKV projections are split along hidden_dim (which equals num_heads * head_dim),
        so positional embeddings need to be split along the head dimension to match.
        """
        if not self.config.get("tensor_parallel", False):
            return pre_infer_out

        tp_group = self.config.get("device_mesh").get_group(mesh_dim="tensor_p")
        tp_rank = dist.get_rank(tp_group)
        tp_size = dist.get_world_size(tp_group)

        # Get num_heads and head_dim from config
        v_num_heads = self.config.get("num_attention_heads", 32)
        v_head_dim = self.config.get("attention_head_dim", 128)
        a_num_heads = self.config.get("audio_num_attention_heads", 32)
        a_head_dim = self.config.get("audio_attention_head_dim", 64)

        v_num_heads_per_rank = v_num_heads // tp_size
        a_num_heads_per_rank = a_num_heads // tp_size

        def split_pe(pe, num_heads, num_heads_per_rank, tp_rank, tp_size):
            """Split positional embeddings along head dimension."""
            if pe is None:
                return None

            if isinstance(pe, tuple):
                cos_freqs, sin_freqs = pe

                if cos_freqs.dim() == 2 and cos_freqs.shape[0] == num_heads:
                    # Shape: [num_heads, head_dim] - split along num_heads dimension
                    cos_freqs_split = cos_freqs[tp_rank * num_heads_per_rank : (tp_rank + 1) * num_heads_per_rank, :]
                    sin_freqs_split = sin_freqs[tp_rank * num_heads_per_rank : (tp_rank + 1) * num_heads_per_rank, :]
                    return (cos_freqs_split, sin_freqs_split)
                elif cos_freqs.dim() == 4:
                    # Shape: [B, H, T, D] where H=num_heads, D=head_dim//2 (for SPLIT rope type)
                    # In apply_split_rotary_emb, if input is 2D [seq_len, num_heads_per_rank * head_dim]
                    # and PE is 4D [B, H, T, D], it will reshape input to [B, T, H, head_dim] then swapaxes to [B, H, T, head_dim]
                    # So H in PE should match num_heads_per_rank in the input
                    # Therefore, we need to split along H dimension: [B, H, T, D] -> [B, H/tp_size, T, D]
                    assert cos_freqs.shape[1] == num_heads, f"PE head dimension mismatch: cos_freqs.shape[1]={cos_freqs.shape[1]}, num_heads={num_heads}"
                    cos_freqs_split = cos_freqs[:, tp_rank * num_heads_per_rank : (tp_rank + 1) * num_heads_per_rank, :, :]
                    sin_freqs_split = sin_freqs[:, tp_rank * num_heads_per_rank : (tp_rank + 1) * num_heads_per_rank, :, :]
                    return (cos_freqs_split, sin_freqs_split)
                else:
                    # For other shapes, split along last dimension (hidden_dim)
                    head_dim = cos_freqs.shape[-1] // num_heads if cos_freqs.dim() > 1 and cos_freqs.shape[-1] % num_heads == 0 else cos_freqs.shape[-1]
                    hidden_dim_per_rank = num_heads_per_rank * head_dim
                    start_idx = tp_rank * hidden_dim_per_rank
                    end_idx = start_idx + hidden_dim_per_rank
                    cos_freqs_split = cos_freqs[..., start_idx:end_idx]
                    sin_freqs_split = sin_freqs[..., start_idx:end_idx]
                    return (cos_freqs_split, sin_freqs_split)
            else:
                # pe is a single tensor, split along last dimension
                head_dim = pe.shape[-1] // num_heads if pe.dim() > 1 and pe.shape[-1] % num_heads == 0 else pe.shape[-1]
                hidden_dim_per_rank = num_heads_per_rank * head_dim
                start_idx = tp_rank * hidden_dim_per_rank
                end_idx = start_idx + hidden_dim_per_rank
                return pe[..., start_idx:end_idx]

        # Process video args
        if pre_infer_out.video_args is not None:
            # Split positional embeddings for video self-attention
            if pre_infer_out.video_args.positional_embeddings is not None:
                pre_infer_out.video_args.positional_embeddings = split_pe(pre_infer_out.video_args.positional_embeddings, v_num_heads, v_num_heads_per_rank, tp_rank, tp_size)

            # Split cross-attention positional embeddings
            if pre_infer_out.video_args.cross_positional_embeddings is not None:
                pre_infer_out.video_args.cross_positional_embeddings = split_pe(pre_infer_out.video_args.cross_positional_embeddings, v_num_heads, v_num_heads_per_rank, tp_rank, tp_size)

        # Process audio args
        if pre_infer_out.audio_args is not None:
            # Split positional embeddings for audio self-attention
            if pre_infer_out.audio_args.positional_embeddings is not None:
                pre_infer_out.audio_args.positional_embeddings = split_pe(pre_infer_out.audio_args.positional_embeddings, a_num_heads, a_num_heads_per_rank, tp_rank, tp_size)

            # Split cross-attention positional embeddings
            if pre_infer_out.audio_args.cross_positional_embeddings is not None:
                pre_infer_out.audio_args.cross_positional_embeddings = split_pe(pre_infer_out.audio_args.cross_positional_embeddings, a_num_heads, a_num_heads_per_rank, tp_rank, tp_size)

        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, x, original_length=None):
        """
        Post-process for sequence parallel: gather results from all ranks and remove padding.

        Args:
            x: Tensor to gather
            original_length: Original sequence length before padding. If provided, truncate to this length.
        """
        world_size = dist.get_world_size(self.seq_p_group)
        gathered_x = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered_x, x, group=self.seq_p_group)
        combined_output = torch.cat(gathered_x, dim=0)

        # Remove padding to restore original length
        if original_length is not None and combined_output.shape[0] > original_length:
            combined_output = combined_output[:original_length]

        return combined_output

    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0 and "wan2.2_moe" not in self.config["model_cls"]:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.post_weight.to_cuda()

        if self.config["enable_cfg"]:
            if self.config["cfg_parallel"]:
                # ==================== CFG Parallel Processing ====================
                cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
                assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
                cfg_p_rank = dist.get_rank(cfg_p_group)
                if cfg_p_rank == 0:
                    v_noise_pred, a_noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)
                else:
                    v_noise_pred, a_noise_pred = self._infer_cond_uncond(inputs, infer_condition=False)

                v_noise_pred_list = [torch.zeros_like(v_noise_pred) for _ in range(2)]
                a_noise_pred_list = [torch.zeros_like(a_noise_pred) for _ in range(2)]
                dist.all_gather(v_noise_pred_list, v_noise_pred, group=cfg_p_group)
                dist.all_gather(a_noise_pred_list, a_noise_pred, group=cfg_p_group)
                v_noise_pred_cond = v_noise_pred_list[0]  # cfg_p_rank == 0
                v_noise_pred_uncond = v_noise_pred_list[1]  # cfg_p_rank == 1
                a_noise_pred_cond = a_noise_pred_list[0]  # cfg_p_rank == 0
                a_noise_pred_uncond = a_noise_pred_list[1]  # cfg_p_rank == 1
            else:
                # ==================== CFG Processing ====================
                v_noise_pred_cond, a_noise_pred_cond = self._infer_cond_uncond(inputs, infer_condition=True)
                v_noise_pred_uncond, a_noise_pred_uncond = self._infer_cond_uncond(inputs, infer_condition=False)

            self.scheduler.v_noise_pred = v_noise_pred_uncond + self.scheduler.sample_guide_scale * (v_noise_pred_cond - v_noise_pred_uncond)
            self.scheduler.a_noise_pred = a_noise_pred_uncond + self.scheduler.sample_guide_scale * (a_noise_pred_cond - a_noise_pred_uncond)
        else:
            # ==================== No CFG ====================
            v_noise_pred, a_noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)
            self.scheduler.v_noise_pred = v_noise_pred
            self.scheduler.a_noise_pred = a_noise_pred

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1 and "wan2.2_moe" not in self.config["model_cls"]:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.post_weight.to_cpu()
