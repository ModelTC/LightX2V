import logging
import math
from dataclasses import replace
from enum import Enum

import torch
from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.model_protocol import LTXModelProtocol
from ltx_core.model.transformer.adaln import AdaLayerNormSingle, adaln_embedding_coefficient
from ltx_core.model.transformer.attention import attention_label
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import (
    DEFAULT_TRANSFORMER_OPS,
    BasicAVTransformerBlock,
    TransformerConfig,
    TransformerOpsConfig,
)
from ltx_core.model.transformer.transformer_args import (
    BlockPerturbationsProcessor,
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from ltx_core.utils import to_denoised
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

logger = logging.getLogger(__name__)


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(torch.nn.Module):
    """
    LTX model transformer implementation.
    This class implements the transformer blocks for the LTX model.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        caption_projection: torch.nn.Module | None = None,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
    ):
        super().__init__()
        # Log the attention backends this transformer is built with. Reading the resolved
        # ``label`` off the ops reports whatever was selected -- AUTOMATIC, an explicit pin
        # (PYTORCH/XFORMERS/FA3/FA4/SDPA_*), or a directly supplied callable -- so this is the
        # single source of truth for which kernel a build uses. Fires once per build.
        logger.info(
            "Building transformer with attention backends -- self: %s, masked: %s",
            attention_label(ops.attention_ops.attention_function),
            attention_label(ops.attention_ops.masked_attention_function),
        )
        self._enable_gradient_checkpointing = False
        self.cross_attention_adaln = cross_attention_adaln
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        self.num_frame_per_block = 1
        self.local_attn_size = -1
        self.sink_size = 0
        self.defer_kv_cache_updates = False
        self.detach_kv_cache_updates = False
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_eps=norm_eps,
                caption_projection=caption_projection,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                norm_eps=norm_eps,
                caption_projection=audio_caption_projection,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            ops=ops,
            apply_gated_attention=apply_gated_attention,
        )
        # Hook for per-block input prep. Compile transforms in `compiling.py`
        # wrap (not replace) this with a processor that also marks the seq dim
        # dynamic, so any caller customisation here is preserved as the inner.
        self.block_input_processor = BlockPerturbationsProcessor()

    @property
    def _adaln_embedding_coefficient(self) -> int:
        return adaln_embedding_coefficient(self.cross_attention_adaln)

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        if caption_projection is not None:
            self.caption_projection = caption_projection

        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=self._adaln_embedding_coefficient)

        self.prompt_adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        if caption_projection is not None:
            self.audio_caption_projection = caption_projection

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=self._adaln_embedding_coefficient,
        )

        self.audio_prompt_adaln_single = AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        ops: TransformerOpsConfig,
        apply_gated_attention: bool,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    ops=ops,
                )
                for _ in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None,
        kv_cache: dict | None = None,
        video_current_start: int = 0,
        audio_current_start: int = 0,
        defer_cache_updates: bool | None = None,
        detach_cache_updates: bool | None = None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        """Process transformer blocks for LTXAV.
        Per-block perturbation masks are precomputed here and attached to each
        modality's ``TransformerArgs`` so the block forward has no per-block
        identity to specialise on — all blocks share a single Dynamo cache slot.
        """
        if perturbations is None:
            batch_size = (video or audio).x.shape[0]
            perturbations = BatchedPerturbationConfig.empty(batch_size)
        if defer_cache_updates is None:
            defer_cache_updates = bool(self.defer_kv_cache_updates)
        defer_cache_updates = bool(defer_cache_updates and torch.is_grad_enabled())
        if detach_cache_updates is None:
            detach_cache_updates = bool(self.detach_kv_cache_updates)
        detach_cache_updates = bool(detach_cache_updates and torch.is_grad_enabled() and not defer_cache_updates)

        cache_update_infos = []

        for block_idx, block in enumerate(self.transformer_blocks):
            if video is not None:
                video = self.block_input_processor(
                    video,
                    perturbations,
                    block_idx,
                    self_attn_type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                    cross_attn_type=PerturbationType.SKIP_A2V_CROSS_ATTN,
                )
            if audio is not None:
                audio = self.block_input_processor(
                    audio,
                    perturbations,
                    block_idx,
                    self_attn_type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                    cross_attn_type=PerturbationType.SKIP_V2A_CROSS_ATTN,
                )

            if self._enable_gradient_checkpointing and self.training:
                checkpoint_tensors = []
                video_template = None
                audio_template = None
                if video is not None:
                    checkpoint_tensors.append(video.x)
                    video_template = replace(video, x=None)
                if audio is not None:
                    checkpoint_tensors.append(audio.x)
                    audio_template = replace(audio, x=None)

                video_kv_cache = kv_cache["video"][block_idx] if kv_cache is not None and kv_cache.get("video") is not None else None
                audio_kv_cache = kv_cache["audio"][block_idx] if kv_cache is not None and kv_cache.get("audio") is not None else None

                def block_forward(
                    *tensor_args,
                    block=block,
                    video_template=video_template,
                    audio_template=audio_template,
                    video_kv_cache=video_kv_cache,
                    audio_kv_cache=audio_kv_cache,
                    video_current_start=video_current_start,
                    audio_current_start=audio_current_start,
                    defer_cache_updates=defer_cache_updates,
                    detach_cache_updates=detach_cache_updates,
                ):
                    arg_idx = 0
                    video_args = video_template
                    audio_args = audio_template
                    if video_args is not None:
                        video_args = replace(video_args, x=tensor_args[arg_idx])
                        arg_idx += 1
                    if audio_args is not None:
                        audio_args = replace(audio_args, x=tensor_args[arg_idx])
                    return block(
                        video=video_args,
                        audio=audio_args,
                        video_kv_cache=video_kv_cache,
                        audio_kv_cache=audio_kv_cache,
                        video_current_start=video_current_start,
                        audio_current_start=audio_current_start,
                        defer_cache_update=defer_cache_updates,
                        detach_cache_update=detach_cache_updates,
                    )

                has_block_mask = any(isinstance(mask, BlockMask) for args in (video, audio) if args is not None for mask in (args.self_attention_mask, args.cross_attention_mask))
                with torch.utils.checkpoint.set_checkpoint_early_stop(not has_block_mask):
                    block_out = torch.utils.checkpoint.checkpoint(
                        block_forward,
                        *checkpoint_tensors,
                        use_reentrant=False,
                    )
            else:
                block_out = block(
                    video=video,
                    audio=audio,
                    video_kv_cache=kv_cache["video"][block_idx] if kv_cache is not None and kv_cache.get("video") is not None else None,
                    audio_kv_cache=kv_cache["audio"][block_idx] if kv_cache is not None and kv_cache.get("audio") is not None else None,
                    video_current_start=video_current_start,
                    audio_current_start=audio_current_start,
                    defer_cache_update=defer_cache_updates,
                    detach_cache_update=detach_cache_updates,
                )

            if isinstance(block_out, tuple) and len(block_out) == 3:
                video, audio, cache_update_info = block_out
                cache_update_infos.append((block_idx, cache_update_info))
            else:
                video, audio = block_out

        if kv_cache is not None and defer_cache_updates and cache_update_infos:
            self._apply_cache_updates(kv_cache, cache_update_infos)
        return video, audio

    @staticmethod
    def _apply_cache_updates(kv_cache, cache_update_infos):
        for block_index, update_info in cache_update_infos:
            for modality_name in ("video", "audio"):
                if update_info.get(modality_name) is None or kv_cache.get(modality_name) is None:
                    continue
                current_end, local_end_index, cache_update_info = update_info[modality_name]
                cache = kv_cache[modality_name][block_index]
                if cache_update_info is not None:
                    if cache_update_info["action"] == "roll_and_insert":
                        sink_tokens = cache_update_info["sink_tokens"]
                        num_rolled_tokens = cache_update_info["num_rolled_tokens"]
                        num_evicted_tokens = cache_update_info["num_evicted_tokens"]
                        if num_rolled_tokens > 0:
                            cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = cache["k"][
                                :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                            ].clone()
                            cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = cache["v"][
                                :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                            ].clone()
                    cache["k"][:, cache_update_info["local_start_index"] : cache_update_info["local_end_index"]] = cache_update_info["new_k"]
                    cache["v"][:, cache_update_info["local_start_index"] : cache_update_info["local_end_index"]] = cache_update_info["new_v"]
                cache["global_end_index"].fill_(current_end)
                cache["local_end_index"].fill_(local_end_index)

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        kv_cache: dict | None = None,
        current_start: int = 0,
        video_current_start: int | None = None,
        audio_current_start: int | None = None,
        defer_cache_updates: bool | None = None,
        detach_cache_updates: bool | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Forward pass for LTX models.
        Returns:
            Processed output tensors
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video_args = self.video_args_preprocessor.prepare(video, audio) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None
        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
            kv_cache=kv_cache,
            video_current_start=current_start if video_current_start is None else video_current_start,
            audio_current_start=current_start if audio_current_start is None else audio_current_start,
            defer_cache_updates=defer_cache_updates,
            detach_cache_updates=detach_cache_updates,
        )

        # Process output
        vx = self._process_output(self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep) if video_out is not None else None
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax

    def _forward_inference(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None = None,
        kv_cache: dict | None = None,
        current_start: int = 0,
        video_current_start: int | None = None,
        audio_current_start: int | None = None,
        defer_cache_updates: bool | None = None,
        detach_cache_updates: bool | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.forward(
            video=video,
            audio=audio,
            perturbations=perturbations,
            kv_cache=kv_cache,
            current_start=current_start,
            video_current_start=video_current_start,
            audio_current_start=audio_current_start,
            defer_cache_updates=defer_cache_updates,
            detach_cache_updates=detach_cache_updates,
        )

    @staticmethod
    def prepare_blockwise_causal_mask_from_block_ids(
        device: torch.device | str,
        block_ids: torch.Tensor,
        *,
        local_attn_blocks: int = -1,
        batch_size: int | None = None,
    ) -> BlockMask:
        block_ids = block_ids.to(device=device, dtype=torch.long)
        total_length = int(block_ids.numel())
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        padded_block_ids = torch.cat([block_ids, block_ids.new_zeros(padded_length)], dim=0)

        def attention_mask(b, h, q_idx, kv_idx):
            is_real_q = q_idx < total_length
            is_real_k = kv_idx < total_length
            q_block = padded_block_ids[q_idx]
            kv_block = padded_block_ids[kv_idx]
            if local_attn_blocks == -1:
                visible = kv_block <= q_block
            else:
                visible = (kv_block <= q_block) & (kv_block >= (q_block - local_attn_blocks + 1))
            return (q_idx == kv_idx) | (is_real_q & is_real_k & visible)

        return create_block_mask(
            attention_mask,
            B=batch_size,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=True,
            device=device,
        )

    @staticmethod
    def prepare_teacher_forcing_mask_from_block_ids(
        device: torch.device | str,
        block_ids: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> BlockMask:
        block_ids = block_ids.to(device=device, dtype=torch.long)
        clean_length = int(block_ids.numel())
        total_length = clean_length * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        all_block_ids = torch.cat([block_ids, block_ids, block_ids.new_zeros(padded_length)], dim=0)

        def attention_mask(b, h, q_idx, kv_idx):
            is_real_q = q_idx < total_length
            is_real_k = kv_idx < total_length
            q_is_noisy = q_idx >= clean_length
            kv_is_noisy = kv_idx >= clean_length
            q_block = all_block_ids[q_idx]
            kv_block = all_block_ids[kv_idx]
            clean_to_clean = (~q_is_noisy) & (~kv_is_noisy) & (kv_block <= q_block)
            noisy_to_clean = q_is_noisy & (~kv_is_noisy) & (kv_block < q_block)
            noisy_to_noisy = q_is_noisy & kv_is_noisy & (kv_block == q_block)
            return (q_idx == kv_idx) | (is_real_q & is_real_k & (clean_to_clean | noisy_to_clean | noisy_to_noisy))

        return create_block_mask(
            attention_mask,
            B=batch_size,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=True,
            device=device,
        )

    @staticmethod
    def prepare_teacher_forcing_cross_mask_from_block_ids(
        device: torch.device | str,
        query_block_ids: torch.Tensor,
        key_block_ids: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> BlockMask:
        query_block_ids = query_block_ids.to(device=device, dtype=torch.long)
        key_block_ids = key_block_ids.to(device=device, dtype=torch.long)
        clean_q_length = int(query_block_ids.numel())
        clean_k_length = int(key_block_ids.numel())
        q_length = clean_q_length * 2
        kv_length = clean_k_length * 2
        q_padded_length = math.ceil(q_length / 128) * 128 - q_length
        kv_padded_length = math.ceil(kv_length / 128) * 128 - kv_length
        all_query_block_ids = torch.cat([query_block_ids, query_block_ids, query_block_ids.new_zeros(q_padded_length)], dim=0)
        all_key_block_ids = torch.cat([key_block_ids, key_block_ids, key_block_ids.new_zeros(kv_padded_length)], dim=0)

        def attention_mask(b, h, q_idx, kv_idx):
            is_real_q = q_idx < q_length
            is_real_k = kv_idx < kv_length
            q_is_noisy = q_idx >= clean_q_length
            kv_is_noisy = kv_idx >= clean_k_length
            q_block = all_query_block_ids[q_idx]
            kv_block = all_key_block_ids[kv_idx]
            same_stream = (q_is_noisy & kv_is_noisy) | ((~q_is_noisy) & (~kv_is_noisy))
            same_block = q_block == kv_block
            # AR inference keeps independent video/audio self-attention caches;
            # A2V and V2A attention only exchange the current chunk. Restricting
            # cross attention to the same stream and block makes parallel TF
            # expose exactly that cross-modal context.
            return is_real_q & is_real_k & same_stream & same_block

        return create_block_mask(
            attention_mask,
            B=batch_size,
            H=None,
            Q_LEN=q_length + q_padded_length,
            KV_LEN=kv_length + kv_padded_length,
            _compile=True,
            device=device,
        )


class LegacyX0Model(torch.nn.Module):
    """
    Legacy X0 model implementation.
    Returns fully denoised output based on the velocities produced by the base model.
    """

    def __init__(self, velocity_model: LTXModelProtocol):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        sigma: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class X0Model(torch.nn.Module):
    """
    X0 model implementation.
    Returns fully denoised outputs based on the velocities produced by the base model.
    Applies scaled denoising to the video and audio according to the timesteps = sigma * denoising_mask.
    """

    def __init__(self, velocity_model: LTXModelProtocol):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return denoised_video, denoised_audio
