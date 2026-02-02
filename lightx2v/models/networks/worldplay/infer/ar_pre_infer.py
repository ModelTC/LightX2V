import torch
from einops import repeat

from lightx2v.models.networks.hunyuan_video.infer.module_io import HunyuanVideo15InferModuleOutput
from lightx2v.models.networks.hunyuan_video.infer.pre_infer import HunyuanVideo15PreInfer
from lightx2v_platform.base.global_var import AI_DEVICE


class WorldPlayARPreInfer(HunyuanVideo15PreInfer):
    """
    Pre-inference module for WorldPlay AR (Autoregressive) model.

    Key differences from WorldPlayPreInfer (Distill):
    - No guidance embedding (AR model doesn't use guidance)
    - Supports chunk-based processing for autoregressive generation
    - Handles memory window selection for long video generation

    Extends HunyuanVideo15PreInfer with action conditioning support.
    """

    def __init__(self, config):
        super().__init__(config)
        self.chunk_latent_frames = config.get("chunk_latent_frames", 4)

    @torch.no_grad()
    def infer(self, weights, inputs):
        """
        Run pre-inference with action conditioning for AR model.

        Args:
            weights: WorldPlayPreWeights containing action_weights
            inputs: Dict with text_encoder_output, image_encoder_output,
                   and optionally pose_output

        Returns:
            HunyuanVideo15InferModuleOutput with action-conditioned vec
        """
        latents = self.scheduler.latents
        grid_sizes_t, grid_sizes_h, grid_sizes_w = latents.shape[2:]

        timesteps = self.scheduler.timesteps
        t = timesteps[self.scheduler.step_index]

        # Get text embeddings (no CFG for AR model)
        if self.scheduler.infer_condition:
            txt, text_mask = (inputs["text_encoder_output"]["context"][0], inputs["text_encoder_output"]["context"][1])
        else:
            txt, text_mask = (inputs["text_encoder_output"]["context_null"][0], inputs["text_encoder_output"]["context_null"][1])

        byt5_txt = inputs["text_encoder_output"]["byt5_features"]
        byt5_text_mask = inputs["text_encoder_output"]["byt5_masks"]
        siglip_output = inputs["image_encoder_output"]["siglip_output"]
        siglip_mask = inputs["image_encoder_output"]["siglip_mask"]
        txt = txt.to(torch.bfloat16)

        # Prepare latent input
        if self.config.get("is_sr_running", False):
            if t < 1000 * self.scheduler.noise_scale:
                condition = self.scheduler.zero_condition
            else:
                condition = self.scheduler.condition
            img = x = latent_model_input = torch.concat([latents, condition], dim=1)
        else:
            cond_latents_concat = self.scheduler.cond_latents_concat
            mask_concat = self.scheduler.mask_concat
            img = x = latent_model_input = torch.concat([latents, cond_latents_concat, mask_concat], dim=1)

        img = img.to(torch.bfloat16)

        t_expand = t.repeat(latent_model_input.shape[0])
        # Note: AR model does NOT use guidance_expand

        img = weights.img_in.apply(img)
        img = img.flatten(2).transpose(1, 2)

        # Compute timestep embedding
        t_freq = self.timestep_embedding(t_expand, self.frequency_embedding_size, self.max_period).to(torch.bfloat16)
        vec = weights.time_in_0.apply(t_freq)
        vec = torch.nn.functional.silu(vec)
        vec = weights.time_in_2.apply(vec)

        # Add action conditioning if available
        # Use full frame count (same as distill), not chunk-based
        if hasattr(self.scheduler, "action") and self.scheduler.action is not None:
            action = self.scheduler.action  # Shape: [B, T]
            B, T = action.shape
            H, W = grid_sizes_h, grid_sizes_w

            # Use grid_sizes_t (actual latent frames) for action
            # Action tensor T should match grid_sizes_t
            actual_T = grid_sizes_t
            if T != actual_T:
                # Truncate or pad action to match actual latent frames
                if T > actual_T:
                    action = action[:, :actual_T]
                else:
                    # Pad with last action
                    pad_size = actual_T - T
                    action = torch.cat([action, action[:, -1:].repeat(1, pad_size)], dim=1)
                T = actual_T

            # Flatten action for per-frame embedding: [B, T] -> [(B*T),]
            action_flat = action.reshape(-1).float()

            # Compute per-frame action embeddings: [(B*T), C]
            action_emb = self._compute_action_embedding(weights.action_weights, action_flat)

            # Expand vec for each frame
            vec_expanded = vec.repeat_interleave(T, dim=0)
            vec_expanded = vec_expanded + action_emb

            # Broadcast to all spatial tokens
            vec = repeat(vec_expanded, "(B T) C -> B (T H W) C", B=B, T=T, H=H, W=W)

            self.scheduler.vec_is_per_token = True
        else:
            self.scheduler.vec_is_per_token = False

        # Process text embeddings
        t_freq = self.timestep_embedding(t_expand, self.frequency_embedding_size, self.max_period).to(torch.bfloat16)
        timestep_aware_representations = weights.txt_in_t_embedder_0.apply(t_freq)
        timestep_aware_representations = torch.nn.functional.silu(timestep_aware_representations)
        timestep_aware_representations = weights.txt_in_t_embedder_2.apply(timestep_aware_representations)

        mask_float = text_mask.float().unsqueeze(-1)
        context_aware_representations = (txt * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        context_aware_representations = context_aware_representations.to(torch.bfloat16)
        context_aware_representations = weights.txt_in_c_embedder_0.apply(context_aware_representations)
        context_aware_representations = torch.nn.functional.silu(context_aware_representations)
        context_aware_representations = weights.txt_in_c_embedder_2.apply(context_aware_representations)

        c = timestep_aware_representations + context_aware_representations
        out = weights.txt_in_input_embedder.apply(txt[0].to(torch.bfloat16))
        txt = self.run_individual_token_refiner(weights, out, text_mask, c)

        txt = txt.unsqueeze(0)
        txt = txt + weights.cond_type_embedding.apply(torch.zeros_like(txt[:, :, 0], device=txt.device, dtype=torch.long))
        byt5_txt = byt5_txt + weights.cond_type_embedding.apply(torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long))
        txt, text_mask = self.reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True)

        siglip_output = siglip_output + weights.cond_type_embedding.apply(2 * torch.ones_like(siglip_output[:, :, 0], dtype=torch.long, device=AI_DEVICE))
        txt, text_mask = self.reorder_txt_token(siglip_output, txt, siglip_mask, text_mask)
        txt = txt[:, : text_mask.sum(), :]

        # Apply silu only if NOT per-token
        if not self.scheduler.vec_is_per_token:
            vec = torch.nn.functional.silu(vec)

        return HunyuanVideo15InferModuleOutput(
            img=img.contiguous(),
            txt=txt.contiguous(),
            vec=vec.contiguous(),
            grid_sizes=(grid_sizes_t, grid_sizes_h, grid_sizes_w),
        )

    def _compute_action_embedding(self, action_weights, action):
        """
        Compute action embedding using TimestepEmbedder-style MLP.

        Args:
            action_weights: WorldPlayActionWeights
            action: Discrete action labels [N] with values 0-80

        Returns:
            Action embedding tensor [N, hidden_size]
        """
        action_freq = self.timestep_embedding(action, self.frequency_embedding_size, self.max_period).to(torch.bfloat16)

        action_emb = action_weights.action_in_0.apply(action_freq)
        action_emb = torch.nn.functional.silu(action_emb)
        action_emb = action_weights.action_in_2.apply(action_emb)

        return action_emb

    @torch.no_grad()
    def infer_chunk(self, weights, inputs, chunk_idx, chunk_latents):
        """
        Run pre-inference for a specific chunk in AR generation.

        Args:
            weights: WorldPlayPreWeights
            inputs: Dict with encoder outputs
            chunk_idx: Current chunk index
            chunk_latents: Latent tensor for this chunk

        Returns:
            HunyuanVideo15InferModuleOutput for this chunk
        """
        # Store chunk index in scheduler
        self.scheduler.chunk_idx = chunk_idx

        # Temporarily replace latents with chunk latents
        original_latents = self.scheduler.latents
        self.scheduler.latents = chunk_latents

        # Run standard inference
        output = self.infer(weights, inputs)

        # Restore original latents
        self.scheduler.latents = original_latents

        return output

    # ========== AR-specific inference methods ==========

    @torch.no_grad()
    def infer_txt_only(self, weights, inputs):
        """
        Process only text embeddings for text KV caching.
        Called once at the beginning of AR generation.

        This method computes text embeddings without processing image latents,
        using a fixed timestep (t=0) for the timestep embedding.

        Args:
            weights: WorldPlayPreWeights containing text processing weights
            inputs: Dict with text_encoder_output, image_encoder_output

        Returns:
            HunyuanVideo15InferModuleOutput with txt and vec (img=None)
        """
        # Get text embeddings (no CFG for AR model)
        if self.scheduler.infer_condition:
            txt, text_mask = (inputs["text_encoder_output"]["context"][0], inputs["text_encoder_output"]["context"][1])
        else:
            txt, text_mask = (inputs["text_encoder_output"]["context_null"][0], inputs["text_encoder_output"]["context_null"][1])

        byt5_txt = inputs["text_encoder_output"]["byt5_features"]
        byt5_text_mask = inputs["text_encoder_output"]["byt5_masks"]
        siglip_output = inputs["image_encoder_output"]["siglip_output"]
        siglip_mask = inputs["image_encoder_output"]["siglip_mask"]
        txt = txt.to(torch.bfloat16)

        # Use fixed timestep t=0 for text KV caching
        t = torch.tensor([0.0], device=AI_DEVICE)
        t_expand = t.repeat(1)  # Batch size 1

        # Compute timestep embedding
        t_freq = self.timestep_embedding(t_expand, self.frequency_embedding_size, self.max_period).to(torch.bfloat16)
        vec = weights.time_in_0.apply(t_freq)
        vec = torch.nn.functional.silu(vec)
        vec = weights.time_in_2.apply(vec)

        # Process text embeddings
        timestep_aware_representations = weights.txt_in_t_embedder_0.apply(t_freq)
        timestep_aware_representations = torch.nn.functional.silu(timestep_aware_representations)
        timestep_aware_representations = weights.txt_in_t_embedder_2.apply(timestep_aware_representations)

        mask_float = text_mask.float().unsqueeze(-1)
        context_aware_representations = (txt * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        context_aware_representations = context_aware_representations.to(torch.bfloat16)
        context_aware_representations = weights.txt_in_c_embedder_0.apply(context_aware_representations)
        context_aware_representations = torch.nn.functional.silu(context_aware_representations)
        context_aware_representations = weights.txt_in_c_embedder_2.apply(context_aware_representations)

        c = timestep_aware_representations + context_aware_representations
        out = weights.txt_in_input_embedder.apply(txt[0].to(torch.bfloat16))
        txt = self.run_individual_token_refiner(weights, out, text_mask, c)

        txt = txt.unsqueeze(0)
        txt = txt + weights.cond_type_embedding.apply(torch.zeros_like(txt[:, :, 0], device=txt.device, dtype=torch.long))
        byt5_txt = byt5_txt + weights.cond_type_embedding.apply(torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long))
        txt, text_mask = self.reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True)

        siglip_output = siglip_output + weights.cond_type_embedding.apply(2 * torch.ones_like(siglip_output[:, :, 0], dtype=torch.long, device=AI_DEVICE))
        txt, text_mask = self.reorder_txt_token(siglip_output, txt, siglip_mask, text_mask)
        txt = txt[:, : text_mask.sum(), :]

        # Apply silu to vec (no per-token for text-only)
        vec = torch.nn.functional.silu(vec)

        # Set flag for text-only mode
        self.scheduler.vec_is_per_token = False

        return HunyuanVideo15InferModuleOutput(
            img=None,  # No image for text-only inference
            txt=txt.contiguous(),
            vec=vec.contiguous(),
            grid_sizes=None,  # No grid sizes for text-only
        )
