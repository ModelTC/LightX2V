import torch
from einops import repeat

from lightx2v.models.networks.hunyuan_video.infer.module_io import HunyuanVideo15InferModuleOutput
from lightx2v.models.networks.hunyuan_video.infer.pre_infer import HunyuanVideo15PreInfer
from lightx2v_platform.base.global_var import AI_DEVICE


class WorldPlayPreInfer(HunyuanVideo15PreInfer):
    """
    Pre-inference module for WorldPlay model.

    Extends HunyuanVideo15PreInfer with action conditioning support.
    """

    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def infer(self, weights, inputs):
        """
        Run pre-inference with action conditioning.

        Args:
            weights: WorldPlayPreWeights containing action_weights
            inputs: Dict with text_encoder_output, image_encoder_output, and optionally pose_output

        Returns:
            HunyuanVideo15InferModuleOutput with action-conditioned vec
        """
        latents = self.scheduler.latents
        grid_sizes_t, grid_sizes_h, grid_sizes_w = latents.shape[2:]

        timesteps = self.scheduler.timesteps
        t = timesteps[self.scheduler.step_index]

        if self.scheduler.infer_condition:
            txt, text_mask = inputs["text_encoder_output"]["context"][0], inputs["text_encoder_output"]["context"][1]
        else:
            txt, text_mask = inputs["text_encoder_output"]["context_null"][0], inputs["text_encoder_output"]["context_null"][1]

        byt5_txt, byt5_text_mask = inputs["text_encoder_output"]["byt5_features"], inputs["text_encoder_output"]["byt5_masks"]
        siglip_output, siglip_mask = inputs["image_encoder_output"]["siglip_output"], inputs["image_encoder_output"]["siglip_mask"]
        txt = txt.to(torch.bfloat16)

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

        img = weights.img_in.apply(img)
        img = img.flatten(2).transpose(1, 2)

        # Check if we have per-frame timestep from BI runner
        timestep_input = getattr(self.scheduler, "timestep_input", None)

        # Always define t_expand for text branch timestep embedding (used later)
        t_expand = t.repeat(latent_model_input.shape[0])

        # Add action conditioning if available
        # HY-WorldPlay computes action embedding per-frame and broadcasts to spatial dimensions
        # Each frame gets its own action embedding, then broadcast to all spatial tokens in that frame
        if hasattr(self.scheduler, "action") and self.scheduler.action is not None:
            action = self.scheduler.action  # Shape: [B, T]
            B, T = action.shape
            H, W = grid_sizes_h, grid_sizes_w

            # For BI model with per-frame timestep
            if timestep_input is not None and timestep_input.numel() > 1:
                # timestep_input is [T] with per-frame timesteps
                # Compute per-frame time embeddings
                t_freq = self.timestep_embedding(timestep_input, self.frequency_embedding_size, self.max_period).to(torch.bfloat16)
                vec_per_frame = weights.time_in_0.apply(t_freq)  # [T, C]
                vec_per_frame = torch.nn.functional.silu(vec_per_frame)
                vec_per_frame = weights.time_in_2.apply(vec_per_frame)  # [T, C]
            else:
                # Global timestep - expand to per-frame
                t_expand = t.repeat(latent_model_input.shape[0])
                t_freq = self.timestep_embedding(t_expand, self.frequency_embedding_size, self.max_period).to(torch.bfloat16)
                vec = weights.time_in_0.apply(t_freq)
                vec = torch.nn.functional.silu(vec)
                vec = weights.time_in_2.apply(vec)
                # Expand to per-frame: [B, C] -> [T, C]
                vec_per_frame = vec.repeat(T, 1)  # [T, C]

            # Flatten action for per-frame embedding: [B, T] -> [T,] (assuming B=1)
            action_flat = action.reshape(-1).float()  # [T]

            # Compute per-frame action embeddings: [T, C]
            action_emb = self._compute_action_embedding(weights.action_weights, action_flat)

            # Add per-frame action embedding to per-frame time embedding
            vec_per_frame = vec_per_frame + action_emb  # [T, C]

            # Broadcast to all spatial tokens: [T, C] -> [B, (T*H*W), C]
            # Use einops repeat like HY-WorldPlay: repeat(vec, "T C -> B (T H W) C", B=B, H=H, W=W)
            vec = repeat(vec_per_frame, "T C -> B (T H W) C", B=B, T=T, H=H, W=W)

            # Store flag in scheduler so transformer_infer can access it
            self.scheduler.vec_is_per_token = True
        else:
            # No action conditioning - use global timestep
            t_expand = t.repeat(latent_model_input.shape[0])
            t_freq = self.timestep_embedding(t_expand, self.frequency_embedding_size, self.max_period).to(torch.bfloat16)
            vec = weights.time_in_0.apply(t_freq)
            vec = torch.nn.functional.silu(vec)
            vec = weights.time_in_2.apply(vec)
            self.scheduler.vec_is_per_token = False

        if self.config.get("is_sr_running", False):
            use_meanflow = self.config.get("video_super_resolution", {}).get("use_meanflow", False)
            if use_meanflow:
                if self.scheduler.step_index == len(timesteps) - 1:
                    timesteps_r = torch.tensor([0.0], device=latent_model_input.device)
                else:
                    timesteps_r = timesteps[self.scheduler.step_index + 1]
                timesteps_r = timesteps_r.repeat(latent_model_input.shape[0])
            else:
                timesteps_r = None

            if timesteps_r is not None:
                t_freq = self.timestep_embedding(timesteps_r, self.frequency_embedding_size, self.max_period).to(torch.bfloat16)
                vec_res = weights.time_r_in_0.apply(t_freq)
                vec_res = torch.nn.functional.silu(vec_res)
                vec_res = weights.time_r_in_2.apply(vec_res)
                vec = vec + vec_res

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

        # Apply silu to vec only if NOT per-token (per-token vec already has silu applied in embedders)
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
            action: Discrete action labels [N] with values 0-80 (can be any batch size)

        Returns:
            Action embedding tensor [N, hidden_size]
        """
        # Convert discrete action to embedding using sinusoidal embedding
        action_freq = self.timestep_embedding(action, self.frequency_embedding_size, self.max_period).to(torch.bfloat16)

        # Pass through MLP
        action_emb = action_weights.action_in_0.apply(action_freq)
        action_emb = torch.nn.functional.silu(action_emb)
        action_emb = action_weights.action_in_2.apply(action_emb)

        return action_emb
