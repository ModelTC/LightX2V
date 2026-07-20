"""Sync/Async pipeline driver for Flux2 PipeFusion.

Orchestrates the denoising loop across pipeline stages:
- **Sync pipeline** (warmup): each timestep, all stages process the full latent
  sequentially (stage 0 -> stage 1 -> ... -> last stage).
- **Async pipeline** (main loop): each timestep, the latent is split into
  patches; stages process different patches concurrently, overlapping compute
  and P2P communication.
"""

import torch
import torch.distributed as dist

from lightx2v.common.distributed import (
    PipelineComm,
    get_pipeline_parallel_world_size,
    get_pipeline_runtime_state,
    get_pp_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)


class Flux2PipelineDriver:
    """Drives the PipeFusion denoising loop for Flux2."""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.state = get_pipeline_runtime_state()
        self.pp_comm = PipelineComm(get_pp_group())
        self._is_first = is_pipeline_first_stage()
        self._is_last = is_pipeline_last_stage()
        self._pp_world_size = get_pipeline_parallel_world_size()
        self._dtype = config.get("dtype", torch.bfloat16)
        if isinstance(self._dtype, str):
            self._dtype = getattr(torch, self._dtype)

    # ==================================================================
    # Public entry point
    # ==================================================================

    def run_pipeline(self, latents, prompt_embeds, text_ids, latent_image_ids, timesteps, scheduler, do_cfg=False, negative_prompt_embeds=None, negative_text_ids=None):
        """Run the full denoising loop with PipeFusion.

        Returns final latents on the last stage, ``None`` on other stages.
        """
        warmup_steps = self.state.warmup_steps

        if self._pp_world_size > 1 and len(timesteps) > warmup_steps:
            latents = self._sync_pipeline(
                latents,
                prompt_embeds,
                text_ids,
                latent_image_ids,
                timesteps[:warmup_steps],
                scheduler,
                do_cfg=do_cfg,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
            )
            latents = self._async_pipeline(
                latents,
                prompt_embeds,
                text_ids,
                latent_image_ids,
                timesteps[warmup_steps:],
                scheduler,
                do_cfg=do_cfg,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
            )
        else:
            latents = self._sync_pipeline(
                latents,
                prompt_embeds,
                text_ids,
                latent_image_ids,
                timesteps,
                scheduler,
                do_cfg=do_cfg,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
            )
        return latents

    # ==================================================================
    # Sync pipeline (warmup)
    # ==================================================================

    def _sync_pipeline(self, latents, prompt_embeds, text_ids, latent_image_ids, timesteps, scheduler, do_cfg=False, negative_prompt_embeds=None, negative_text_ids=None):
        self.state.set_patched_mode(patch_mode=False)

        for step_idx, t in enumerate(timesteps):
            scheduler.step_index = step_idx
            scheduler.step_pre(step_idx)

            if do_cfg:
                # Conditional pass
                cond_result = self._sync_pass(
                    latents,
                    prompt_embeds,
                    text_ids,
                    latent_image_ids,
                    t,
                    scheduler,
                )
                # Unconditional pass
                uncond_result = self._sync_pass(
                    latents,
                    negative_prompt_embeds,
                    negative_text_ids or text_ids,
                    latent_image_ids,
                    t,
                    scheduler,
                )
                if self._is_last:
                    noise_pred_cond = cond_result
                    noise_pred_uncond = uncond_result
                    guidance_scale = self.config.get("sample_guide_scale", 1.0)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    scheduler.noise_pred = noise_pred
                    scheduler.latents = latents
                    scheduler.step_post()
                    latents = scheduler.latents
            else:
                noise_pred = self._sync_pass(
                    latents,
                    prompt_embeds,
                    text_ids,
                    latent_image_ids,
                    t,
                    scheduler,
                )
                if self._is_last:
                    scheduler.noise_pred = noise_pred
                    scheduler.latents = latents
                    scheduler.step_post()
                    latents = scheduler.latents

            # P2P: last stage sends updated latents to first stage (circular)
            # Only rank 0 needs updated latents (for x_embedder in next step).
            # Ranks 1-6 don't participate — no global sync barrier.
            if self._pp_world_size > 1:
                if self._is_last:
                    # Last stage sends to first stage
                    dist.send(latents.contiguous(), dst=self.pp_comm.ranks[0], group=self.pp_comm.pp_group)
                elif self._is_first:
                    # First stage receives from last stage
                    latents = torch.empty_like(latents)
                    dist.recv(latents, src=self.pp_comm.ranks[-1], group=self.pp_comm.pp_group)
                # Ranks 1-6: no op (don't need updated latents)

        return latents

    def _sync_pass(self, latents, prompt_embeds, text_ids, latent_image_ids, t, scheduler):
        """Single sync forward pass through all stages.

        Returns ``noise_pred`` on last stage, ``None`` on other stages.
        P2P always carries separate (image, text) streams.
        """
        # NOTE: do NOT clear KV cache here. Sync mode populates per-patch
        # slots so async mode can use them as "stale" KV for global attention.

        if self._is_first:
            # First stage: run pre_infer + blocks
            pre_infer_out = self.model.pre_infer.infer(
                weights=self.model.pre_weight,
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
            )
            hidden_states, enc_hidden, num_txt = self.model.transformer_infer.infer(self.model.transformer_weights, pre_infer_out)

            if self._is_last:
                return self._run_post_infer(hidden_states, enc_hidden, num_txt, pre_infer_out.timestep)
            else:
                # Always send both latent and encoder_hidden_state (skip_shape
                # to avoid .item() CPU-GPU sync on receiver)
                self.pp_comm.pipeline_send(hidden_states, name="latent", skip_shape=True)
                self.pp_comm.pipeline_send(enc_hidden, name="encoder_hidden_state", skip_shape=True)
                return None
        else:
            # Non-first stage: always receive both streams
            # Pass pre-computed shapes to avoid .item() CPU-GPU sync
            inner_dim = self.config.get("num_attention_heads", 24) * self.config.get("attention_head_dim", 64)
            if latent_image_ids.ndim == 3:
                img_len = latent_image_ids.shape[1]
            else:
                img_len = latent_image_ids.shape[0]
            if prompt_embeds is not None:
                txt_len = prompt_embeds.shape[1] if prompt_embeds.ndim == 3 else prompt_embeds.shape[0]
            else:
                txt_len = 0
            hidden_states = self.pp_comm.pipeline_recv(name="latent", shape=(img_len, inner_dim), dtype=self._dtype)
            enc_hidden = self.pp_comm.pipeline_recv(name="encoder_hidden_state", shape=(txt_len, inner_dim), dtype=self._dtype)

            pre_infer_out = self.model.pre_infer.infer_partial(
                weights=self.model.pre_weight,
                hidden_states=hidden_states,
                encoder_hidden_states=enc_hidden,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
            )
            hidden_states, enc_hidden, num_txt = self.model.transformer_infer.infer(self.model.transformer_weights, pre_infer_out)

            if self._is_last:
                return self._run_post_infer(hidden_states, enc_hidden, num_txt, pre_infer_out.timestep)
            else:
                self.pp_comm.pipeline_send(hidden_states, name="latent", skip_shape=True)
                self.pp_comm.pipeline_send(enc_hidden, name="encoder_hidden_state", skip_shape=True)
                return None

    # ==================================================================
    # Async pipeline (main loop)
    # ==================================================================

    def _async_pipeline(self, latents, prompt_embeds, text_ids, latent_image_ids, timesteps, scheduler, do_cfg=False, negative_prompt_embeds=None, negative_text_ids=None):
        self.state.set_patched_mode(patch_mode=True)
        num_patch = self.state.num_pipeline_patch
        patch_token_nums = self.state.pp_patches_token_num
        inner_dim = self.config.get("num_attention_heads", 24) * self.config.get("attention_head_dim", 64)
        # Raw latent channels (before x_embedder): rank=0 recv from last stage
        # gets raw latents [1, L, C_in]; other stages recv embedded [L, D]
        raw_channels = getattr(self.model, "in_channels", self.config.get("transformer_in_channels", self.config.get("in_channels", 128)))

        # Split latents into patches (dim=1 for [B, L, C])
        if self._is_first or self._is_last:
            patch_latents = list(latents.split(patch_token_nums, dim=1))
        else:
            patch_latents = [None] * num_patch

        # Split image ids by patch
        patch_latent_image_ids = []
        for start, end in self.state.pp_patches_token_start_end_idx_global:
            if latent_image_ids.ndim == 3:
                patch_latent_image_ids.append(latent_image_ids[:, start:end, :])
            else:
                patch_latent_image_ids.append(latent_image_ids[start:end, :])

        # Compute txt_len for buffer allocation
        if prompt_embeds is not None:
            txt_len = prompt_embeds.shape[1] if prompt_embeds.ndim == 3 else prompt_embeds.shape[0]
        else:
            txt_len = 0

        # Pre-allocate recv buffers and pre-post all receives
        # First stage: receives raw latents [1, L, C_in] from last stage (circular)
        # Non-first stages: receives embedded encoder + latent [L, D] from previous stage
        recv_timesteps = len(timesteps) - 1 if self._is_first else len(timesteps)
        for _ in range(recv_timesteps):
            if not self._is_first:
                self.pp_comm.add_pipeline_recv_task(
                    0,
                    "encoder_hidden_state",
                    shape=(txt_len, inner_dim),
                    dtype=self._dtype,
                )
            for patch_idx in range(num_patch):
                # First stage (rank=0) receives raw latents [1, L, C_in] from
                # last stage; other stages receive embedded [L, D]
                if self._is_first:
                    latent_shape = (1, patch_token_nums[patch_idx], raw_channels)
                else:
                    latent_shape = (patch_token_nums[patch_idx], inner_dim)
                self.pp_comm.add_pipeline_recv_task(
                    patch_idx,
                    "latent",
                    shape=latent_shape,
                    dtype=self._dtype,
                )

        last_patch_latents = [None] * num_patch if self._is_last else None
        first_async_recv = True
        total_steps = len(timesteps)

        # Track pending isend requests to prevent tensor GC before send completes
        pending_isends = []

        for i, t in enumerate(timesteps):
            scheduler.step_index = i + self.state.warmup_steps
            scheduler.step_pre(scheduler.step_index)

            for patch_idx in range(num_patch):
                if self._is_last:
                    last_patch_latents[patch_idx] = patch_latents[patch_idx]

                # ---- 1. Receive current patch's data ----
                if self._is_first and i == 0:
                    pass  # first stage, first step: has initial latents
                else:
                    if first_async_recv:
                        if not self._is_first and patch_idx == 0:
                            self.pp_comm.recv_next()
                        self.pp_comm.recv_next()
                        first_async_recv = False
                    if not self._is_first and patch_idx == 0:
                        last_encoder_hidden_states = self.pp_comm.get_pipeline_recv_data(0, "encoder_hidden_state")
                    if not (self._is_first and i == 0):
                        patch_latents[patch_idx] = self.pp_comm.get_pipeline_recv_data(patch_idx, "latent")

                # ---- 2. Compute (default stream) ----
                cur_enc = prompt_embeds if self._is_first else last_encoder_hidden_states
                result = self._async_backbone(
                    patch_latents[patch_idx],
                    cur_enc,
                    text_ids,
                    patch_latent_image_ids[patch_idx],
                    scheduler,
                )

                # ---- 3. Send result (default stream, after compute) ----
                # Store isend request to prevent tensor GC before send completes
                if self._is_last:
                    noise_pred = result
                    scheduler.scheduler._step_index = i + self.state.warmup_steps
                    patch_latents[patch_idx] = scheduler.step_post_patch(noise_pred, last_patch_latents[patch_idx], t)
                    if i != total_steps - 1:
                        req = self.pp_comm.pipeline_isend(patch_latents[patch_idx], name="latent", segment_idx=patch_idx)
                        pending_isends.append((req, patch_latents[patch_idx]))
                else:
                    hidden_states, next_enc = result
                    if patch_idx == 0:
                        req = self.pp_comm.pipeline_isend(next_enc, name="encoder_hidden_state", segment_idx=0)
                        pending_isends.append((req, next_enc))
                    req = self.pp_comm.pipeline_isend(hidden_states, name="latent", segment_idx=patch_idx)
                    pending_isends.append((req, hidden_states))

                # ---- 4. Post next irecv (default stream — NCCL internal
                # stream handles the actual async transfer; no cross-stream
                # sync needed.) ----
                if not (self._is_first and i == 0):
                    is_last_step = i == total_steps - 1
                    is_last_patch = patch_idx == num_patch - 1
                    if not (is_last_step and is_last_patch):
                        if self._is_first:
                            self.pp_comm.recv_next()
                        else:
                            if is_last_patch:
                                self.pp_comm.recv_next()
                            self.pp_comm.recv_next()

                # ---- 5. Wait for old isends (limit pending to prevent GC issues) ----
                while len(pending_isends) > num_patch * 2:
                    old_req, _ = pending_isends.pop(0)
                    old_req.wait()

                self.state.next_patch()

        # Wait for all remaining isends before returning
        for req, _ in pending_isends:
            req.wait()
        pending_isends.clear()

        if self._is_last:
            return torch.cat(patch_latents, dim=1)
        return None

    def _async_backbone(self, patch_latent, encoder_hidden_states, text_ids, patch_img_ids, scheduler):
        """Backbone forward for a single patch in async mode."""
        if self._is_first:
            pre_infer_out = self.model.pre_infer.infer(
                weights=self.model.pre_weight,
                hidden_states=patch_latent,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=text_ids,
                img_ids=patch_img_ids,
            )
        else:
            pre_infer_out = self.model.pre_infer.infer_partial(
                weights=self.model.pre_weight,
                hidden_states=patch_latent,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=text_ids,
                img_ids=patch_img_ids,
            )

        hidden_states, enc_hidden, num_txt = self.model.transformer_infer.infer(self.model.transformer_weights, pre_infer_out)

        if self._is_last:
            return self._run_post_infer(hidden_states, enc_hidden, num_txt, pre_infer_out.timestep)
        else:
            return (hidden_states, enc_hidden)

    # ==================================================================
    # Shared helpers
    # ==================================================================

    def _run_post_infer(self, hidden_states, enc_hidden, num_txt, timestep):
        """Run post_infer on the last stage."""
        if enc_hidden is None and num_txt > 0:
            hidden_states = hidden_states[num_txt:, ...]
        noise_pred = self.model.post_infer.infer(self.model.post_weight, hidden_states, timestep)
        return noise_pred
