"""Sync/async PipeFusion pipeline driver for MiniMax-H3.

Stage 0 embeds the packed sequence and slices it into [text | patch] pieces;
middle stages run their block subset per patch; the last stage post-processes
per patch and circles the updated latents back to stage 0.  All layout
metadata is recomputed locally from the deterministic scheduler state, so only
activations cross the P2P links.
"""

from dataclasses import replace

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
from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3PreInferOutput
from lightx2v.utils.envs import GET_DTYPE

# Latent streams the last stage circles back to stage 0 per step (video +
# audio); the in-flight isend cap must leave room for both.
_RING_LATENT_STREAMS = 2


class MiniMaxH3PipelineDriver:
    """Drives the PipeFusion denoising loop for MiniMax-H3."""

    def __init__(self, model, config):
        self.model = model
        self.state = get_pipeline_runtime_state()
        self.pp_comm = PipelineComm(get_pp_group())
        self._is_first = is_pipeline_first_stage()
        self._is_last = is_pipeline_last_stage()
        self._pp_world_size = get_pipeline_parallel_world_size()
        self._dtype = GET_DTYPE()
        self._hidden_size = int(config.get("hidden_size", 5376))
        # temb (the timestep embedding) is recomputed locally on every stage from
        # the deterministic scheduler state and the time-MLP weights each stage
        # holds, so the P2P carries a single hidden-state stream.  With
        # use_adaln_cache the time-MLP weights are not loaded and temb is None
        # (cached AdaLN tables replace it on every stage).
        self._local_temb_enabled = not model.use_adaln_cache
        self._warmup_p2p()

    def _warmup_p2p(self):
        """Pre-create every neighbour pair's NCCL P2P comm.

        NCCL creates a per-pair comm lazily on the first P2P op; the rendezvous
        needs both ranks to enter together, which the pipeline's staggered
        first P2P can deadlock.
        """
        dist.barrier()
        tiny = torch.zeros(1, dtype=self._dtype, device=f"cuda:{torch.cuda.current_device()}")
        if self.pp_comm.rank % 2 == 0:
            self.pp_comm.pipeline_send(tiny)
            self.pp_comm.pipeline_recv((1,), self._dtype)
        else:
            self.pp_comm.pipeline_recv((1,), self._dtype)
            self.pp_comm.pipeline_send(tiny)

    def _local_temb(self):
        """Recompute temb on non-first stages (single hidden stream per step)."""
        if not self._local_temb_enabled:
            return None
        return self.model.pre_infer.compute_temb(self.model.pre_weight)

    # Public entry point

    def run_pipeline(self, prompt_embeds, scheduler, num_steps=None):
        """Run the denoising loop; num_steps limits it (warmup), None = full schedule.

        Latents are updated in place on the last stage only.
        """
        total_steps = scheduler.infer_steps if num_steps is None else int(num_steps)
        if not 1 <= total_steps <= scheduler.infer_steps:
            raise ValueError(f"num_steps must be within [1, {scheduler.infer_steps}], got {total_steps}")
        self._init_metadata(scheduler)
        warmup = self.state.warmup_steps
        if self._pp_world_size > 1 and total_steps > warmup:
            self._sync_pipeline(prompt_embeds, scheduler, range(warmup))
            self._async_pipeline(prompt_embeds, scheduler, range(warmup, total_steps))
        else:
            self._sync_pipeline(prompt_embeds, scheduler, range(total_steps))

    def _init_metadata(self, scheduler):
        layout = scheduler.layout
        self._text_len = int(layout.text_indices.numel())
        self._seq_len = int(layout.sequence_length)
        self._rotary = self.model.pre_infer._rotary_embedding(layout.position_ids)
        self._cond_video = int(layout.num_condition_video_rows)
        self._cond_audio = int(layout.num_condition_audio_rows)

        text_rows = torch.arange(self._text_len, device=layout.video_indices.device)
        bounds = self.state.pp_patches_token_start_end_idx_global
        self._patch_rows = []
        self._patch_video_pos = []
        self._patch_video_rows = []
        self._patch_audio_pos = []
        self._patch_audio_rows = []
        for p, n in enumerate(self.state.pp_patches_token_num):
            start = self._text_len + bounds[p][0]
            end = start + n
            self._patch_rows.append(torch.cat((text_rows, torch.arange(start, end, device=layout.video_indices.device))))
            # Target video/audio rows whose packed index falls inside this patch.
            # _pos are positions within the patch-only rows seen by post_infer
            # (text rows are sliced off); _rows are the scheduler latent rows
            # they update.
            tv = layout.video_indices[self._cond_video :]
            mask = (tv >= start) & (tv < end)
            self._patch_video_rows.append(torch.nonzero(mask).squeeze(-1) + self._cond_video)
            self._patch_video_pos.append(tv[mask] - start)
            ta = layout.audio_indices[self._cond_audio :]
            mask_a = (ta >= start) & (ta < end)
            self._patch_audio_rows.append(torch.nonzero(mask_a).squeeze(-1) + self._cond_audio)
            self._patch_audio_pos.append(ta[mask_a] - start)

    # Sync pipeline (warmup)

    def _sync_pipeline(self, prompt_embeds, scheduler, steps):
        self.state.set_patched_mode(False)
        for i in steps:
            scheduler.step_pre(i)
            if self._is_first:
                pre = self.model.pre_infer.infer(self.model.pre_weight, prompt_embeds)
                hidden = self.model.transformer_infer.infer(self.model.transformer_weights, pre)
                self.pp_comm.pipeline_send(hidden)
            else:
                hidden = self.pp_comm.pipeline_recv((self._seq_len, self._hidden_size), self._dtype)
                temb = self._local_temb()
                pre = self.model.pre_infer._metadata(hidden, temb)
                hidden = self.model.transformer_infer.infer(self.model.transformer_weights, pre)
                if self._is_last:
                    output = self.model.post_infer.infer(self.model.post_weight, hidden, pre)
                    scheduler.video_noise_pred = output.video
                    scheduler.audio_noise_pred = output.audio
                    scheduler.step_post()
                else:
                    self.pp_comm.pipeline_send(hidden)

            # Circular P2P: last stage sends updated latents to first stage.
            if self._pp_world_size > 1:
                if self._is_last:
                    self.pp_comm.pipeline_send(scheduler.video_latents)
                    self.pp_comm.pipeline_send(scheduler.audio_latents)
                elif self._is_first:
                    video = self.pp_comm.pipeline_recv(scheduler.video_latents.shape, scheduler.video_latents.dtype)
                    audio = self.pp_comm.pipeline_recv(scheduler.audio_latents.shape, scheduler.audio_latents.dtype)
                    scheduler.video_latents.copy_(video)
                    scheduler.audio_latents.copy_(audio)

    # Async pipeline (main loop)

    def _async_pipeline(self, prompt_embeds, scheduler, steps):
        self.state.set_patched_mode(True)
        num_patch = self.state.num_pipeline_patch
        patch_len = [self._text_len + n for n in self.state.pp_patches_token_num]
        total_steps = len(steps)
        last_step = steps[-1]

        # Pre-allocate recv buffers and pre-post all receives.
        # First stage receives the full updated latents (circular); non-first
        # stages receive one hidden stream per patch (temb is recomputed
        # locally).
        recv_steps = total_steps - 1 if self._is_first else total_steps
        for _ in range(recv_steps):
            if self._is_first:
                self.pp_comm.add_pipeline_recv_task(0, "video_latent", shape=scheduler.video_latents.shape, dtype=scheduler.video_latents.dtype)
                self.pp_comm.add_pipeline_recv_task(0, "audio_latent", shape=scheduler.audio_latents.shape, dtype=scheduler.audio_latents.dtype)
            else:
                for p in range(num_patch):
                    self.pp_comm.add_pipeline_recv_task(p, "hidden", shape=(patch_len[p], self._hidden_size), dtype=self._dtype)

        # Track pending isend requests to prevent tensor GC before send completes.
        pending_isends = []

        for i, step_idx in enumerate(steps):
            scheduler.step_pre(step_idx)
            if self._is_first:
                if i > 0:
                    self.pp_comm.recv_next()
                    self.pp_comm.recv_next()
                    scheduler.video_latents.copy_(self.pp_comm.get_pipeline_recv_data(0, "video_latent"))
                    scheduler.audio_latents.copy_(self.pp_comm.get_pipeline_recv_data(0, "audio_latent"))
                pre = self.model.pre_infer.infer(self.model.pre_weight, prompt_embeds)
                for p in range(num_patch):
                    pre_patch = self._slice_patch(pre, p)
                    hidden = self.model.transformer_infer.infer(self.model.transformer_weights, pre_patch)
                    req, sent = self.pp_comm.pipeline_isend(hidden, "hidden", p)
                    pending_isends.append((req, sent))
                    self.state.next_patch()
            else:
                # Post the whole step's receives up front so the NCCL transfers
                # overlap with the compute of earlier patches.
                for p in range(num_patch):
                    self.pp_comm.recv_next()
                temb = self._local_temb()
                for p in range(num_patch):
                    hidden = self.pp_comm.get_pipeline_recv_data(p, "hidden")
                    pre_patch = self._build_patch_pre(hidden, temb, p, scheduler)
                    hidden = self.model.transformer_infer.infer(self.model.transformer_weights, pre_patch)
                    if self._is_last:
                        post_pre = self._post_pre(pre_patch, p)
                        output = self.model.post_infer.infer(self.model.post_weight, hidden[self._text_len :], post_pre)
                        scheduler.step_post_patch(
                            self._patch_video_rows[p],
                            self._patch_audio_rows[p],
                            output.video,
                            output.audio,
                        )
                    else:
                        req, sent = self.pp_comm.pipeline_isend(hidden, "hidden", p)
                        pending_isends.append((req, sent))
                    self.state.next_patch()

                if self._is_last and step_idx != last_step:
                    req, sent = self.pp_comm.pipeline_isend(scheduler.video_latents.clone(), "video_latent", 0)
                    pending_isends.append((req, sent))
                    req, sent = self.pp_comm.pipeline_isend(scheduler.audio_latents.clone(), "audio_latent", 0)
                    pending_isends.append((req, sent))

            # Limit in-flight isends to prevent tensor GC issues.
            while len(pending_isends) > num_patch * _RING_LATENT_STREAMS:
                old_req, _ = pending_isends.pop(0)
                old_req.wait()

        for req, _ in pending_isends:
            req.wait()
        pending_isends.clear()

    # Per-patch pre/post helpers

    def _slice_patch(self, pre, p):
        """Slice a stage-0 (full-sequence) pre_infer output to [text | patch]."""
        row_idx = self._patch_rows[p]
        return replace(
            pre,
            hidden_states=pre.hidden_states[row_idx],
            timestep_indices=pre.timestep_indices[row_idx],
            adaln_indices=pre.adaln_indices[row_idx],
            rotary_emb=tuple(t[row_idx] for t in pre.rotary_emb),
        )

    def _build_patch_pre(self, hidden, temb, p, scheduler):
        """Build the [text | patch] pre_infer output on a non-first stage."""
        layout = scheduler.layout
        row_idx = self._patch_rows[p]
        timestep_indices = scheduler.timestep_indices[row_idx]
        return MiniMaxH3PreInferOutput(
            hidden_states=hidden,
            temb=temb,
            timestep_indices=timestep_indices,
            adaln_indices=timestep_indices * 3 + layout.token_tags[row_idx].clamp(min=0),
            rotary_emb=tuple(t[row_idx] for t in self._rotary),
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
        )

    def _post_pre(self, pre_patch, p):
        """Derive the patch-local pre_infer output for post_infer on the last stage."""
        return replace(
            pre_patch,
            hidden_states=pre_patch.hidden_states[self._text_len :],
            timestep_indices=pre_patch.timestep_indices[self._text_len :],
            video_indices=self._patch_video_pos[p],
            audio_indices=self._patch_audio_pos[p],
        )
