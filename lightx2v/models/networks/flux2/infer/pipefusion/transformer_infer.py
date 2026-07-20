"""PipeFusion-enabled transformer infer for Flux2.

Subclasses ``Flux2TransformerInfer`` to:
1. Run only the current pipeline stage's block subset.
2. Apply stale-KV caching in async (patched) mode: image KV is cached across
   patches while text KV stays fresh.
3. Return ``(hidden_states, encoder_hidden_states, num_txt_tokens)`` so the
   pipeline driver can P2P-pass intermediate activations between stages.
"""

import torch
import torch.nn.functional as F

from ..transformer_infer import Flux2TransformerInfer


class Flux2PipeFusionTransformerInfer(Flux2TransformerInfer):
    """Transformer infer with PipeFusion block splitting and stale-KV cache."""

    def __init__(self, config):
        super().__init__(config)
        from lightx2v.common.distributed import (
            get_pipeline_runtime_state,
            is_pipeline_first_stage,
            is_pipeline_last_stage,
        )

        self.pipeline_state = get_pipeline_runtime_state()
        self._is_first_stage = is_pipeline_first_stage()
        self._is_last_stage = is_pipeline_last_stage()

        # Stale-KV cache: block_idx -> [ [k, v] per patch slot ]
        self._kv_cache: dict = {}

        # Pre-allocated full K/V buffers per block (lazily created on first async use).
        # Avoids repeated torch.cat allocations per timestep.
        self._full_k_bufs: dict = {}
        self._full_v_bufs: dict = {}

    # ------------------------------------------------------------------
    # Stale-KV hook (overrides base class no-op)
    # ------------------------------------------------------------------

    def _maybe_apply_stale_kv(self, key, value, num_txt_tokens, block_idx):
        """Per-patch-slot KV cache for PipeFusion.

        Semantics:
        - Cache is indexed by (block_idx, patch_slot). Each slot stores the
          image K/V computed for that patch when it was last processed.
        - SYNC mode: split full image K/V by patch, populate ALL slots.
          Return input unchanged (full attention runs normally).
        - ASYNC mode: update current patch's slot with fresh K/V; use full
          cache (fresh + stale from prior timestep) for attention.

        Optimization: pre-allocated buffers + copy_ instead of torch.cat
        to avoid memory allocations per generation.
        """
        num_patch = self.pipeline_state.num_pipeline_patch
        if num_patch <= 1 or num_txt_tokens <= 0:
            return key, value

        # Split text / image along sequence dim
        text_key, img_key = key.split([num_txt_tokens, key.shape[0] - num_txt_tokens], dim=0)
        text_value, img_value = value.split([num_txt_tokens, value.shape[0] - num_txt_tokens], dim=0)

        patch_token_nums = self.pipeline_state.pp_patches_token_num

        if not self.pipeline_state.patch_mode:
            # Sync mode: split full image K/V by patch, populate all slots.
            # .clone() ensures cached tensors own their storage (views of
            # transient QKV would become invalid after this timestep).
            if block_idx not in self._kv_cache:
                self._kv_cache[block_idx] = [None] * num_patch
            split_ks = img_key.split(patch_token_nums, dim=0)
            split_vs = img_value.split(patch_token_nums, dim=0)
            for i in range(num_patch):
                self._kv_cache[block_idx][i] = [
                    split_ks[i].clone(),
                    split_vs[i].clone(),
                ]
            return key, value

        # ---- Async mode ----

        cur_slot = self.pipeline_state.pipeline_patch_idx
        if block_idx not in self._kv_cache:
            self._kv_cache[block_idx] = [None] * num_patch

        # Store fresh K/V in cache (clone for persistence across timesteps)
        self._kv_cache[block_idx][cur_slot] = [img_key.clone(), img_value.clone()]

        # Build full K/V using pre-allocated buffer + copy_ (avoids torch.cat)
        total_img = sum(patch_token_nums)
        full_len = num_txt_tokens + total_img

        if block_idx not in self._full_k_bufs or self._full_k_bufs[block_idx].shape[0] != full_len or self._full_k_bufs[block_idx].dtype != key.dtype:
            self._full_k_bufs[block_idx] = torch.empty(full_len, *key.shape[1:], dtype=key.dtype, device=key.device)
            self._full_v_bufs[block_idx] = torch.empty(full_len, *value.shape[1:], dtype=value.dtype, device=value.device)

        buf_k = self._full_k_bufs[block_idx]
        buf_v = self._full_v_bufs[block_idx]

        # Copy text K/V (fresh, from current patch's computation)
        buf_k[:num_txt_tokens].copy_(text_key)
        buf_v[:num_txt_tokens].copy_(text_value)

        # Copy each slot's image K/V into buffer
        offset = num_txt_tokens
        for slot in range(num_patch):
            n = patch_token_nums[slot]
            if slot == cur_slot:
                # Fresh from this patch (copy from img_key, already cloned to cache)
                buf_k[offset : offset + n].copy_(img_key)
                buf_v[offset : offset + n].copy_(img_value)
            else:
                # Stale from cache (previous timestep)
                cached = self._kv_cache[block_idx][slot]
                buf_k[offset : offset + n].copy_(cached[0])
                buf_v[offset : offset + n].copy_(cached[1])
            offset += n

        return buf_k[:full_len], buf_v[:full_len]

    def clear_kv_cache(self):
        """Clear stale-KV cache.

        NOTE: stale-KV cache persists ACROSS timesteps by design — that's the
        whole point of "stale" KV. This method is provided for defensive
        cleanup only and should NOT be called between timesteps in async mode.
        """
        self._kv_cache.clear()

    # ------------------------------------------------------------------
    # PipeFusion forward
    # ------------------------------------------------------------------

    def infer(self, block_weights, pre_infer_out):
        """Run this stage's blocks only.

        Returns ``(hidden_states, encoder_hidden_states, num_txt_tokens)``.

        For non-last stages, streams are ALWAYS split back to (image, text)
        before returning, so P2P always carries separate streams with
        consistent shapes.
        """
        hidden_states = pre_infer_out.hidden_states
        encoder_hidden_states = pre_infer_out.encoder_hidden_states
        timestep = pre_infer_out.timestep
        image_rotary_emb = pre_infer_out.image_rotary_emb

        # Compute num_txt_tokens
        if encoder_hidden_states is not None:
            num_txt_tokens = encoder_hidden_states.shape[0]
        else:
            # Streams already concatenated by previous stage — split them
            txt_ids = pre_infer_out.txt_ids
            num_txt_tokens = txt_ids.shape[0] if txt_ids is not None else 0
            if num_txt_tokens > 0:
                encoder_hidden_states = hidden_states[:num_txt_tokens, ...]
                hidden_states = hidden_states[num_txt_tokens:, ...]

        image_rotary_emb = self._prepare_image_rotary_emb(image_rotary_emb, num_txt_tokens)

        # Modulation embeddings (computed on every stage)
        timestep_act = F.silu(timestep)
        double_stream_mod_img = block_weights.double_stream_modulation_img_linear.apply(timestep_act)
        double_stream_mod_txt = block_weights.double_stream_modulation_txt_linear.apply(timestep_act)
        single_stream_mod = block_weights.single_stream_modulation_linear.apply(timestep_act)

        # Double-stream blocks (this stage's subset)
        for block in block_weights.double_blocks:
            encoder_hidden_states, hidden_states = self.infer_double_stream_block(
                block,
                hidden_states,
                encoder_hidden_states,
                double_stream_mod_img,
                double_stream_mod_txt,
                image_rotary_emb,
            )

        # Single-stream blocks: cat [text, image], run, then split back
        has_single = len(block_weights.single_blocks) > 0
        if has_single:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)

            for block in block_weights.single_blocks:
                hidden_states = self.infer_single_stream_block(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    image_rotary_emb,
                    num_txt_tokens=num_txt_tokens,
                )

            # Split back to (text, image)
            encoder_hidden_states = hidden_states[:num_txt_tokens, ...]
            hidden_states = hidden_states[num_txt_tokens:, ...]

        return hidden_states, encoder_hidden_states, num_txt_tokens
