"""PipeFusion-enabled transformer infer for Flux2.

Subclasses ``Flux2TransformerInfer`` to:
1. Run only the current pipeline stage's block subset.
2. Apply stale-KV caching in async (patched) mode: image KV is cached across
   patches (stale-KV approximation), while text KV is recomputed from the
   encoder hidden states it is given. NOTE: on non-first stages the async
   driver feeds patch-0 text hidden states to every patch (a second,
   stale-text approximation) — see ``pipeline_driver._async_pipeline``.
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
        from .pipeline_state import (
            get_pipeline_runtime_state,
            is_pipeline_first_stage,
            is_pipeline_last_stage,
        )

        self.pipeline_state = get_pipeline_runtime_state()
        self._is_first_stage = is_pipeline_first_stage()
        self._is_last_stage = is_pipeline_last_stage()

        self._full_k_bufs: dict = {}
        self._full_v_bufs: dict = {}

    # ------------------------------------------------------------------
    # Stale-KV hook (overrides base class no-op)
    # ------------------------------------------------------------------

    def _maybe_apply_stale_kv(self, key, value, num_txt_tokens, block_idx, block_type=None):
        """Per-patch-slot KV cache for PipeFusion.

        Semantics:
        - Keyed by ((block_type, block_idx)); double and single blocks both
          number their blocks from 0, so ``block_type`` disambiguates.
        - SYNC mode: copy the full K/V into the persistent full buffer so async
          mode can reuse it. Return input unchanged (full attention runs normally).
        - ASYNC mode: refresh the text and the current patch's image slot; other
          image slots keep their previous step's values (stale-KV approximation).
        """
        cache_key = (block_type, block_idx)
        num_patch = self.pipeline_state.num_pipeline_patch
        if num_patch <= 1 or num_txt_tokens <= 0:
            return key, value

        patch_token_nums = self.pipeline_state.pp_patches_token_num
        full_len = num_txt_tokens + sum(patch_token_nums)
        buf_k, buf_v = self._get_full_bufs(cache_key, full_len, key, value)

        if not self.pipeline_state.patch_mode:
            # Sync mode: store the full K/V for later async reuse.
            buf_k.copy_(key)
            buf_v.copy_(value)
            return key, value

        # ---- Async mode ----
        cur_slot = self.pipeline_state.pipeline_patch_idx

        # Refresh text K/V and the current patch's image slot; other image
        # slots keep the previous step's (stale) values.
        buf_k[:num_txt_tokens].copy_(key[:num_txt_tokens])
        buf_v[:num_txt_tokens].copy_(value[:num_txt_tokens])

        offset = num_txt_tokens + sum(patch_token_nums[:cur_slot])
        n = patch_token_nums[cur_slot]
        buf_k[offset : offset + n].copy_(key[num_txt_tokens:])
        buf_v[offset : offset + n].copy_(value[num_txt_tokens:])

        return buf_k[:full_len], buf_v[:full_len]

    def _get_full_bufs(self, cache_key, full_len, key, value):
        """Return (k_buf, v_buf) pre-allocated to the full sequence length."""
        if cache_key not in self._full_k_bufs or self._full_k_bufs[cache_key].shape[0] != full_len or self._full_k_bufs[cache_key].dtype != key.dtype:
            self._full_k_bufs[cache_key] = torch.empty(full_len, *key.shape[1:], dtype=key.dtype, device=key.device)
            self._full_v_bufs[cache_key] = torch.empty(full_len, *value.shape[1:], dtype=value.dtype, device=value.device)
        return self._full_k_bufs[cache_key], self._full_v_bufs[cache_key]

    def clear_kv_cache(self):
        """Clear stale-KV cache.

        NOTE: stale-KV cache persists ACROSS timesteps by design — that's the
        whole point of "stale" KV. This method is provided for defensive
        cleanup only and should NOT be called between timesteps in async mode.
        """
        self._full_k_bufs.clear()
        self._full_v_bufs.clear()

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
        image_rotary_positions = pre_infer_out.image_rotary_positions

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
                image_rotary_positions,
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
                    image_rotary_positions,
                    num_txt_tokens=num_txt_tokens,
                )

            # Split back to (text, image)
            encoder_hidden_states = hidden_states[:num_txt_tokens, ...]
            hidden_states = hidden_states[num_txt_tokens:, ...]

        return hidden_states, encoder_hidden_states, num_txt_tokens
