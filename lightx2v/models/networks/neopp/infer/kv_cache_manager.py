import torch


class KVCacheManager:
    """Manages the pre-allocated KV cache buffers for conditional and unconditional passes.

    The past KV (text prefix) is fixed across all diffusion steps for a given request.
    This class pre-allocates a combined [past + current] buffer per pass type, fills the
    past prefix once, and writes the current image KV via in-place slice assignment each
    layer — avoiding repeated torch.cat allocations.
    """

    def __init__(self):
        self._kv_buf_cond = None
        self._kv_buf_cond_key = None
        self._kv_buf_uncond = None
        self._kv_buf_uncond_key = None
        self._kv_buf = None
        self._kv_past_seq = None

    def prepare(self, past_key_values: torch.Tensor, seq_len_q: int, num_layers: int, is_condition: bool) -> None:
        """Select and (if needed) reallocate the active KV buffer for this infer pass.

        Args:
            past_key_values: [num_layers, 2, past_seq, num_kv_heads, head_dim]
            seq_len_q: sequence length of the current image query tokens
            num_layers: number of decoder layers
            is_condition: True for conditional pass, False for unconditional
        """
        past_seq = past_key_values.shape[2]
        buf_key = (
            past_seq,
            seq_len_q,
            past_key_values.shape[3],
            past_key_values.shape[4],
            past_key_values.dtype,
            past_key_values.device,
        )

        if is_condition:
            if self._kv_buf_cond_key != buf_key:
                self._kv_buf_cond = torch.empty(
                    num_layers, 2, past_seq + seq_len_q,
                    past_key_values.shape[3], past_key_values.shape[4],
                    dtype=past_key_values.dtype, device=past_key_values.device,
                )
                self._kv_buf_cond[:, :, :past_seq] = past_key_values
                self._kv_buf_cond_key = buf_key
            self._kv_buf = self._kv_buf_cond
        else:
            if self._kv_buf_uncond_key != buf_key:
                self._kv_buf_uncond = torch.empty(
                    num_layers, 2, past_seq + seq_len_q,
                    past_key_values.shape[3], past_key_values.shape[4],
                    dtype=past_key_values.dtype, device=past_key_values.device,
                )
                self._kv_buf_uncond[:, :, :past_seq] = past_key_values
                self._kv_buf_uncond_key = buf_key
            self._kv_buf = self._kv_buf_uncond

        self._kv_past_seq = past_seq

    def update(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
        """Write current layer's K/V into the buffer tail and return full K/V views.

        Args:
            layer_idx: decoder layer index
            key_states: [cur_seq, num_kv_heads, head_dim]
            value_states: [cur_seq, num_kv_heads, head_dim]

        Returns:
            Tuple of (key, value) views covering [past_seq + cur_seq] tokens.
        """
        self._kv_buf[layer_idx, 0, self._kv_past_seq :] = key_states
        self._kv_buf[layer_idx, 1, self._kv_past_seq :] = value_states
        return self._kv_buf[layer_idx, 0], self._kv_buf[layer_idx, 1]
