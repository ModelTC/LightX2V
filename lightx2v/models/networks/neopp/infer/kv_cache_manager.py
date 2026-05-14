import torch


class KVCacheManager:
    """Manages the KV cache buffer for a single diffusion step.

    The buffer is allocated fresh each step (via ``prepare()``) so that
    Dynamo/Inductor sees it as a local tensor inside the compiled region
    rather than an escaped eager-mode buffer.  This prevents Inductor from
    generating an oversized fused Triton kernel that tries to inline the
    slice-scatter updates together with surrounding matmuls.
    """

    def __init__(self):
        self._kv_buf = None
        self._kv_past_seq = None

    def prepare(self, past_key_values: torch.Tensor, seq_len_q: int) -> None:
        """Allocate a fresh KV buffer for this step and copy the prefix.

        Args:
            past_key_values: [num_layers, 2, past_seq, num_kv_heads, head_dim]
            seq_len_q: sequence length of the current image query tokens
        """
        past_seq = past_key_values.shape[2]
        num_layers = past_key_values.shape[0]
        num_kv = past_key_values.shape[1]
        num_kv_heads = past_key_values.shape[3]
        head_dim = past_key_values.shape[4]
        self._kv_buf = torch.empty(
            num_layers,
            num_kv,
            past_seq + seq_len_q,
            num_kv_heads,
            head_dim,
            dtype=past_key_values.dtype,
            device=past_key_values.device,
        )
        self._kv_buf[:, :, :past_seq] = past_key_values
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

    def clear(self):
        self._kv_buf = None
        self._kv_past_seq = None
