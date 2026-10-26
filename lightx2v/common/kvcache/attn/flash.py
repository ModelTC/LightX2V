"""Single-call FlashAttention over direct views or reusable KV workspaces."""


class FlashAttentionWorkspace:
    def __init__(self):
        self.buffers = {}

    def prepare(self, cache_key, prototype, tokens):
        signature = cache_key, tokens, prototype.shape[1:], prototype.dtype, prototype.device
        if signature not in self.buffers:
            shape = (tokens, *prototype.shape[1:])
            self.buffers[signature] = prototype.new_empty(shape), prototype.new_empty(shape)
        return self.buffers[signature]

    def reset(self, keep_storage=False):
        if not keep_storage:
            self.buffers.clear()


def apply_flash_attention(query, spans, attention, *, workspace=(), softmax_scale=None):
    spans = tuple((key, value) for key, value in spans if key.shape[0])
    if len(spans) == 1:
        key, value = spans[0]
    else:
        key, value = workspace
        offset = 0
        for source_key, source_value in spans:
            end = offset + source_key.shape[0]
            key[offset:end].copy_(source_key)
            value[offset:end].copy_(source_value)
            offset = end
    return attention.apply(q=query, k=key, v=value, causal=False, softmax_scale=softmax_scale)
