class Hunyuan3DPostInfer:
    """Post-processing for Hunyuan3D shape DiT: final norm + projection."""

    def __init__(self, config):
        self.config = config
        self.scheduler = None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        flat = weights.norm_final.apply(flat)
        flat = flat.reshape(batch_size, seq_len, hidden_dim)
        flat = flat[:, 1:]
        out = weights.linear.apply(flat.reshape(-1, hidden_dim))
        return out.reshape(batch_size, seq_len - 1, -1)
