class NeoppPostInfer:
    def __init__(self, config):
        self.config = config
        self.t_eps = 0.02

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, pre_infer_out, hidden_states):
        v_pred = (hidden_states - pre_infer_out.z) / (1 - pre_infer_out.t).clamp_min(self.t_eps)
        return v_pred
