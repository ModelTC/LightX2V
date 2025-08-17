class QwenImagePostInfer:
    def __init__(self, config, norm_out, proj_out):
        self.config = config
        self.norm_out = norm_out
        self.proj_out = proj_out
    
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
    
    def infer(self, hidden_states, temb):
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        return output
    