from ..scheduler import HunyuanScheduler
import torch


class HunyuanSchedulerTeaCaching(HunyuanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        self.transformer_infer.clear()


class HunyuanSchedulerTaylorCaching(HunyuanScheduler):
    def __init__(self, config):
        super().__init__(config)
        pattern = [True, False, False, False]
        self.caching_records = (pattern * ((config.infer_steps + 3) // 4))[: config.infer_steps]

    def clear(self):
        self.transformer_infer.clear()


class HunyuanSchedulerAdaCaching(HunyuanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        self.transformer_infer.clear()


class HunyuanSchedulerCustomCaching(HunyuanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        self.transformer_infer.clear()
