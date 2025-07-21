from lightx2v.models.schedulers.wan.scheduler import WanScheduler


class WanSchedulerTeaCaching(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        self.transformer_infer.clear()


class WanSchedulerTaylorCaching(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

        pattern = [True, False, False, False]
        self.caching_records = (pattern * ((config.infer_steps + 3) // 4))[: config.infer_steps]
        self.caching_records_2 = (pattern * ((config.infer_steps + 3) // 4))[: config.infer_steps]

    def clear(self):
        self.transformer_infer.clear()


class WanSchedulerAdaCaching(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        self.transformer_infer.clear()


class WanSchedulerCustomCaching(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        self.transformer_infer.clear()


class WanSchedulerFirstBlock(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        self.transformer_infer.clear()
