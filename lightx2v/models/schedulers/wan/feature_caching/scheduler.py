from lightx2v.models.schedulers.wan.scheduler import WanScheduler


class WanSchedulerCaching(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        self.transformer_infer.clear()


class WanSchedulerTaylorCaching(WanSchedulerCaching):
    def __init__(self, config):
        super().__init__(config)

        pattern = [True, False, False, False]
        self.caching_records = (pattern * ((config.infer_steps + 3) // 4))[: config.infer_steps]
        self.caching_records_2 = (pattern * ((config.infer_steps + 3) // 4))[: config.infer_steps]
