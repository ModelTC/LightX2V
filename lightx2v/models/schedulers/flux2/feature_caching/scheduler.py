from lightx2v.models.schedulers.flux2.scheduler import Flux2DevScheduler, Flux2Scheduler


class Flux2SchedulerCaching(Flux2Scheduler):
    def __init__(self, config):
        super().__init__(config)
        self.caching_records_2 = [True] * self.infer_steps

    def _refresh_caching_records(self):
        self.caching_records = [True] * self.infer_steps
        self.caching_records_2 = [True] * self.infer_steps

    def set_timesteps(self):
        super().set_timesteps()
        self._refresh_caching_records()

    def clear(self):
        if self.transformer_infer is not None:
            self.transformer_infer.clear()


class Flux2DevSchedulerCaching(Flux2SchedulerCaching, Flux2DevScheduler):
    pass
