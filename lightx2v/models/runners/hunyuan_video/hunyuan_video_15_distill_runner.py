import os
from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner import HunyuanVideo15Runner
from lightx2v.models.schedulers.hunyuan_video.step_distill.scheduler import HunyuanVideo15StepDistillScheduler
from lightx2v.models.schedulers.hunyuan_video.scheduler import HunyuanVideo15SRScheduler
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("hunyuan_video_1.5_distill")
class HunyuanVideo15DistillRunner(HunyuanVideo15Runner):
    def __init__(self, config):
        super().__init__(config)

    def init_scheduler(self):
        self.scheduler = HunyuanVideo15StepDistillScheduler(self.config)

        if self.sr_version is not None:
            self.scheduler_sr = HunyuanVideo15SRScheduler(self.config_sr)
        else:
            self.scheduler_sr = None
