from lightx2v.models.runners.worldplay.worldplay_ar_runner import WorldPlayARRunner
from lightx2v.models.schedulers.worldplay.scheduler import WorldPlayDistillScheduler
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("worldplay_distill")
class WorldPlayDistillRunner(WorldPlayARRunner):
    """Autoregressive WorldPlay generation with a few-step schedule."""

    scheduler_class = WorldPlayDistillScheduler
