from loguru import logger

from lightx2v.models.schedulers.wan.changing_resolution.scheduler import WanScheduler4ChangingResolutionInterface
from lightx2v.models.schedulers.wan.feature_caching.scheduler import WanSchedulerCaching, WanSchedulerTaylorCaching
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.models.schedulers.wan.step_distill.scheduler import Wan21MeanFlowStepDistillScheduler, WanStepDistillScheduler

SUPPORTED_DISTILL_METHODS = {
    "wan2.1": ("dmd2", "mean_flow"),
    "wan2.2_moe": ("dmd2",),
}


def get_wan_distill_method(config):
    distill_method = config.get("distill_method")
    if distill_method is None:
        return None

    model_cls = config["model_cls"]
    supported_methods = SUPPORTED_DISTILL_METHODS.get(model_cls, ())
    if distill_method not in supported_methods:
        supported = ", ".join(supported_methods) or "none"
        raise NotImplementedError(f"{model_cls} does not support distill_method {distill_method!r}; supported methods: {supported}")
    return distill_method


def create_wan_scheduler(config):
    feature_caching = config["feature_caching"]
    distill_method = get_wan_distill_method(config)
    if distill_method == "dmd2":
        if feature_caching != "NoCaching":
            raise NotImplementedError("DMD2 does not support feature caching")
        scheduler_class = WanStepDistillScheduler
    elif distill_method == "mean_flow":
        if feature_caching != "NoCaching":
            raise NotImplementedError("MeanFlow does not support feature caching")
        scheduler_class = Wan21MeanFlowStepDistillScheduler
    elif distill_method is None:
        if feature_caching == "NoCaching":
            scheduler_class = WanScheduler
        elif feature_caching == "TaylorSeer":
            scheduler_class = WanSchedulerTaylorCaching
        elif feature_caching in ["Tea", "Ada", "Custom", "FirstBlock", "DualBlock", "DynamicBlock", "Mag"]:
            scheduler_class = WanSchedulerCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {feature_caching}")

    logger.info(f"Using {scheduler_class.__name__}")
    if config.get("changing_resolution", False):
        return WanScheduler4ChangingResolutionInterface(scheduler_class, config)
    return scheduler_class(config)
