import torch
from loguru import logger
from torch.nn.parallel import DistributedDataParallel

from lightx2v_train.runtime.distributed import is_distributed


class LightX2VDistributedDataParallel(DistributedDataParallel):
    """DDP wrapper that keeps the denoiser usable as the original transformer.

    LightX2V replaces ``model.transformer`` with this wrapper when data
    parallelism is enabled.  The extra forwarding below lets existing model,
    LoRA, checkpoint, and Wan causal-mask code keep calling attributes and
    methods on ``model.transformer`` without having to special-case
    ``DistributedDataParallel.module`` everywhere.
    """

    @property
    def __class__(self):
        # Preserve class-based checks such as isinstance(transformer, CausalWanModel)
        # after the transformer is wrapped by DDP.
        return self.module.__class__

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as error:
            try:
                module = super().__getattr__("module")
            except AttributeError:
                raise error
            # Expose attributes/methods from the wrapped denoiser directly.
            return getattr(module, name)

    def __setattr__(self, name, value):
        modules = self.__dict__.get("_modules")
        wrapped = modules.get("module") if modules is not None else None
        if name == "block_mask" and wrapped is not None:
            # Causal Wan stores the attention block mask on the transformer used
            # by forward(), so write this field through to the wrapped module.
            setattr(wrapped, name, value)
            return
        super().__setattr__(name, value)

    def state_dict(self, *args, **kwargs):
        # Save plain denoiser keys instead of DDP's "module."-prefixed keys.
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        # Match the plain state_dict() format used for non-DDP checkpoints.
        return self.module.load_state_dict(*args, **kwargs)


def ddp_config(config):
    distributed_config = config.get("distributed", {})
    config_value = distributed_config.get("dp", {})
    return config_value or {}


def ddp_enabled(config):
    return is_distributed() and ddp_config(config).get("enabled", False)


def is_ddp_module(module):
    return isinstance(module, DistributedDataParallel)


def unwrap_ddp_module(module):
    while is_ddp_module(module):
        module = module.module
    return module


def set_ddp_gradient_sync(module, enabled):
    if is_ddp_module(module):
        module.require_backward_grad_sync = enabled


def _ddp_kwargs(config):
    config = ddp_config(config)
    kwargs = {
        "broadcast_buffers": config.get("broadcast_buffers", False),
        "find_unused_parameters": config.get("find_unused_parameters", False),
        "static_graph": config.get("static_graph", False),
        "gradient_as_bucket_view": config.get("gradient_as_bucket_view", False),
    }
    if torch.cuda.is_available():
        kwargs["device_ids"] = [torch.cuda.current_device()]
        kwargs["output_device"] = torch.cuda.current_device()
    return kwargs


def apply_ddp(model, config):
    if not ddp_enabled(config) or is_ddp_module(model.denoiser_module()):
        return model

    denoiser = model.denoiser_module()

    if not any(param.requires_grad for param in denoiser.parameters()):
        logger.info("DP(DDP) skipped for {} because the denoiser has no trainable parameters.", model.__class__.__name__)
        return model

    ddp_kwargs = _ddp_kwargs(config)
    wrapped = LightX2VDistributedDataParallel(denoiser, **ddp_kwargs)
    if getattr(model, "transformer", None) is not denoiser:
        raise RuntimeError(f"{model.__class__.__name__} must store its trainable denoiser in self.transformer to use DP(DDP).")
    model.transformer = wrapped
    logger.info(
        "DP(DDP) transformer wrapped: broadcast_buffers={} find_unused_parameters={} static_graph={}",
        ddp_kwargs["broadcast_buffers"],
        ddp_kwargs["find_unused_parameters"],
        ddp_kwargs["static_graph"],
    )
    return model
