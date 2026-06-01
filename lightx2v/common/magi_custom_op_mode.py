"""Global switch: use magi subgraph-boundary custom ops only when MAGI compile is active."""

_use_magi_custom_ops = False


def set_magi_custom_op_mode(enabled: bool) -> None:
    global _use_magi_custom_ops
    _use_magi_custom_ops = bool(enabled)


def use_magi_custom_ops() -> bool:
    return _use_magi_custom_ops


def configure_dynamo_for_magi_compile() -> None:
    """Apply Dynamo settings required by magi_compiler (call when use_magi_compile=True)."""
    import torch._dynamo as _dynamo

    _dynamo.config.capture_scalar_outputs = False
    _dynamo.config.specialize_int = False
    _dynamo.config.automatic_dynamic_shapes = True
