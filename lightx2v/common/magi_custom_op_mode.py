"""Global switch: use magi subgraph-boundary custom ops only when MAGI compile is active."""

_use_magi_custom_ops = False


def set_magi_custom_op_mode(enabled: bool) -> None:
    global _use_magi_custom_ops
    _use_magi_custom_ops = bool(enabled)


def use_magi_custom_ops() -> bool:
    return _use_magi_custom_ops
