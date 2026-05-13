# Minimal stubs for Lyra-2 callback classes.
# Only the classes needed as default field values in Config are defined here.
# Full training callback logic is not needed for inference.

import attrs


@attrs.define(slots=False)
class EMAModelCallback:
    """Stub for training-only EMA model callback."""

    pass


@attrs.define(slots=False)
class ProgressBarCallback:
    """Stub for training-only progress bar callback."""

    pass
