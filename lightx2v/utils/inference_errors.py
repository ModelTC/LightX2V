"""Exit on fatal device errors without touching the broken GPU runtime."""

import os
from contextlib import contextmanager


def terminate_inference_process(reason):
    # Avoid GPU cleanup, distributed barriers and potentially blocked log sinks.
    try:
        os.set_blocking(2, False)
        os.write(2, f"LightX2V fatal inference failure: {reason}; exiting for supervisor restart\n".encode()[:4096])
    finally:
        os._exit(1)


# NVIDIA explicitly requires process restart for these CUDA Runtime errors.
# Text verified with CUDA 12.8 cudaGetErrorString (no GPU work required).
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
_FATAL_DEVICE_MESSAGES = frozenset(
    (
        "cuda error: invalid access of peer gpu memory over nvlink or a hardware error",  # 226
        "cuda error: an illegal memory access was encountered",  # 700
        "cuda error: the launch timed out and was terminated",  # 702
        "cuda error: device-side assert triggered",  # 710
        "cuda error: hardware stack error",  # 714
        "cuda error: an illegal instruction was encountered",  # 715
        "cuda error: misaligned address",  # 716
        "cuda error: operation not supported on global/shared address space",  # 717
        "cuda error: invalid program counter",  # 718
        "cuda error: unspecified launch failure",  # 719
        "cuda error: tensor memory not completely freed",  # 721
        "cuda error: the mps client has been terminated by the server",  # 810
        # 911: only the explicit asynchronous error form; generic external errors are ambiguous.
        "cuda error: an async error has occured in external entity outside of cuda",
        # Moore Threads XID errors explicitly classified FATAL. These accepted
        # message forms identify the fault; XID numbers are NOT MUSA API codes.
        # https://docs.mthreads.com/gmc/gmc-doc-online/dcgm/historicalversion/v1.0.0/xid_guide/
        "musa error: gpu initialization failure",  # 0x00000001
        "musa error: gpu hardware failure",  # 0x01000006
        "musa error: uncorrectable uncontained ecc error",  # 0x02000002
        "musa error: ecc error counter overflow",  # 0x02000004
        "musa error: page retire failed",  # 0x02000006
        "musa error: axi error",  # 0x02000007
        "musa error: axi bus error",  # explicit AXI wording
        "musa error: gpu bios trap",  # 0x05000002
        "musa error: gpu shutdown",  # 0x05000004
        "musa error: mtlink unrecoverable error",  # 0x07000003
    )
)


def is_fatal_device_error(error):
    # Cleanup/wrapper failures can hide the original device exception.
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        # Match the full first line of a RuntimeError, not arbitrary mentions of
        # a fault in input validation, paths, warnings or diagnostic prose.
        if isinstance(error, RuntimeError) and str(error).split("\n", 1)[0].strip().lower() in _FATAL_DEVICE_MESSAGES:
            return True
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return False


def terminate_on_fatal_device_error(error):
    """Handle fatal exceptions before normal failure handling or cleanup."""
    if is_fatal_device_error(error):
        terminate_inference_process(f"{type(error).__name__}: {error}")


@contextmanager
def fatal_device_error_guard():
    """Check body errors before exiting an earlier context in the same `with`.

    Context construction/entry/exit errors still need an enclosing guard when
    they must be checked before another outer device context exits.
    """
    try:
        yield
    except Exception as error:
        terminate_on_fatal_device_error(error)
        raise
