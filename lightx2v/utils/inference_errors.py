"""Exit on fatal device errors without touching the broken GPU runtime."""

import os
import traceback
from contextlib import contextmanager


def terminate_inference_process(reason, error=None):
    # Reopen stderr with an independent file description: changing fd 2's
    # O_NONBLOCK flag would also change stderr inherited by sibling ranks.
    fd = None
    try:
        fd = os.open("/proc/self/fd/2", os.O_WRONLY | os.O_NONBLOCK | os.O_APPEND | os.O_CLOEXEC)
        payload = f"LightX2V fatal inference failure: {reason}; exiting for supervisor restart\n".encode()[:1024]
        if error is not None:
            # Read frame metadata only, without formatting locals or looking up
            # source files. Keep this distinct from an uncaught Python traceback.
            stack = []
            seen = set()
            while error is not None and id(error) not in seen and len(seen) < 8:
                seen.add(id(error))
                frames = list(traceback.walk_tb(error.__traceback__))[-8:]
                stack.append(f"{type(error).__name__} stack (most recent call last):\n")
                stack.extend(f'  File "{frame.f_code.co_filename}", line {line}, in {frame.f_code.co_name}\n' for frame, line in frames)
                error = error.__cause__ if error.__cause__ is not None else error.__context__
            payload += "".join(stack).encode()[-3072:]
        os.write(fd, payload)
    except Exception:
        # Closed/full/unsupported stderr and diagnostic failures must not stop exit.
        pass
    finally:
        try:
            if fd is not None:
                os.close(fd)
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
    )
)

# MUSA 5.2.0 driver_types.h explicitly requires process termination/relaunch
# for these runtime codes. Strings were queried from libmusart.so.5.2.0.
# torch_musa 2.9.1.post1+musa5.2.0 native checks use "MUSA error: ";
# its generated CUDA-compatible checks use "CUDA error: ". XID names are
# diagnostic categories, not verified runtime exception strings.
_MUSA_FATAL_RUNTIME_MESSAGES = (
    "invalid access of peer gpu memory over mtlink or a hardware error",  # 226: musaErrorContained
    "an illegal memory access was encountered",  # 700: musaErrorIllegalAddress
    "the launch timed out and was terminated",  # 702: musaErrorLaunchTimeout
    "device-side assert triggered",  # 710: musaErrorAssert
    "hardware stack error",  # 714: musaErrorHardwareStackError
    "an illegal instruction was encountered",  # 715: musaErrorIllegalInstruction
    "misaligned address",  # 716: musaErrorMisalignedAddress
    "operation not supported on global/shared address space",  # 717: musaErrorInvalidAddressSpace
    "invalid program counter",  # 718: musaErrorInvalidPc
    "unspecified launch failure",  # 719: musaErrorLaunchFailure
    "tensor memory not completely freed",  # 721: musaErrorTensorMemoryLeak
    "operation aborted",  # 783: musaErrorAborted
    "out of bound",  # 784: musaErrorOutOfBound
    "fp invalid vector",  # 785: musaErrorFpInvalidVector
    "fp invalid tensor",  # 786: musaErrorFpInvalidTensor
    "fp overflow vector",  # 787: musaErrorFpOverflowVector
    "fp overflow tensor",  # 788: musaErrorFpOverflowTensor
    "read write conflict",  # 789: musaErrorReadWriteConflict
    "read write no barrier",  # 790: musaErrorReadWriteNoBarrier
    "cluster write fail",  # 791: musaErrorClusterWriteFail
    "cluster read fail",  # 792: musaErrorClusterReadFail
    "tme illegal parameters",  # 793: musaErrorTmeIllegalParameters
    "tce illegal active mask",  # 794: musaErrorTceIllegalActiveMask
    "kill timeout",  # 795: musaErrorKillTimeout
    "axi bus timeout",  # 796: musaErrorAxiBusTimeout
    "axi bus decode error",  # 797: musaErrorAxiBusDecodeError
    "axi bus slave error",  # 798: musaErrorAxiBusSlaveError
    "context switch timeout",  # 799: musaErrorCswTimeout
    "the mps client has been terminated by the server",  # 810: musaErrorMpsClientTerminated
    "an async error has occured in external entity outside of musa",  # 911: musaErrorExternalDevice
)
_FATAL_DEVICE_MESSAGES |= frozenset(prefix + message for prefix in ("musa error: ", "cuda error: ") for message in _MUSA_FATAL_RUNTIME_MESSAGES)


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
        terminate_inference_process(f"{type(error).__name__}: {error}", error=error)


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
