"""ROCm-only SageAttention / Triton num_stages workaround.

On gfx1201 / gfx1100, Triton's AMD software-pipeliner miscompiles kernels at
num_stages>=3 ("operation destroyed but still has uses"). We clamp num_stages<=2
on SageAttention's Triton kernels and on any torch.compile / Inductor Triton
kernels, and force Inductor to compile in-process so the clamp reaches them.

Applied via the amd_rocm platform's ``on_sage_attn2_init`` hook: only when a
SageAttention2 backend is built, and only on the validated archs.

Toggle with the ``LIGHTX2V_ROCM_TRITON_MAX_STAGES`` env var (default 2): it is
the clamp ceiling, and setting it to 0 (or a negative value) disables the whole
workaround — e.g. once Triton fixes the pipeliner bug, so kernels can use their
own num_stages again without a code change.
"""

import os

import torch
from loguru import logger

_MAX_STAGES = int(os.getenv("LIGHTX2V_ROCM_TRITON_MAX_STAGES", "2"))
_VALIDATED_ARCHS = ("gfx1201", "gfx1100")


def _is_validated_rocm_arch():
    if getattr(torch.version, "hip", None) is None or not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith(_VALIDATED_ARCHS)
    except Exception:
        return False


def _clamp_sageattn_kernels():
    """Clamp num_stages on SageAttention's own Triton kernels."""
    import importlib
    import pkgutil

    try:
        import sageattention
        from triton.runtime.jit import JITFunction
    except ImportError:
        return
    patched = 0
    for info in pkgutil.iter_modules(sageattention.__path__, sageattention.__name__ + "."):
        try:
            module = importlib.import_module(info.name)
        except Exception as e:
            logger.debug(f"[SageAttention] ROCm: skipped {info.name}: {e}")
            continue
        for fn in vars(module).values():
            if not isinstance(fn, JITFunction) or getattr(fn, "_rocm_clamped", False):
                continue

            def run(*args, _run=fn.run, **kwargs):
                if isinstance(kwargs.get("num_stages"), int):
                    kwargs["num_stages"] = min(kwargs["num_stages"], _MAX_STAGES)
                return _run(*args, **kwargs)

            fn.run, fn._rocm_clamped = run, True
            patched += 1
    if patched:
        logger.info(f"[SageAttention] ROCm: clamped num_stages<={_MAX_STAGES} on {patched} kernels")


def _clamp_triton_compile():
    """Clamp num_stages for every Triton compilation (incl. Inductor kernels)."""
    try:
        import triton
    except ImportError:
        return
    if getattr(triton.compile, "_rocm_clamped", False):
        return
    _original = triton.compile

    def compile(*args, **kwargs):
        opts = kwargs.get("options")
        if isinstance(opts, dict) and isinstance(opts.get("num_stages"), int):
            opts["num_stages"] = min(opts["num_stages"], _MAX_STAGES)
        elif isinstance(getattr(opts, "num_stages", None), int):
            try:
                opts.num_stages = min(opts.num_stages, _MAX_STAGES)
            except Exception:
                pass
        return _original(*args, **kwargs)

    compile._rocm_clamped = True
    triton.compile = compile


def _force_inductor_single_thread():
    """Compile Inductor kernels in-process so the triton.compile clamp reaches them."""
    try:
        from torch._inductor import config as inductor_config
    except ImportError:
        return
    if getattr(inductor_config, "compile_threads", 1) != 1:
        inductor_config.compile_threads = 1
        logger.info("[SageAttention] ROCm: inductor compile_threads=1 so the num_stages clamp reaches compiled kernels")


def apply_rocm_sage_patches():
    """Install the workaround. No-op when disabled (LIGHTX2V_ROCM_TRITON_MAX_STAGES<=0)
    or off the validated RDNA archs."""
    if _MAX_STAGES <= 0:
        logger.info("[SageAttention] ROCm: num_stages workaround disabled via LIGHTX2V_ROCM_TRITON_MAX_STAGES")
        return
    if not _is_validated_rocm_arch():
        return
    _clamp_sageattn_kernels()
    _clamp_triton_compile()
    _force_inductor_single_thread()
