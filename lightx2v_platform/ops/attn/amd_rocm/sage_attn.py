"""ROCm-only SageAttention / Triton num_stages workarounds.

On gfx1201 (ROCm/Triton), Triton's AMD software-pipeliner pass miscompiles
kernels at num_stages>=3 ("operation destroyed but still has uses"). These
patches clamp num_stages<=2 on both SageAttention's own Triton kernels and any
torch.compile / Inductor Triton kernels, and force Inductor to compile
in-process so the clamp actually reaches those kernels.

Imported (and applied) only on the amd_rocm platform via
``lightx2v_platform/ops/attn/amd_rocm/__init__.py``. Each function is a no-op
off ROCm, so importing it elsewhere is harmless.
"""

import torch
from loguru import logger


def clamp_sageattn_triton_num_stages(max_stages=2):
    """Clamp launch-time num_stages on every SageAttention Triton kernel."""
    if getattr(torch.version, "hip", None) is None:
        return
    import importlib
    import pkgutil

    try:
        import sageattention
        from triton.runtime.jit import JITFunction
    except ImportError:
        return

    patched = 0
    for module_info in pkgutil.iter_modules(sageattention.__path__, sageattention.__name__ + "."):
        try:
            module = importlib.import_module(module_info.name)
        except Exception as e:
            logger.debug(f"[SageAttention] ROCm: skipped {module_info.name} while installing num_stages clamp: {e}")
            continue
        for value in vars(module).values():
            if not isinstance(value, JITFunction) or getattr(value, "_rocm_num_stages_clamped", False):
                continue
            original_run = value.run

            def run(*args, _original_run=original_run, **kwargs):
                stages = kwargs.get("num_stages")
                if isinstance(stages, int) and stages > max_stages:
                    kwargs["num_stages"] = max_stages
                return _original_run(*args, **kwargs)

            try:
                value.run = run
                value._rocm_num_stages_clamped = True
                patched += 1
            except Exception:
                continue
    if patched:
        logger.info(f"[SageAttention] ROCm: clamped Triton num_stages<={max_stages} on {patched} kernels")


def clamp_triton_compile_num_stages(max_stages=2):
    """Clamp num_stages for every Triton compilation (incl. torch.compile /
    Inductor kernels) on ROCm, where num_stages>=3 crashes the AMD pipeliner."""
    if getattr(torch.version, "hip", None) is None:
        return
    try:
        import triton
    except ImportError:
        return
    if getattr(triton.compile, "_rocm_num_stages_clamped", False):
        return
    _original_compile = triton.compile

    def compile(*args, **kwargs):
        options = kwargs.get("options")
        if isinstance(options, dict) and isinstance(options.get("num_stages"), int) and options["num_stages"] > max_stages:
            options["num_stages"] = max_stages
        elif options is not None and isinstance(getattr(options, "num_stages", None), int) and options.num_stages > max_stages:
            try:
                options.num_stages = max_stages
            except Exception:
                pass
        return _original_compile(*args, **kwargs)

    compile._rocm_num_stages_clamped = True
    triton.compile = compile


def force_rocm_inductor_single_thread():
    """Make torch.compile / Inductor compile in-process on ROCm.

    The num_stages clamp on ``triton.compile`` only takes effect in the process
    that installed it. Inductor's default multi-threaded pool compiles kernels
    in worker subprocesses that never see the clamp, so an Inductor-generated
    kernel can still hit the num_stages>=3 pipeliner crash. Compiling in-process
    keeps the clamp effective (one-time, warmup-only single-threaded cost)."""
    if getattr(torch.version, "hip", None) is None:
        return
    try:
        from torch._inductor import config as inductor_config
    except ImportError:
        return
    if getattr(inductor_config, "compile_threads", 1) != 1:
        inductor_config.compile_threads = 1
        logger.info("[SageAttention] ROCm: set inductor compile_threads=1 so the Triton num_stages clamp reaches torch.compile kernels")


_VALIDATED_ARCHS = ("gfx1201", "gfx1100")


def _is_validated_rocm_arch():
    """True only on the RDNA arch(es) where this workaround is validated.

    The num_stages>=3 pipeliner crash was reproduced and the clamp validated on
    gfx1201 (R9700) and gfx1100 (W7900). Other archs (e.g. MI300) are left
    untouched so their kernels keep their own num_stages tuning.
    """
    if getattr(torch.version, "hip", None) is None or not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False
    return any(arch.startswith(a) for a in _VALIDATED_ARCHS)


def apply_rocm_sage_patches():
    """Install the Triton num_stages workaround. Called only when a SageAttention2
    backend is actually constructed (via the amd_rocm platform's
    on_sage_attn2_init hook), so non-SageAttention ROCm workloads are untouched.
    No-op unless running on a validated RDNA arch, where Triton's pipeliner
    miscompiles at num_stages>=3."""
    if not _is_validated_rocm_arch():
        return
    clamp_sageattn_triton_num_stages()
    clamp_triton_compile_num_stages()
    force_rocm_inductor_single_thread()
