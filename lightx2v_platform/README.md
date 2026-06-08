# lightx2v_platform

**\[ English | [中文](README_zh.md) \]**

`lightx2v_platform` is a functional platform that is independent of `lightx2v`. It is used to align inference interfaces for non-NVIDIA chip backends. To support a new chip backend, you only need to focus on the implementation inside `lightx2v_platform`.

Currently supported backends include:
- Cambricon MLU590
- MetaX C500
- Hygon DCU
- Ascend 910B
- AMD ROCm
- MThreads MUSA
- Enflame S60 (GCU)
- Intel AIPC PTL
- iluvatar

For the corresponding Docker environments, see: https://github.com/ModelTC/LightX2V/tree/main/dockerfiles/platforms

For the corresponding usage scripts, see: https://github.com/ModelTC/LightX2V/tree/main/scripts/platforms

## Out-of-tree platform plugins

Besides in-tree backends, a chip backend can be shipped as a separate
pip-installable package and discovered at runtime via entry points — no edits to
this repository required.

A plugin package declares an entry point in the `lightx2v.platform_plugins`
group:

```toml
# plugin package's pyproject.toml
[project.entry-points."lightx2v.platform_plugins"]
my_backend = "my_backend_pkg:register"
```

The referenced callable registers a `Device` class into
`PLATFORM_DEVICE_REGISTER` and its operators into the `PLATFORM_*` op registries:

```python
# my_backend_pkg/__init__.py
def register():
    from . import device   # @PLATFORM_DEVICE_REGISTER("my_backend")
    from .ops import register_ops
    register_ops()
```

`set_ai_device()` scans this group before initialising the device and before the
op registries are imported, so plugin registrations reach the framework. Then:

```bash
pip install my-backend-package
PLATFORM=my_backend python lightx2v/infer.py ...
```

When no plugins are installed this scan is a no-op, and a plugin that fails to
load is logged and skipped rather than aborting device setup.
