import os

from loguru import logger

from lightx2v_platform import *


def _load_platform_plugins():
    """Discover out-of-tree platform backends via entry points.

    Third-party packages register under the ``lightx2v.platform_plugins`` entry
    point group. Each entry point is a zero-arg callable that registers its
    Device class into ``PLATFORM_DEVICE_REGISTER`` and its ops into the
    ``PLATFORM_*`` op registries.

    This runs before ``init_ai_device`` (so a plugin-provided device is visible
    to the lookup) and before ``lightx2v_platform.ops`` is imported, i.e. before
    ``lightx2v.utils.registry_factory`` snapshots the platform registries via
    ``merge()``. That ordering is what makes plugin registrations reach the
    framework-facing registries.

    No effect when no plugins are installed; a failing plugin is logged and
    skipped rather than aborting device setup.
    """
    try:
        from importlib.metadata import entry_points
    except Exception:  # pragma: no cover - importlib.metadata is stdlib on 3.8+
        return

    try:
        eps = entry_points(group="lightx2v.platform_plugins")
    except TypeError:
        # importlib.metadata < 3.10 returns a dict-like mapping.
        eps = entry_points().get("lightx2v.platform_plugins", [])

    for ep in eps:
        try:
            ep.load()()
            logger.info(f"Loaded LightX2V platform plugin: {ep.name}")
        except Exception as e:
            logger.warning(f"Failed to load platform plugin '{ep.name}': {e}")


def set_ai_device():
    platform = os.getenv("PLATFORM", "cuda")
    _load_platform_plugins()
    init_ai_device(platform)
    check_ai_device(platform)


set_ai_device()
from lightx2v_platform.ops import *  # noqa: E402
