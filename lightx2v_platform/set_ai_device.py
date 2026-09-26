import os

from loguru import logger

from lightx2v_platform import *  # noqa: F403


def _load_platform_plugins(platform):
    """Discover out-of-tree platform backends via entry points.

    Third-party packages register under the ``lightx2v.platform_plugins`` entry
    point group. Each entry point is a zero-arg callable that registers its
    Device class into ``PLATFORM_DEVICE_REGISTER`` and may return a second
    callable that loads ops after device initialization.

    This runs before ``init_ai_device`` (so a plugin-provided device is visible
    to the lookup) and before ``lightx2v_platform.ops`` is imported, i.e. before
    ``lightx2v.utils.registry_factory`` snapshots the selected platform's
    registries via ``merge()``. That ordering is what makes plugin
    registrations reach the framework-facing registries.

    A plugin for the selected platform is required to load successfully. An
    unrelated optional plugin may fail without preventing device setup.
    """
    loaders = []
    try:
        from importlib.metadata import entry_points
    except Exception:  # pragma: no cover - importlib.metadata is stdlib on 3.8+
        return loaders

    try:
        eps = entry_points(group="lightx2v.platform_plugins")
    except TypeError:
        # importlib.metadata < 3.10 returns a dict-like mapping.
        eps = entry_points().get("lightx2v.platform_plugins", [])

    for ep in eps:
        try:
            plugin_loader = ep.load()()
            if callable(plugin_loader) and ep.name == platform:
                loaders.append((ep, plugin_loader))
            logger.info(f"Loaded LightX2V platform plugin: {ep.name}")
        except Exception as e:
            if ep.name == platform:
                raise RuntimeError(f"Failed to load selected platform plugin '{ep.name}'") from e
            logger.warning(f"Failed to load platform plugin '{ep.name}': {e}")
    return loaders


def set_ai_device():
    platform = os.getenv("PLATFORM", "cuda")
    plugin_loaders = _load_platform_plugins(platform)
    if platform == "biren_supa" and not plugin_loaders:
        raise RuntimeError(
            "Using PLATFORM=biren_supa requires the "
            "lightx2v-patch-biren package. Install it before starting "
            "Biren inference."
        )
    init_ai_device(platform)
    check_ai_device(platform)
    # Register in-tree shallow platform operators before optional plugins
    # install model-level patches or framework registries are snapshotted.
    import importlib

    importlib.import_module("lightx2v_platform.ops")

    for entry_point, load_ops in plugin_loaders:
        try:
            load_ops()
            logger.info(f"Loaded ops for LightX2V platform plugin: {entry_point.name}")
        except Exception as error:
            if entry_point.name == platform:
                raise RuntimeError(f"Failed to load selected platform ops '{entry_point.name}'") from error
            logger.warning(f"Failed to load ops for platform plugin '{entry_point.name}': {error}")


set_ai_device()
