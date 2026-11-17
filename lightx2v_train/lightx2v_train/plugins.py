import importlib


def load_plugins(modules):
    """Import the registration modules explicitly selected for this task."""
    if not isinstance(modules, list):
        raise TypeError("config.plugins must be a list of Python module names.")
    for name in modules:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid plugin module name: {name!r}.")

    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load training plugin {name!r}: {exc}") from exc
