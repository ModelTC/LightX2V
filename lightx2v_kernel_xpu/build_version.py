"""Generate hardware-specific wheel versions from the XPU build target."""

import os
import re
import sys

BASE_VERSION = "0.0.1"
TARGET_VERSION_SUFFIX = {
    "bmg": "bmg",
    "ptl-h": "ptlh",
}


def _xpu_target():
    target = os.environ.get("XPU_TARGET")
    if target is None:
        match = re.search(
            r"(?:^|\s)-DXPU_TARGET(?::STRING)?=([^\s]+)",
            os.environ.get("CMAKE_ARGS", ""),
        )
        target = match.group(1) if match else "bmg"
    if target not in TARGET_VERSION_SUFFIX:
        supported = ", ".join(TARGET_VERSION_SUFFIX)
        raise RuntimeError(f"Unsupported XPU_TARGET={target!r}; use {supported}")
    return target


def dynamic_metadata(field, settings=None):
    if field != "version":
        raise RuntimeError(f"Unsupported dynamic metadata field: {field}")
    if sys.platform == "win32":
        return BASE_VERSION
    return f"{BASE_VERSION}+{TARGET_VERSION_SUFFIX[_xpu_target()]}"
