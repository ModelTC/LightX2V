"""Select the CPU platform before importing LightX2V; no model weights required."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cpu_platform  # noqa: E402,F401
