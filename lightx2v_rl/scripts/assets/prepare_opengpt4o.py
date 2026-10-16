#!/usr/bin/env python3
"""Extract reference images and prepare portable T2I/I2I RL manifests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lightx2v_rl.data.prepare_opengpt4o import main

if __name__ == "__main__":
    main()
