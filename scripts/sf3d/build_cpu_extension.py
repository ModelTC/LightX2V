"""Build the released SF3D CPU extensions without MetaX cu-bridge."""

import os
import runpy
import sys

# Import first: MetaX torch changes CXX during initialization.
import torch.utils.cpp_extension  # noqa: F401

os.environ.update(CC="/usr/bin/gcc", CXX="/usr/bin/g++")
sys.argv = ["setup.py", "bdist_wheel"]
runpy.run_path("setup.py", run_name="__main__")
