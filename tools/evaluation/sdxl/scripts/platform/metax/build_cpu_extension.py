"""Build released CPU extensions with the system compiler on MetaX."""

import os
import runpy
import sys

# MetaX's torch import selects cu-bridge. Import it before selecting the CPU
# compiler; setup.py then reuses the initialized module without changing CXX.
import torch.utils.cpp_extension  # noqa: F401

os.environ.update(CC="/usr/bin/gcc", CXX="/usr/bin/g++")
sys.argv = ["setup.py", "bdist_wheel"]
runpy.run_path("setup.py", run_name="__main__")
