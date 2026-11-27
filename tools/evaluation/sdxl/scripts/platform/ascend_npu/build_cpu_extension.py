"""Build the pinned MMCV CPU extension with the system C++ compiler."""
import os
import runpy
import sys
import torch.utils.cpp_extension

os.environ.update(CC="/usr/bin/gcc", CXX="/usr/bin/g++")
sys.argv = ["setup.py", "bdist_wheel"]
runpy.run_path("setup.py", run_name="__main__")
