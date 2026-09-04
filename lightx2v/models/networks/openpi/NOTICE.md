# OpenPI attribution

The `pi0.py`, `gemma.py`, and `preprocessing.py` implementation in this
directory is adapted from Physical Intelligence's OpenPI project at commit
`15a9616a00943ada6c20a0f158e3adb39df2ccac`. The files under
`transformers_replace/` are vendored from that revision without behavioral changes.

OpenPI and the copied Hugging Face Transformers source files are distributed
under the Apache License 2.0. The localization changes replace OpenPI/JAX
imports with LightX2V-local, PyTorch-only modules while intentionally retaining
the official model parameter names for SafeTensors compatibility.
