# External code and weights

Model inference comes from the fixed ModelTC/LightX2V inference PR, and the original model project/package pinned in configs/nvidia/source-lock.json. No external model implementation is vendored in this evaluation PR. Check each source’s license and model/data access terms before downloading or redistribution.

The HumanML feature networks, skeleton/motion conversion and metric functions are imported from EricGuo5513/momask-codes at the locked commit. Runtime compatibility changes have both source and patched SHA256 values in source-lock.json. Benchmark-specific FK/resampling, cropping and aggregation live in evaluate.py. HumanML3D, GloVe, Qwen and CLIP assets remain external.

Optional NVIDIA resource acquisition uses NVIDIA `nvidia-ml-py` 12.550.52 (BSD license), installed separately and not vendored. Resource policy and PID ancestry remain provided by the pinned LightX2V `gpu_watch.py`.
