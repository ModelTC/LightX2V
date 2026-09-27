# Robotics evaluation configuration

`eval.yaml` is the single defaults file for the LIBERO, LIBERO-plus and RoboTwin
batch entrypoints under `scripts/bench/robotics/`. These are strict OmegaConf
`key=value` overrides, not a Hydra launcher or training configs.

See [the benchmark guide](../../../scripts/bench/robotics/README.md) for commands,
per-benchmark overrides, model adapters, simulator dependencies and seed caches.

No default paths to machine-specific model weights are embedded here. Supply
`base_ckpt` + `lora_path`, or a merged/original `ckpt`, plus matching model assets
and dataset statistics. Simulator code defaults to repository submodules.
