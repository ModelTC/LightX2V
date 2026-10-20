# VGGT-1B BF16 inference

LightX2V provides native reconstruction (`recon`) with camera, depth, point-map and confidence outputs. Ordered parallel CPU image preprocessing preserves the serial pixel transform.

See `scripts/bench/vggt/README.md` for installation, pinned assets, ordinary inference, exact single-device A/B measurements and limitations. Model and operator implementations use LightX2V registries without vendor branches. CUDA/A100 and MetaX performance have not been measured.
