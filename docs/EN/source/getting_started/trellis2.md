# TRELLIS.2-4B BF16 inference

LightX2V provides the native 512 image-to-3D (`i23d`) pipeline, including three 12-step samplers, geometry GLB and texture voxel outputs. Sorted neighbor lookup accelerates sparse decoding.

See `scripts/bench/trellis2/README.md` for installation, pinned assets, ordinary inference, exact single-device A/B measurements and limitations. Model and operator implementations use LightX2V registries without vendor branches. CUDA/A100 and MetaX performance have not been measured.
