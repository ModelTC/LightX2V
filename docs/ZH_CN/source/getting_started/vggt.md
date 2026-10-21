# VGGT-1B BF16 推理

原生 `recon` runner 输出相机、深度、点图和置信度；有序多线程预处理与串行路径使用相同像素变换。

See `scripts/bench/vggt/README.md` for installation, pinned assets, ordinary inference, exact single-device A/B measurements and limitations. Model and operator implementations use LightX2V registries without vendor branches. CUDA/A100 and MetaX performance have not been measured.
