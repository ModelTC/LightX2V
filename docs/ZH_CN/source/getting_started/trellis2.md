# TRELLIS.2-4B BF16 推理

原生 `i23d` runner 支持 512 三阶段采样、几何 GLB 和纹理体素；sorted 邻居构建加速稀疏解码。

See `scripts/bench/trellis2/README.md` for installation, pinned assets, ordinary inference, exact single-device A/B measurements and limitations. Model and operator implementations use LightX2V registries without vendor branches. CUDA/A100 and MetaX performance have not been measured.
