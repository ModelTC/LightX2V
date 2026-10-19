# VGGT 与 TRELLIS.2

LightX2V 提供 VGGT-1B 重建任务（`recon`）和 TRELLIS.2-4B 的 512 图像生成三维任务（`i23d`）原生 BF16 runner。网络位于 `lightx2v/models/networks/`，TRELLIS 的条件编码、采样和解码分别放在已有的 encoder、scheduler、decoder 目录中。

固定版本的权重和输入下载、CLI 命令、环境版本、输出格式以及 A/B 复现方法见仓库内 `scripts/bench/reconstruction/README.md`。

VGGT 的优化是保持输入顺序的图像预处理并行化；TRELLIS 的邻居映射算法可独立于采样器和 attention backend 选择。Linear 和 attention 复用现有算子注册机制，公共稀疏算子不依赖具体模型或厂商包。

目前验证范围是每个进程独占一张 MLU590。尚未验证 CUDA/A100、MetaX 的性能，以及分布式推理和 CPU offload。TRELLIS 导出几何 GLB，并返回纹理体素属性；首版不包含纹理烘焙和级联分辨率。使用时还须遵守随模型源码保留的许可证。

旧 TRELLIS 实验存在稀疏卷积权重索引错误和 decoder BF16 转换遗漏，新实现修正了这两点。因此需要验收本版本的 A/B 数值和性能，不能直接套用旧实验的几何精度指标。
