# Stable Fast 3D → LightX2V (A100)

目标权重为 `stabilityai/stable-fast-3d`，revision
`f0c9a8ffd62cb1bbc8a7a53c9f87a0be1b6be778`。
代码移植自 `stability-ai/stable-fast-3d` commit
`ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2`，模型目录保留其许可证。
运行时不调用上游 SF3D、transformers 或 OpenCLIP 的神经网络实现。

先执行 `bash scripts/three_models/install_a100.sh`。另外需要 CUDA toolkit、C++ 编译器
及两个 SF3D 几何扩展。已有编译环境可直接复用；新环境从上述固定 commit 构建：

```bash
git clone https://github.com/Stability-AI/stable-fast-3d.git /tmp/sf3d-geometry-build
git -C /tmp/sf3d-geometry-build checkout ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2
# 每个扩展的 setup.py 依赖工作目录，须分别进入目录构建。
(cd /tmp/sf3d-geometry-build/texture_baker && TORCH_CUDA_ARCH_LIST=8.0 python -m pip install --no-build-isolation .)
(cd /tmp/sf3d-geometry-build/uv_unwrapper && python -m pip install --no-build-isolation .)
```

只安装 `texture_baker` 和 `uv_unwrapper`，不安装上游 `sf3d` 神经网络包。
背景去除使用 rembg 的 U2Net；首次使用需准备 `~/.u2net/u2net.onnx`，离线机器应预先复制。
固定 tetrahedra 网格不提交到代码仓库。请先运行以下准备脚本；脚本固定上游版本并校验 SHA256，推理阶段只读取本地文件：

```bash
python scripts/sf3d/prepare_assets.py --model-path /path/to/stable-fast-3d
# 已有网格时避免重复下载：添加 --source /existing/160_tets.npz
```

网格保存到权重目录的 `assets/160_tets.npz`，也可通过部署 JSON 的 `tets_path` 指定。

本地权重目录只需 `config.yaml` 和 `model.safetensors`。DINOv2 和 OpenCLIP 视觉塔
已在根检查点中，不再按 config.yaml 的远程/本地预训练路径重复下载。训练检查点中
OpenCLIP 的文本塔被明确排除；其余网络参数严格加载。

```bash
MODEL_PATH=/path/to/stable-fast-3d \
INPUT_PATH=/path/to/one-image.jpg \
SAVE_RESULT_PATH=save_results/sf3d/object.glb \
bash scripts/platforms/nvidia/run_sf3d.sh
```

每次输入一张图，前景比例 0.85，条件图 512，isosurface=160，纹理 1024，remesh=none。
输出含 PBR 纹理、法线、roughness、metallic 的 GLB。`.glb.json` 记录单次端到端时延、
顶点数和面数。时延包含读图、背景去除、神经网络、提取网格、UV、纹理及序列化，
不含模型与 rembg session 初始化。没有运行 GT 子集或完整 GSO 基准。

默认启用 `deterministic_algorithms=true`，启动脚本在 Python 启动前导出
`CUBLAS_WORKSPACE_CONFIG=:4096:8`。原生 SF3D 的 CUDA 法线/切线归约在只固定 seed 时
仍可能改变 UV 和贴图；与原模型比较完整 GLB 时，双方应采用相同的确定性设置：

```python
torch.use_deterministic_algorithms(True)
```

该设置作用于整个推理进程。随机抽取的三个 GSO 首视图已验证：在相同设置下，
网格数组、UV、颜色/法线贴图和材质参数完全相同，导出的 GLB 文件 SHA256 也相同。

`weights/`、`infer/` 分别负责加载/结构与计算；相机调制 DINO、triplane 和材质网络均为
原生适配。公共 MM、LayerNorm、Conv2d、Attention 经注册表复用，FP32 权重配合 BF16
autocast，网格查询保持原生 FP32。GroupNorm、采样、三角化、UV 和纹理烘焙保留各自职责。
MLU、MetaX、Ascend 入口仅占位；A100 单卡以外的量化、并行与 offload 尚未实现。
