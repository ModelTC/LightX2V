# Stable Fast 3D → LightX2V (A100)

目标权重为 `stabilityai/stable-fast-3d`，revision
`f0c9a8ffd62cb1bbc8a7a53c9f87a0be1b6be778`。
代码移植自 `stability-ai/stable-fast-3d` commit
`ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2`，模型目录保留其许可证。
运行时不调用上游 SF3D、transformers 或 OpenCLIP 的神经网络实现。

先执行 `bash scripts/sf3d/install_a100.sh`。另外需要 CUDA toolkit、C++ 编译器
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
已在根检查点中，不再按 config.yaml 的远程/本地预训练路径重复下载。适配器严格加载单图网格生成所需的网络参数。

```bash
MODEL_PATH=/path/to/stable-fast-3d \
INPUT_PATH=/path/to/one-image.jpg \
SAVE_RESULT_PATH=save_results/sf3d/object.glb \
bash scripts/platforms/nvidia/run_sf3d.sh
```

每次输入一张图，前景比例 0.85，条件图 512，isosurface=160，纹理 1024，remesh=none。
输出为包含 PBR 纹理、法线、roughness 和 metallic 的 GLB 文件。

默认开启确定性推理，配合 SEED 控制生成结果。启动脚本会在 Python 启动前设置
CUBLAS_WORKSPACE_CONFIG；使用 Python 接口时，此设置及确定性算法选项作用于整个进程。

支持 NVIDIA A100 单卡。MLU、MetaX、Ascend 的脚本和配置为占位入口，执行时会提示未实现。

推理精度：DINOv2、OpenCLIP、相机嵌入、双流 Transformer 和三平面上采样使用 BF16；归一化统计、材质分布参数、三平面采样坐标、几何/材质场解码、等值面提取、UV 和纹理烘焙保留 FP32。纹理烘焙扩展要求 FP32 输入。
