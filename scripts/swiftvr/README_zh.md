# SwiftVR 部署指南

[English](README.md) | [简体中文](README_zh.md)

[技术博客](https://light-ai.top/LightX2V-BLOG/posts/SwiftVR/)

SwiftVR 支持图像与视频的恢复和超分辨率，统一使用 `--model_cls swiftvr --task sr`。

## 1. 模型下载与转换

完成环境安装后，在 LightX2V 仓库根目录下载并转换官方权重 [H-oliday/SwiftVR](https://huggingface.co/H-oliday/SwiftVR)：

```bash
hf download H-oliday/SwiftVR --local-dir /path/to/SwiftVR
python tools/convert/examples/convert_swiftvr.py \
  --source /path/to/SwiftVR \
  --output /path/to/SwiftVR_lightx2v
```

## 2. 环境安装

根据 GPU 型号选择 [LightX2V Docker 镜像](https://hub.docker.com/r/lightx2v/lightx2v/tags)。在宿主机的工作目录中拉取对应镜像：

```bash
# H100, A100等
docker pull lightx2v/lightx2v:26062001

# RTX 5090
docker pull lightx2v/lightx2v:26062001-cu130-5090-fix-260921
```

使用所选镜像启动容器：

```bash
docker run --gpus all -itd --ipc=host --name [容器名] -v [挂载设置] --entrypoint /bin/bash [镜像id]
```

在容器内下载源码并安装依赖：

```bash
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
python -m pip install "av>=17" decord imageio-ffmpeg huggingface_hub
```

自行搭建环境时，可参考[安装指南](../../docs/ZH_CN/source/getting_started/quickstart.md)。

## 3. 离线推理

运行前，修改对应脚本中的 `lightx2v_path`（仓库路径）、`model_path`（转换后的模型路径）、`config_path`（配置路径）和 `CUDA_VISIBLE_DEVICES`（GPU 编号），并设置输入路径与输出路径 `output_path`。

根据 GPU 选择 `configs/swiftvr/` 下的配置：

| GPU | 配置 |
| --- | --- |
| H100 | `h100/swiftvr.json` |
| A800 | `a800/swiftvr.json` |
| RTX 5090 | `5090/swiftvr.json` |

启用编译时，在配置文件中添加 `"use_compile": true`，或者使用对应的 `swiftvr_compile.json`。

```bash
# 图像超分
bash scripts/swiftvr/inference/run_swiftvr_image_sr.sh

# 视频超分
bash scripts/swiftvr/inference/run_swiftvr_video_sr.sh
```

每次仅指定 `--image_path` 或 `--video_path` 中的一项，输出文件分别使用 `.png` 或 `.mp4` 格式。默认放大 2 倍（`--sr_ratio 2`）；可修改倍率，或使用 `--size 1440 2560` 指定输出尺寸（高、宽）。两者同时设置时，`size` 优先。

多卡视频推理可参考 `h100/swiftvr_parallel.json`。四卡推理需设置 `parallel.chunk_p_size=4`、`CUDA_VISIBLE_DEVICES=0,1,2,3` 和 `--nproc_per_node 4`，然后运行：

```bash
bash scripts/swiftvr/inference/run_swiftvr_video_sr_4gpu.sh
```

## 4. 服务化部署与 API 调用

修改对应服务启动脚本中的仓库路径、模型路径、配置路径和 GPU 编号，然后选择单卡或四卡方式启动：

```bash
# 单卡，默认使用 H100 编译配置
bash scripts/swiftvr/server/start_server.sh

# 四卡，按上文对齐并行配置
bash scripts/swiftvr/server/start_server_4gpu.sh
```

默认端口为 `8000`，同一个 `sr` 服务支持图像和视频超分。

发送请求前，修改客户端脚本中的 `url` 和 `message`：输入路径填写服务端可访问的绝对路径，`save_result_path` 指定服务端输出文件。使用 `sr_ratio` 设置缩放倍率，或使用 `size: [高, 宽]` 指定输出尺寸，规则与离线推理相同。

服务启动及预热完成后，在另一终端执行对应的请求脚本：

```bash
# 图像超分
python scripts/swiftvr/server/post_image.py

# 视频超分
python scripts/swiftvr/server/post_video.py
```
