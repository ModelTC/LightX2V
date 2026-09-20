# Qwen-Image-2.1 使用说明

## 1. 模型下载

官方模型权重：[Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)。下载命令：

```bash
hf download Qwen/Qwen-Image-2.1 --local-dir Qwen/Qwen-Image-2.1
```

## 2. 环境安装

使用 [LightX2V Docker 镜像](https://hub.docker.com/r/lightx2v/lightx2v/tags)。在宿主机的工作目录中拉取镜像并启动容器：

```bash
docker pull lightx2v/lightx2v:26062001
docker run --gpus all -itd --ipc=host --name [容器名] -v [挂载设置] --entrypoint /bin/bash [镜像id]
```

在容器内下载并安装 LightX2V 源码，无需安装即可运行：

```bash
git clone https://github.com/ModelTC/LightX2V.git
```

后续命令均在容器内的 LightX2V 仓库根目录执行。

## 3. 离线推理

先修改对应 bash 脚本中的 `lightx2v_path`（仓库路径）、`model_path`（模型根目录）和 `CUDA_VISIBLE_DEVICES`（GPU 编号）。

```bash
# 文生图
bash scripts/qwen_image_21/qwen_image_21_t2i.sh

# 图生图
bash scripts/qwen_image_21/qwen_image_21_i2i.sh
```

直接在脚本中修改提示词 `--prompt`、参考图 `--image_path`、尺寸 `--size`、种子 `--seed` 和输出路径 `--save_result_path`。

`--size` 顺序为高、宽，均须为 32 的正整数倍。默认生成 1024×1024 图片。

## 4. 服务化部署与 API 调用

先修改 `server/start_server.sh` 中的仓库路径、模型路径和 GPU 编号，然后启动：

```bash
bash scripts/qwen_image_21/server/start_server.sh
```

默认端口为 `8000`。服务只加载一套权重，启动时无需指定 `task`，同一个服务支持文生图和图生图。

服务启动完成后，新开终端进入同一容器并发送请求：

```bash
docker exec -it -w /workspace/LightX2V lightx2v-qwen21 bash

# 文生图
python scripts/qwen_image_21/server/post_t2i.py

# 图生图
python scripts/qwen_image_21/server/post_i2i.py
```

直接修改代码中的 `url`、`message`、`output_path`；图生图还需设置 `image_path`。两个脚本已分别设置 `task: "t2i"` 和 `task: "i2i"`。

图生图脚本读取客户端本地图片，将其编码为 Base64 后，通过请求的 `image_path` 字段上传。文生图和图生图均等待生成完成后，接收服务端返回的 PNG 二进制数据，并保存到客户端 `output_path`。
