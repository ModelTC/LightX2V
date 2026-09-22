# Qwen-Image-2.1 部署指南

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

### 3.1 通用案例

先修改对应 bash 脚本中的 `lightx2v_path`（仓库路径）、`model_path`（模型根目录）和 `CUDA_VISIBLE_DEVICES`（GPU 编号）。

```bash
# 文生图
bash scripts/qwen_image_21/qwen_image_21_t2i.sh

# 图生图
bash scripts/qwen_image_21/qwen_image_21_i2i.sh
```

直接在脚本中修改提示词 `--prompt`、参考图 `--image_path`、尺寸 `--size`、种子 `--seed` 和输出路径 `--save_result_path`。

使用 `--size 高 宽`。宽高均须不小于 32，非 32 倍数会自动向下对齐。

图生图可删除脚本中的 `--size` 参数，按参考图比例自动确定输出尺寸。总像素数约为配置中的 `resolution²`（`resolution` 默认为 `1024`），宽高就近对齐到 32 的倍数。

默认配置已经启用通用优化路径：FlashAttention3、FlashInfer RoPE、Triton LayerNorm 与 modulation、融合 QK RMSNorm、融合 Transformer block 算子、关闭 CFG 的推理路径及 OpenCV 结果保存。

### 3.2 RTX 5090 FP8 案例

该案例使用 FP8 linear 和 FP16 累加。首先转换 DiT 权重：

```bash
python tools/convert/converter.py \
    --source /path/to/Qwen-Image-2.1/transformer \
    --output /path/to/Qwen-Image-2.1-fp8-f16-accum \
    --output_name qwen_image_21_fp8_f16_accum \
    --model_type qwen_image_21_dit \
    --quantization_profile qwen-image-21-fp8-f16-accum \
    --quantized --linear_type fp8 --device cuda:0 --single_file
```

将所选 JSON 配置中的 `dit_quantized_ckpt` 指向转换得到的 `.safetensors` 文件，然后设置脚本中的 `lightx2v_path` 和 `model_path`。`model_path` 仍须指向原始模型目录。FP16 累加路径要求 lightx2v-kernel 提供 SM120 算子。

```bash
# 文生图
bash scripts/qwen_image_21/qwen_image_21_t2i_fp8_f16_accum_5090.sh

# 图生图
bash scripts/qwen_image_21/qwen_image_21_i2i_fp8_f16_accum_5090.sh
```

RTX 5090 配置保留适用的通用优化，并使用：

- 条件编码器阶段 CPU offload；
- qmax 14 的 FP8 DiT block linear 权重、qmax 7 的动态 FP8 激活及 FP16 累加；
- dense SageAttention2。

| GPU | 任务 | 输出分辨率 | 端到端耗时 |
| --- | --- | ---: | ---: |
| RTX 5090 | T2I | 1024×1024 | **5.930 s** |
| RTX 5090 | I2I | 1024×1024 | **7.144 s** |

两类任务均使用当前代码，测试条件为 40 steps、seed 42、关闭 CFG，连续执行三次请求并取中位数。I2I 使用单张 1024×1024 参考图，输出尺寸为 1024×1024。端到端耗时包含输入编码、condition-KV prefill、去噪、VAE 解码、后处理和 PNG 保存，不包含模型加载及 Runner 的一次性初始化。

### 3.3 AMD ROCm 显卡（Radeon AI PRO R9700 / Radeon PRO W7900）

上面 5090 那套（FlashAttention3、FlashInfer RoPE、CUTLASS FP8、SageAttention2 的 CUDA kernel）都是 CUDA 专用。下面的配置让 Qwen-Image-2.1 跑在 AMD RDNA 显卡上，走 torch 算子推理路径（`torch_sdpa`、`torch_real_rope`、torch LayerNorm/modulation），配合 ROCm 可用的量化、`torch.compile` 和 SageAttention2。它们**在加载时直接量化官方 BF16 权重**——无需 `tools/convert` 转换、无需 `dit_quantized_ckpt`。

验证环境为 `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` 镜像（PyTorch 2.10、Triton 3.6），transformers 使用 4.57.x（`Qwen3VLProcessor` 需要较新版本；请固定到你实测过的版本，而非只写下限）。安装该镜像未自带的运行时依赖：

```bash
pip install "transformers>=4.57" diffusers ftfy accelerate loguru omegaconf einops \
    qtorch langdetect tqdm imageio imageio-ffmpeg opencv-python-headless comfy-kitchen \
    peft gguf prometheus_client fastapi uvicorn pydantic aiohttp pyzmq python-multipart \
    PyJWT jsonschema sageattention
```

通过 `scripts/platforms/amd_rocm/` 下的平台脚本运行——脚本里会 `export PLATFORM=amd_rocm` 并 source 公共环境，从而激活 `lightx2v_platform/` 下的 AMD ROCm 平台后端（注册 ROCm 的 FP8/INT8 GEMM，并自动应用 SageAttention 的 Triton 规避）。单卡即可——条件编码器走 CPU offload，且该路径**无需编译 aiter**。

先在脚本顶部设置 `lightx2v_path` 和 `model_path`，然后运行：

```bash
# R9700（FP8）
bash scripts/platforms/amd_rocm/qwen_image_21_r9700_t2i.sh
# W7900（INT8）
bash scripts/platforms/amd_rocm/qwen_image_21_w7900_t2i.sh
```

每个脚本默认选该卡最快的配置；改脚本里的 `--config_json` 可切换到其它配置。各配置（均在 `configs/platforms/amd_rocm/` 下）：

| 显卡（架构） | 配置 | 内容 |
| --- | --- | --- |
| R9700（gfx1201 / RDNA4） | `qwen_image_21_r9700_fp8.json` | FP8（`torch._scaled_mm`）+ `torch.compile` + SageAttention2 |
| W7900（gfx1100 / RDNA3） | `qwen_image_21_w7900_int8.json` | INT8（`torch._int_mm`）+ `torch.compile` + SageAttention2 |
| 通用 | `qwen_image_21_bf16.json` | BF16 eager 基线 |

`dit_quant_scheme` 为 `fp8-rocm`（R9700）或 `int8-rocm`（W7900），由 `lightx2v_platform/ops/mm/amd_rocm/` 下的 AMD ROCm 平台算子注册：权重 per-channel 对称量化 + 激活 per-token 动态量化，均在加载时从 BF16 量化得到。VAE 走原生卷积——AMD ROCm 平台会关闭 cuDNN/MIOpen，从而绕开 gfx1201 上观察到的非确定性 NaN 卷积路径。SageAttention/Inductor 在 gfx1201 / gfx1100 上所需的 Triton `num_stages` 规避会自动装载，且仅在真正构造 SageAttention2 backend 时才生效。想拿画质换速度，可把配置里的 `infer_steps` 调小（如 25）或传 `--infer_steps 25`。

| 显卡 | 量化 | 步数 | 端到端 |
| --- | --- | ---: | ---: |
| R9700 | FP8 | 40 | ~24 s |
| R9700 | FP8 | 25 | ~17 s |
| W7900 | INT8 | 40 | ~46 s |
| W7900 | INT8 | 25 | ~29 s |

测试条件：1024×1024、seed 42、关闭 CFG、单卡、warmup 后稳态（3 次取中位数）。端到端包含文本编码、condition-KV prefill、去噪、VAE 解码与 PNG 保存，不含模型加载与一次性初始化。在所测试的 prompt/seed 上，25 步相比 40 步未观察到明显画质退化——这不是系统性画质评估。编译为一次性 warmup 开销（约 1.5–3 分钟）；`TORCHINDUCTOR_COMPILE_THREADS=1` 在 SageAttention 路径上会自动设置。

## 4. 服务化部署与 API 调用

先修改 `server/start_server.sh` 中的仓库路径、模型路径和 GPU 编号，然后启动：

```bash
bash scripts/qwen_image_21/server/start_server.sh
```

默认端口为 `8000`。服务只加载一套权重，启动时无需指定 `task`，同一个服务支持文生图和图生图。

服务启动完成后，在同一容器新开终端并发送请求：

```bash
# 文生图
python scripts/qwen_image_21/server/post_t2i.py

# 图生图
python scripts/qwen_image_21/server/post_i2i.py
```

直接修改代码中的 `url`、`message`、`output_path`；图生图还需设置 `image_path`。两个脚本已分别设置 `task: "t2i"` 和 `task: "i2i"`。

服务请求使用相同的尺寸规则。图生图需自动确定尺寸时，删除 `message` 中的 `size` 字段。

图生图脚本读取客户端本地图片，将其编码为 Base64 后，通过请求的 `image_path` 字段上传。文生图和图生图均等待生成完成后，接收服务端返回的 PNG 二进制数据，并保存到客户端 `output_path`。
