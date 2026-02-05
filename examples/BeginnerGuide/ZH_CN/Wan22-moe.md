# 从Wan2.2体验MoE

本文档包含 Wan2.2-T2V-A14B 和 Wan2.2-I2V-A14B 模型的使用示例。

## 准备环境

请参考[01.PrepareEnv](01.PrepareEnv.md)

## 开始运行

准备模型
```
# 从huggingface下载
hf download Wan-AI/Wan2.2-T2V-A14B --local-dir Wan-AI/Wan2.2-T2V-A14B
hf download Wan-AI/Wan2.2-I2V-A14B --local-dir Wan-AI/Wan2.2-I2V-A14B

# 下载蒸馏模型
hf download lightx2v/Wan2.2-Distill-Models --local-dir lightx2v/Wan2.2-Distill-Models
hf download lightx2v/Wan2.2-Distill-Loras --local-dir lightx2v/Wan2.2-Distill-Loras

# 下载 UMT5-XXL 文本编码器 FP8 量化权重
hf download lightx2v/Encoders --include "models_t5_umt5-xxl-enc-fp8.pth" --local-dir encoders/t5
```
我们提供三种方式，来运行 Wan2.2-T2V-A14B 和 Wan2.2-I2V-A14B 模型生成视频：

1. 运行脚本生成: 预设的bash脚本，可以直接运行，便于快速验证

    1.1 单卡推理

    1.2 单卡offload推理

    1.3 多卡并行推理

2. 启动服务生成: 先启动服务，再发请求，适合多次推理和实际的线上部署

    2.1 单卡推理

    2.2 单卡offload推理

    2.3 多卡并行推理

3. python代码生成: 用python代码运行，便于集成到已有的代码环境中

    3.1 单卡推理

    3.2 单卡offload推理

    3.3 多卡并行推理

### 运行脚本生成
```
git clone https://github.com/ModelTC/LightX2V.git

# 运行下面的脚本之前，需要将脚本中的lightx2v_path和model_path替换为实际路径
# 例如：lightx2v_path=/home/user/LightX2V
# 例如：model_path=/home/user/models/Wan-AI/Wan2.2-T2V-A14B
```

#### 1.1 单卡推理

Wan2.2-T2V-A14B
```
# model_path=/home/user/models/Wan-AI/Wan2.2-T2V-A14B
cd LightX2V/scripts/wan22
bash run_wan22_moe_t2v.sh

# 步数蒸馏模型 Lora
cd LightX2V/scripts/wan22/distill
bash run_wan22_moe_t2v_distill_lora_4step.sh
```
使用单张H100，运行时间及使用`watch -n 1 nvidia-smi`观测的峰值显存测试如下：
1. Wan2.2-T2V-A14B模型：Total Cost cost 823.492400 seconds；78940MiB
2. 步数蒸馏模型 Lora：Total Cost cost 43.418076 seconds；71810MiB

Wan2.2-I2V-A14B
```
cd LightX2V/scripts/wan22
bash run_wan22_moe_i2v.sh

# 步数蒸馏模型 Lora
cd LightX2V/scripts/wan22/distill
bash run_wan22_moe_i2v_distill_lora_4step.sh

# 步数蒸馏模型 merge Lora
bash run_wan22_moe_i2v_distill_model_4step.sh

# 步数蒸馏+FP8量化模型
bash run_wan22_moe_i2v_distill_fp8_4step.sh
```
使用单张H100，运行时间及观测的峰值显存测试如下：
1. Wan2.2-I2V-A14B模型：Total Cost cost 430.041016 seconds；79058MiB
2. 步数蒸馏模型 Lora：Total Cost cost 75.052103 seconds；77228MiB
3. 步数蒸馏模型 merge Lora：Total Cost cost 67.036318 seconds；79108MiB
4. 步数蒸馏+FP8量化模型：Total Cost cost 57.134773 seconds；47274MiB

注意：bash脚本中的model_path为pre-train原模型的路径；config文件中的lora_configs、dit_original_ckpt和dit_quantized_ckpt为所使用的蒸馏模型路径，需要修改为绝对路径，例如：/home/user/models/lightx2v/Wan2.2-Distill-Loras/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors

#### 1.2 单卡offload推理

如下修改 config 文件中的 cpu_offload，开启offload
```
    "cpu_offload": true
```
Wan2.2-T2V-A14B
```
# model_path=/home/user/models/Wan-AI/Wan2.2-T2V-A14B
cd LightX2V/scripts/wan22
bash run_wan22_moe_t2v.sh

# 步数蒸馏模型 Lora
cd LightX2V/scripts/wan22/distill
bash run_wan22_moe_t2v_distill_lora_4step.sh
```
使用单张H100，运行时间及使用`watch -n 1 nvidia-smi`观测的峰值显存测试如下：
1. Wan2.2-T2V-A14B模型：Total Cost cost 839.815357 seconds；51894MiB
2. 步数蒸馏模型 Lora：Total Cost cost 65.021112 seconds；44640MiB

Wan2.2-I2V-A14B
```
cd LightX2V/scripts/wan22
bash run_wan22_moe_i2v.sh

# 步数蒸馏模型 Lora
cd LightX2V/scripts/wan22/distill
bash run_wan22_moe_i2v_distill_lora_4step.sh

# 步数蒸馏模型 merge Lora
bash run_wan22_moe_i2v_distill_model_4step.sh

# 步数蒸馏+FP8量化模型
bash run_wan22_moe_i2v_distill_fp8_4step.sh
```
使用单张H100，运行时间及观测的峰值显存测试如下：
1. Wan2.2-I2V-A14B模型：Total Cost cost 477.574723 seconds；43666MiB
2. 步数蒸馏模型 Lora：Total Cost cost 100.342990 seconds；34426MiB
3. 步数蒸馏模型 merge Lora：Total Cost cost 88.365886 seconds；34426MiB
4. 步数蒸馏+FP8量化模型：Total Cost cost 69.006447 seconds；29250MiB

#### 1.3 多卡并行推理

Wan2.2-T2V-A14B
```
# 运行前需将CUDA_VISIBLE_DEVICES替换为实际用的GPU
# 同时config文件中的parallel参数也需对应修改，满足cfg_p_size * seq_p_size = GPU数目
cd LightX2V/scripts/dist_infer
bash run_wan22_moe_t2v_cfg_ulysses.sh

# 步数蒸馏模型 Lora
cd LightX2V/scripts/wan22/distill
bash run_wan22_moe_t2v_distill_lora_4step_cfg_ulysses.sh
```
使用8张H100，运行时间及观测的峰值显存测试如下：
1. Wan2.2-T2V-A14B模型：Total Cost cost 152.303705 seconds；74370MiB
2. 步数蒸馏模型 Lora：Total Cost cost 74.180393 seconds；71220MiB

Wan2.2-I2V-A14B
```
cd LightX2V/scripts/dist_infer
bash run_wan22_moe_i2v_cfg_ulysses.sh

# 步数蒸馏模型 Lora
cd LightX2V/scripts/wan22/distill
bash run_wan22_moe_i2v_distill_lora_4step_cfg_ulysses.sh

# 步数蒸馏模型 merge Lora
cd LightX2V/scripts/wan22/distill
bash run_wan22_moe_i2v_distill_model_4step_cfg_ulysses.sh

# 步数蒸馏+FP8量化模型
cd LightX2V/scripts/wan22/distill
bash run_wan22_moe_i2v_distill_fp8_4step_cfg_ulysses.sh
```
使用8张H100，运行时间及观测的峰值显存测试如下：
1. Wan2.2-I2V-A14B模型：Total Cost cost 258.464609 seconds；74510MiB
2. 步数蒸馏模型 Lora：Total Cost cost 55.125003 seconds；73776MiB
3. 步数蒸馏模型 merge Lora：Total Cost cost 41.272628 seconds；73096MiB
4. 步数蒸馏+FP8量化模型：Total Cost cost 31.315791 seconds；42120MiB

解释细节
run_wan22_moe_t2v.sh脚本内容如下
```
#!/bin/bash

# set path firstly
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls wan2.2_moe \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan22/wan_moe_t2v.json \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan22_moe_t2v.mp4

```
`export CUDA_VISIBLE_DEVICES=0` 表示使用0号显卡

`source ${lightx2v_path}/scripts/base/base.sh` 设置一些基础的环境变量

`--model_cls wan2.2_moe` 表示使用wan2.2_moe模型

`--task t2v` 表示使用t2v任务

`--model_path` 表示模型的路径

`--config_json` 表示配置文件的路径

`--prompt` 表示提示词

`--negative_prompt` 表示负向提示词

`--save_result_path` 表示保存结果的路径

wan_moe_t2v.json 内容如下
```
{
    "infer_steps": 40,
    "target_video_length": 81,
    "text_len": 512,
    "target_height": 720,
    "target_width": 1280,
    "self_attn_1_type": "flash_attn3",
    "cross_attn_1_type": "flash_attn3",
    "cross_attn_2_type": "flash_attn3",
    "sample_guide_scale": [
        4.0,
        3.0
    ],
    "sample_shift": 12.0,
    "enable_cfg": true,
    "cpu_offload": false,
    "offload_granularity": "model",
    "t5_cpu_offload": false,
    "vae_cpu_offload": false,
    "boundary": 0.875
}

```
`infer_steps` 表示推理的步数

`target_video_length` 表示目标视频的帧数

`target_height` 表示目标视频的高度

`target_width` 表示目标视频的宽度

`self_attn_1_type`, `cross_attn_1_type`, `cross_attn_2_type` 表示wan2.2模型内部的三个注意力层的算子的类型，这里使用flash_attn3，仅限于Hopper架构的显卡(H100, H20等)，其他显卡可以使用flash_attn2进行替代

`sample_guide_scale` 表示CFG引导尺度，两个值分别对应高噪声阶段和低噪声阶段

`enable_cfg` 表示是否启用cfg，这里设置为true，表示会推理两次，第一次使用正向提示词，第二次使用负向提示词，这样可以得到更好的效果，但是会增加推理时间，如果是已经做了CFG蒸馏的模型，这里就可以设置为false

`cpu_offload` 表示是否启用cpu offload，这里设置为false，表示不启用cpu offload，如果显存不足，则需要开启cpu_offload。

`t5_cpu_offload`、`vae_cpu_offload` 表示是否卸载 T5 编码器到 CPU 、是否卸载 VAE 解码器到 CPU。若是显存紧张，可设置成true，但是会导致推理时间变长

`boundary` 表示高低噪声阶段的分界比例

wan_moe_t2v_distill_lora.json内容如下：
```
{
    "infer_steps": 4,
    "target_video_length": 81,
    "text_len": 512,
    "target_height": 480,
    "target_width": 832,
    "self_attn_1_type": "flash_attn3",
    "cross_attn_1_type": "flash_attn3",
    "cross_attn_2_type": "flash_attn3",
    "sample_guide_scale": [
        4.0,
        3.0
    ],
    "sample_shift": 5.0,
    "enable_cfg": false,
    "cpu_offload": true,
    "offload_granularity": "model",
    "t5_cpu_offload": false,
    "vae_cpu_offload": false,
    "boundary_step_index": 2,
    "denoising_step_list": [
        1000,
        750,
        500,
        250
    ],
    "lora_configs": [
        {
            "name": "high_noise_model",
            "path": "lightx2v/Wan2.2-Distill-Loras/wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors",
            "strength": 1.0
        },
        {
            "name": "low_noise_model",
            "path": "lightx2v/Wan2.2-Distill-Loras/wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors",
            "strength": 1.0
        }
    ]
}
```
`boundary_step_index` 表示噪声阶段分界索引，切换高噪声模型和低噪声模型

`lora_configs`: 包含两个LoRA适配器，高噪声模型负责生成视频的高频细节和结构，低噪声模型负责平滑噪声和优化全局一致性。这种分工使得模型能够在不同阶段专注于不同的生成任务，从而提升整体性能。需要修改成绝对路径。

wan_moe_i2v_distill_model.json内容如下
```
{
    "infer_steps": 4,
    "target_video_length": 81,
    "text_len": 512,
    "target_height": 720,
    "target_width": 1280,
    "self_attn_1_type": "flash_attn3",
    "cross_attn_1_type": "flash_attn3",
    "cross_attn_2_type": "flash_attn3",
    "sample_guide_scale": [
        3.5,
        3.5
    ],
    "sample_shift": 5.0,
    "enable_cfg": false,
    "cpu_offload": true,
    "offload_granularity": "block",
    "t5_cpu_offload": false,
    "vae_cpu_offload": false,
    "use_image_encoder": false,
    "boundary_step_index": 2,
    "denoising_step_list": [
        1000,
        750,
        500,
        250
    ],
    "high_noise_original_ckpt": "lightx2v/Wan2.2-Distill-Models/wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors",
    "low_noise_original_ckpt": "lightx2v/Wan2.2-Distill-Models/wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors"
}

```
`high_noise_original_ckpt` 表示高噪声阶段使用的蒸馏模型路径，需要修改成绝对路径

`low_noise_original_ckpt` 表示低噪声阶段使用的蒸馏模型路径，需要修改成绝对路径

wan_moe_i2v_distill_quant.json内容如下：
```
{
    "infer_steps": 4,
    "target_video_length": 81,
    "text_len": 512,
    "target_height": 720,
    "target_width": 1280,
    "self_attn_1_type": "flash_attn3",
    "cross_attn_1_type": "flash_attn3",
    "cross_attn_2_type": "flash_attn3",
    "sample_guide_scale": [
        3.5,
        3.5
    ],
    "sample_shift": 5.0,
    "enable_cfg": false,
    "cpu_offload": false,
    "offload_granularity": "block",
    "t5_cpu_offload": false,
    "vae_cpu_offload": false,
    "use_image_encoder": false,
    "boundary_step_index": 2,
    "denoising_step_list": [
        1000,
        750,
        500,
        250
    ],
    "dit_quantized": true,
    "dit_quant_scheme": "fp8-sgl",
    "t5_quantized": true,
    "t5_quant_scheme": "fp8-sgl",
    "t5_quantized_ckpt": "encoders/t5/models_t5_umt5-xxl-enc-fp8.pth",
    "high_noise_quantized_ckpt": "lightx2v/Wan2.2-Distill-Models/wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors",
    "low_noise_quantized_ckpt": "lightx2v/Wan2.2-Distill-Models/wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"
}

```
`dit_quantized`	表示是否启用 DIT 量化，设置为True表示对模型核心的 DIT 模块做量化处理

`dit_quant_scheme` 表示 DIT 量化方案，指定量化类型为 "fp8-sgl"（fp8-sgl表示使用sglang的fp8 kernel进行推理）

`t5_quantized`  表示是否启用 T5 量化

`t5_quantized_ckpt` 表示使用的 t5 量化模型路径，需要修改成绝对路径

`high_noise_quantized_ckpt` 表示高噪声阶段使用的步数蒸馏+FP8量化模型路径，需要修改成绝对路径

`low_noise_quantized_ckpt` 表示低噪声阶段使用的蒸馏+FP8量化模型路径,需要修改成绝对路径

### 2. 启动服务生成

#### 2.1单卡推理
```
# 运行下面的脚本之前，需要将脚本中的lightx2v_path、model_path以及config_json替换为实际路径
# 例如：lightx2v_path=/home/user/LightX2V
# 例如：model_path=/home/user/models/Wan-AI/Wan2.2-T2V-A14B
# 例如：config_json ${lightx2v_path}/configs/wan22/wan_moe_t2v.json
```
启动服务
```
cd LightX2V/scripts/server

# 切换model_path和config_json路径体验不同模型
# 1. Wan2.2-T2V-A14B
# model_path=/home/user/models/Wan-AI/Wan2.2-T2V-A14B
# --model_cls wan2.2_moe \
# --task t2v \
# --config_json ${lightx2v_path}/configs/wan22/wan_moe_t2v.json \
# 2. Wan2.2-I2V-A14B
# model_path=/home/user/models/Wan-AI/Wan2.2-T2V-A14B
# --model_cls wan2.2_moe \
# --task i2v \
# --config_json ${lightx2v_path}/configs/wan22/wan_moe_i2v.json \

bash start_server.sh
```
向服务端发送请求

此处需要打开第二个终端作为用户
```
cd LightX2V/scripts/server

# 此时生成视频，url = "http://localhost:8000/v1/tasks/video/"
python post.py
```
发送完请求后，可以在服务端看到推理的日志

#### 2.2 单卡offload推理

如下修改 config 文件中的 cpu_offload，开启offload
```
    "cpu_offload": true,
```
启动服务
```
cd LightX2V/scripts/server

bash start_server.sh
```
向服务端发送请求
```
cd LightX2V/scripts/server

# 此时生成视频，url = "http://localhost:8000/v1/tasks/video/"
python post.py
```

#### 2.3 多卡并行推理

启动服务
```
cd LightX2V/scripts/server
# 1. Wan2.2-T2V-A14B
# model_path=/home/user/models/Wan-AI/Wan2.2-T2V-A14B
# --model_cls wan2.2_moe \
# --task t2v \
# --config_json ${lightx2v_path}/configs/dist_infer/wan22_moe_t2v_cfg_ulysses.json \
# 2. Wan2.2-I2V-A14B
# model_path=/home/user/models/Wan-AI/Wan2.2-T2V-A14B
# --model_cls wan2.2_moe \
# --task i2v \
# --config_json ${lightx2v_path}/configs/dist_infer/wan22_moe_i2v_cfg_ulysses.json \

bash start_server_cfg_ulysses.sh
```
向服务端发送请求

```
cd LightX2V/scripts/server

# 此时生成视频，url = "http://localhost:8000/v1/tasks/video/"
python post.py
```
运行时间及观测的每张卡峰值显存测试如下：
1. 单卡推理：Run DiT cost 795.959228 seconds；RUN pipeline cost 796.483249 seconds；78940MiB
2. 单卡offload推理：Run DiT cost 797.796916 seconds；RUN pipeline cost 798.422972 seconds；51894MiB
3. 多卡并行推理：Run DiT cost 118.976583 seconds；RUN pipeline cost 119.531743 seconds；74370MiB

解释细节

start_server.sh脚本内容如下
```
#!/bin/bash

# set path firstly
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh


# Start API server with distributed inference service
python -m lightx2v.server \
--model_cls wan2.1 \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan/wan_t2v.json \
--host 0.0.0.0 \
--port 8000

echo "Service stopped"

```
`--host 0.0.0.0`和`--port 8000`，表示服务起在本机ip的8000端口上

post.py内容如下
```
import requests
from loguru import logger

if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/video/"

    message = {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
        "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "image_path": "",
        "seed": 42,
        "save_result_path": "./cat_boxing_seed42.mp4",
    }

    logger.info(f"message: {message}")

    response = requests.post(url, json=message)

    logger.info(f"response: {response.json()}")

```
`url = "http://localhost:8000/v1/tasks/video/" `表示向本机ip的8000端口上，发送一个视频生成任务。如果是图像生成任务，url需改成 url = "http://localhost:8000/v1/tasks/image/"

`message字典` 表示向服务端发送的请求的内容，其中`seed`若不指定，每次发送请求会随机生成一个`seed`，`save_result_path`若不指定也会生成一个和任务id一致命名的文件

### python代码生成

#### 3.1单卡推理

```
cd LightX2V/examples/wan

# 修改model_path、save_result_path、config_json为实际的绝对路径

PYTHONPATH=/home/user/LightX2V python wan_moe_t2v.py
```
注意1：设置运行中的参数中，推荐使用传入config_json的方式，用来和前面的运行脚本生成视频和启动服务生成视频进行超参数对齐

注意2：PYTHONPATH的路径需为绝对路径

#### 3.2 单卡offload推理

如下修改 config 文件中的 cpu_offload，开启offload
```
    "cpu_offload": true
```
```
cd LightX2V/examples/wan

PYTHONPATH=/home/user/LightX2V python wan_t2v.py
```

#### 3.3 多卡并行推理
```
cd LightX2V/examples/wan
# 代码中需将config_json改成：LightX2V/configs/dist_infer/wan22_moe_t2v_cfg_ulysses.json

PROFILING_DEBUG_LEVEL=2 PYTHONPATH=/home/user/LightX2V torchrun --nproc_per_node=8 wan_moe_t2v.py
```
运行时间及观测的每张卡峰值显存测试如下：
1. 单卡推理：Run DiT cost 796.039108 seconds；RUN pipeline cost 796.525144 seconds；78944MiB
2. 单卡offload推理：Run DiT cost 797.845191 seconds；RUN pipeline cost 798.353524 seconds；51898MiB
3. 多卡并行推理：Run DiT cost 119.016977 seconds；RUN pipeline cost 119.522429 seconds；74370MiB

解释细节

wan_moe_t2v.py内容如下
```
from lightx2v import LightX2VPipeline

# Initialize pipeline for Wan2.2 T2V task
pipe = LightX2VPipeline(
    model_path="/home/user/models/Wan-AI/Wan2.2-T2V-A14B",
    model_cls="wan2.2_moe",
    task="t2v",
)

# Alternative: create generator from config JSON file
pipe.create_generator(config_json="/home/user/LightX2V/configs/wan22/wan_moe_t2v.json")

# Create generator with specified parameters
#pipe.create_generator(
#   attn_mode="sage_attn2",
#   infer_steps=50,
#    height=480,  # Can be set to 720 for higher resolution
#    width=832,  # Can be set to 1280 for higher resolution
#    num_frames=81,
#    guidance_scale=5.0,
#    sample_shift=5.0,
#)

seed = 42
prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
save_result_path = "/home/user/LightX2V/save_results/output_lightx2v_wan22_moe_t2v.mp4"

pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)

```
注意1：需要修改 model_path、config_json、save_result_path 为实际的路径

注意2：设置运行中的参数中，推荐使用传入config_json的方式，用来和前面的运行脚本生成视频和启动服务生成视频进行超参数对齐
