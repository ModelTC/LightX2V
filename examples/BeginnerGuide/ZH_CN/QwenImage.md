# 从Qwen Image体验T2I与I2I

本文档包含 Qwen Image 和 Qwen Image Edit 模型的使用示例。

其中文生图模型使用的是Qwen-Image-2512，图像编辑模型使用的是Qwen-Image-Edit-2511，都为目前最新的模型。

## 准备环境

请参考[01.PrepareEnv](01.PrepareEnv.md)

## 开始运行

准备模型

```
# 从huggingface下载
# 推理2512文生图原始模型
hf download Qwen/Qwen-Image-2512 --local-dir Qwen/Qwen-Image-2512

# 推理2512文生图步数蒸馏模型
hf download lightx2v/Qwen-Image-2512-Lightning --local-dir Qwen/Qwen-Image-2512-Lightning

# 推理2511图像编辑原始模型
hf download Qwen/Qwen-Image-Edit-2511 --local-dir Qwen/Qwen-Image-2511

# 推理2511图像编辑步数蒸馏模型
hf download lightx2v/Qwen-Image-Edit-2511-Lightning --local-dir Qwen/Qwen-Image-2511-Lightning
```

我们提供三种方式，来运行 Qwen Image 模型生成图片：

1. 运行脚本生成图片: 预设的bash脚本，可以直接运行，便于快速验证
2. 启动服务生成图片: 先启动服务，再发请求，适合多次推理和实际的线上部署
3. python代码生成图片: 用python代码运行，便于集成到已有的代码环境中

### 运行脚本生成图片

```
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V/scripts/qwen_image

# 运行下面的脚本之前，需要将脚本中的lightx2v_path和model_path替换为实际路径
# 例如：lightx2v_path=/home/user/LightX2V
# 例如：model_path=/home/user/models/Qwen/Qwen-Image-2511
```

文生图模型
```
# 推理2512文生图原始模型，默认是50步
bash qwen_image_t2i_2512.sh

# 推理2512文生图步数蒸馏模型，默认是8步，需要修改config_json文件中的lora_configs的路径
bash qwen_image_t2i_2512_distill.sh

# 推理2512文生图步数蒸馏+FP8量化模型，默认是8步，需要修改config_json文件中的dit_quantized_ckpt的路径
bash qwen_image_t2i_2512_distill_fp8.sh
```

注意1：在qwen_image_t2i_2512_distill.sh、qwen_image_t2i_2512_distill_fp8.sh脚本中，model_path与qwen_image_t2i_2512.sh保持一致，都为Qwen-Image-2512模型的本地路径

注意2：需要修改的config_json文件在LightX2V/configs/qwen_image中，lora_configs、dit_quantized_ckpt分别为所使用蒸馏模型的本地路径


图像编辑模型
```
# 推理2511图像编辑原始模型，默认是40步
bash qwen_image_i2i_2511.sh

# 推理2511图像编辑步数蒸馏模型，默认是8步，需要修改config_json文件中的lora_configs的路径
bash qwen_image_i2i_2511_distill.sh

# 推理2511图像编辑步数蒸馏+FP8量化模型，默认是8步，需要修改config_json文件中的dit_quantized_ckpt的路径
bash qwen_image_i2i_2511_distill_fp8.sh
```
注意1：bash脚本的model_path都为Qwen-Image-2511路径，config_json文件中需要修改的路径分别为所使用蒸馏模型的路径

注意2：需要修改bash脚本中的图片路径image_path，可以传入你自己的图片来测试模型

解释细节

qwen_image_t2i_2512.sh脚本内容如下：
```
#!/bin/bash

# set path firstly
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls qwen_image \
--task t2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/qwen_image/qwen_image_t2i_2512.json \
--prompt 'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition, Ultra HD, 4K, cinematic composition.' \
--negative_prompt " " \
--save_result_path ${lightx2v_path}/save_results/qwen_image_t2i_2512.png \
--seed 42
```
`source ${lightx2v_path}/scripts/base/base.sh` 设置一些基础的环境变量

`--model_cls qwen_image` 表示使用qwen_image模型

`--task t2i` 表示使用t2i任务

`--model_path` 表示模型的路径

`--config_json` 表示配置文件的路径

`--prompt` 表示提示词

`--negative_prompt` 表示负向提示词

qwen_image_t2i_2512.json内容如下
```
{
    "infer_steps": 50,
    "aspect_ratio": "16:9",
    "prompt_template_encode": "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    "prompt_template_encode_start_idx": 34,
    "attn_type": "flash_attn3",
    "enable_cfg": true,
    "sample_guide_scale": 4.0
}
```
`infer_steps` 表示推理的步数

`aspect_ratio` 表示目标图片的宽高比

`prompt_template_encode` 表示提示词编码的模板

`prompt_template_encode_start_idx` 表示提示词模板的有效起始索引

`attn_type` 表示模型内部的注意力层算子的类型，这里使用flash_attn3，仅限于Hopper架构的显卡(H100, H20等)，其他显卡可以使用flash_attn2进行替代

`enable_cfg` 表示是否启用cfg，这里设置为true，表示会推理两次，第一次使用正向提示词，第二次使用负向提示词，这样可以得到更好的效果，但是会增加推理时间

`sample_guide_scale` 表示 CFG 引导强度，控制 CFG 的作用力度

qwen_image_t2i_2512_distill.json内容如下：
```
{
    "infer_steps": 8,
    "aspect_ratio": "16:9",
    "prompt_template_encode": "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    "prompt_template_encode_start_idx": 34,
    "attn_type": "flash_attn3",
    "enable_cfg": false,
    "sample_guide_scale": 4.0,
    "lora_configs": [
        {
          "path": "lightx2v/Qwen-Image-2512-Lightning/Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors",
          "strength": 1.0
        }
      ]
}
```
`infer_steps` 表示推理的步数，这是蒸馏模型，推理步数蒸馏成8步

`enable_cfg` 表示是否启用cfg，已经做了CFG蒸馏的模型，设置为false

`lora_configs` 表示Lora权重配置，需修改路径为本地实际路径

qwen_image_t2i_2512_distill_fp8.json内容如下：
```
{
    "infer_steps": 8,
    "aspect_ratio": "16:9",
    "prompt_template_encode": "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    "prompt_template_encode_start_idx": 34,
    "attn_type": "flash_attn3",
    "enable_cfg": false,
    "sample_guide_scale": 4.0,
    "dit_quantized": true,
    "dit_quantized_ckpt": "lightx2v/Qwen-Image-2512-Lightning/qwen_image_2512_fp8_e4m3fn_scaled_8steps_v1.0.safetensors",
    "dit_quant_scheme": "fp8-sgl"
}
```
`dit_quantized`	表示是否启用 DIT 量化，设置为True表示对模型核心的 DIT 模块做量化处理

`dit_quantized_ckpt` 表示 DIT 量化权重路径，指定 FP8 量化后的 DIT 权重文件的本地路径

`dit_quant_scheme` 表示 DIT 量化方案，指定量化类型为 "fp8-sgl"（fp8-sgl表示使用sglang的fp8 kernel进行推理）

### 启动服务生成图片

启动服务
```
cd LightX2V/scripts/server

# 运行下面的脚本之前，需要将脚本中的lightx2v_path和model_path替换为实际路径
# 例如：lightx2v_path=/home/user/LightX2V
# 例如：model_path=/home/user/models/Qwen/Qwen-Image-2511
# 同时：config_json也需要配成对应的模型config路径
# 例如：config_json ${lightx2v_path}/configs/qwen_image/qwen_image_t2i_2512.json

bash start_server_t2i.sh
```
向服务端发送请求

此处需要打开第二个终端作为用户
```
cd LightX2V/scripts/server

# 运行post.py前，需要将脚本中的url修改为 url = "http://localhost:8000/v1/tasks/image/"
python post.py
```
发送完请求后，可以在服务端看到推理的日志

### python代码生成图片


运行步数蒸馏 + FP8 量化模型

运行 `qwen_2511_fp8.py` 脚本，该脚本使用步数蒸馏和 FP8 量化优化的模型：
```
cd examples/qwen_image/

# 运行前需设置环境变量
export PYTHONPATH=/home/user/LightX2V

# 运行前需修改脚本中的路径为实际路径，包括：model_path、dit_quantized_ckpt、image_path、save_result_path
python qwen_2511_fp8.py
```
该方式通过步数蒸馏技术减少推理步数，同时使用 FP8 量化降低模型大小和内存占用，实现更快的推理速度。

解释细节：

qwen_2511_fp8.py脚本内容如下：
```
"""
Qwen-image-edit image-to-image generation example.
This example demonstrates how to use LightX2V with Qwen-Image-Edit model for I2I generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for Qwen-image-edit I2I task
# For Qwen-Image-Edit-2511, use model_cls="qwen-image-edit-2511"
pipe = LightX2VPipeline(
    model_path="/path/to/Qwen-Image-Edit-2511",
    model_cls="qwen-image-edit-2511",
    task="i2i",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(
#     config_json="../configs/qwen_image/qwen_image_i2i_2511_distill_fp8.json"
# )

# Enable offloading to significantly reduce VRAM usage with minimal speed impact
# Suitable for RTX 30/40/50 consumer GPUs
# pipe.enable_offload(
#     cpu_offload=True,
#     offload_granularity="block", #["block", "phase"]
#     text_encoder_offload=True,
#     vae_offload=False,
# )

# Load fp8 distilled weights (and int4 Qwen2_5 vl model (optional))
pipe.enable_quantize(
    dit_quantized=True,
    dit_quantized_ckpt="lightx2v/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
    quant_scheme="fp8-sgl",
    # text_encoder_quantized=True,
    # text_encoder_quantized_ckpt="lightx2v/Encoders/GPTQModel/Qwen25-VL-4bit-GPTQ",
    # text_encoder_quant_scheme="int4"
)

# Create generator manually with specified parameters
pipe.create_generator(
    attn_mode="flash_attn3",
    resize_mode="adaptive",
    infer_steps=8,
    guidance_scale=1,
)

# Generation parameters
seed = 42
prompt = "Replace the polka-dot shirt with a light blue shirt."
negative_prompt = ""
image_path = "/path/to/img.png"  # or "/path/to/img_0.jpg,/path/to/img_1.jpg"
save_result_path = "/path/to/save_results/output.png"

# Generate video
pipe.generate(
    seed=seed,
    image_path=image_path,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
```
注意1：可以通过传入config的方式，设置运行中的参数，也可以通过函数参数传入的方式，设置运行中的参数二者只能选其一，不可同时使用。脚本中使用的是函数参数传入，将传入config的部分注释，推荐使用传入config的方式。对于A100-80G, 4090-24G和5090-32G等显卡，把flash_attn3替换为flash_attn2

注意2：RTX 30/40/50 GPUs可以启用 Offload 优化显存

运行 Qwen-Image-Edit-2511 模型 + 蒸馏 LoRA

运行 qwen_2511_with_distill_lora.py 脚本，该脚本使用 Qwen-Image-Edit-2511 基础模型配合蒸馏 LoRA：

```
cd examples/qwen_image/

# 运行前需修改脚本中的路径为实际路径，包括：model_path、pipe.enable_lora中的path、image_path、save_result_path
python qwen_2511_with_distill_lora.py
```

该方式使用完整的 Qwen-Image-Edit-2511 模型，并通过蒸馏 LoRA 进行模型优化，在保持模型性能的同时提升推理效率。
