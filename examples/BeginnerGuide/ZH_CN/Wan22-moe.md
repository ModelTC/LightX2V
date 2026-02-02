# 

本文档包含 Wan2.2-T2V-A14B 和 Wan2.2-I2V-A14B 模型的使用示例。

## 准备环境

请参考[01.PrepareEnv](01.PrepareEnv.md)

## 开始运行

准备模型
```
# 从huggingface下载
hf download Wan-AI/Wan2.2-T2V-A14B --local-dir Wan-AI/Wan2.2-T2V-A14B
hf download Wan-AI/Wan2.2-I2V-A14B --local-dir Wan-AI/Wan2.2-I2V-A14B

hf download lightx2v/Wan2.2-Distill-Models --local-dir Wan-AI/Wan2.2-Distill-Models
hf download lightx2v/Wan2.2-Distill-Loras --local-dir Wan-AI/Wan2.2-Distill-Loras
```

### 运行脚本生成

Wan2.2-T2V-A14B
```
# 运行前需将CUDA_VISIBLE_DEVICES替换为实际用的GPU
# 同时config文件中的parallel参数也需对应修改，满足cfg_p_size * seq_p_size = GPU数目
cd LightX2V/scripts/dist_infer
bash bash run_wan22_moe_t2v_cfg_ulysses.sh

# 步数蒸馏模型 Lora
# 修改 config_json 为LightX2V/configs/wan22/wan_moe_t2v_distill_lora.json，并修改其中的lora_configs为所使用的蒸馏模型路径
cd LightX2V/scripts/wan22
bash run_wan22_moe_t2v_distill.sh
```

Wan2.2-I2V-A14B
```
cd LightX2V/scripts/dist_infer
bash run_wan22_moe_i2v_cfg_ulysses.sh

# 步数蒸馏模型 Lora
# 修改 config_json 为LightX2V/configs/wan22/wan_moe_i2v_distill_with_lora.json
cd LightX2V/scripts/wan22
bash run_wan22_moe_i2v_distill.sh

# 步数蒸馏模型 merge Lora
# 修改 config_json 为LightX2V/configs/wan22/wan_moe_i2v_distill.json
cd LightX2V/scripts/wan22
bash run_wan22_moe_i2v_distill.sh

# 步数蒸馏+FP8量化模型
# 修改 config_json 为LightX2V/configs/wan22/wan_moe_i2v_distill_quant.json
cd LightX2V/scripts/wan22
bash run_wan22_moe_i2v_distill.sh
```
解释细节

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

`lora_configs`: 包含两个LoRA适配器，高噪声模型负责生成视频的高频细节和结构，低噪声模型负责平滑噪声和优化全局一致性。这种分工使得模型能够在不同阶段专注于不同的生成任务，从而提升整体性能。

wan_moe_i2v_distill.json内容如下
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
`high_noise_original_ckpt` 表示高噪声阶段使用的蒸馏模型路径

`low_noise_original_ckpt` 表示低噪声阶段使用的蒸馏模型路径

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
    "dit_quantized": true,
    "dit_quant_scheme": "fp8-sgl",
    "t5_quantized": false,
    "t5_quant_scheme": "fp8-sgl",
    "high_noise_quantized_ckpt": "lightx2v/Wan2.2-Distill-Models/wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors",
    "low_noise_quantized_ckpt": "lightx2v/Wan2.2-Distill-Models/wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"
}
```
`high_noise_quantized_ckpt` 表示高噪声阶段使用的步数蒸馏+FP8量化模型路径

`low_noise_quantized_ckpt` 表示低噪声阶段使用的蒸馏+FP8量化模型路径

### 启动服务生成

启动服务
```
cd LightX2V/scripts/server

# 运行下面的脚本之前，需要将脚本中的lightx2v_path和model_path替换为实际路径
# 例如：lightx2v_path=/home/user/LightX2V
# 例如：model_path=/home/user/models/Wan-AI/Wan2.2-T2V-A14B
# 同时：config_json也需要配成对应的模型config路径
# 例如：config_json ${lightx2v_path}/configs/wan22/wan_moe_t2v.json

# 切换model_path和config_json路径体验不同模型
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

### python代码生成
