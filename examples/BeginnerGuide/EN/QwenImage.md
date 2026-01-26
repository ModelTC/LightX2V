# Experience T2I and I2I with Qwen Image

This document contains usage examples for Qwen Image and Qwen Image Edit models.

Among them, the text-to-image model used is Qwen-Image-2512, and the image editing model is Qwen-Image-Edit-2511, both are the latest available versions of the respective models at present.

## Prepare the Environment

Please refer to [01.PrepareEnv](01.PrepareEnv.md)

## Getting Started

Prepare the model

```
# download from huggingface
# Inference with 2512 text-to-image original model
hf download Qwen/Qwen-Image-2512 --local-dir Qwen/Qwen-Image-2512

# Inference with 2512 text-to-image step-distilled model
hf download lightx2v/Qwen-Image-2512-Lightning --local-dir Qwen/Qwen-Image-2512-Lightning

# Inference with 2511 image editing original model
hf download Qwen/Qwen-Image-Edit-2511 --local-dir Qwen/Qwen-Image-2511

# Inference with 2511 image editing step-distilled model
hf download lightx2v/Qwen-Image-Edit-2511-Lightning --local-dir Qwen/Qwen-Image-2511-Lightning
```

We provide three ways to run the QwenImage model to generate images:

1. Run a script to generate images: Preset bash scripts for quick verification.
2. Start a service to generate images: Start the service and send requests, suitable for multiple inferences and actual deployment.
3. Generate images with Python code: Run with Python code, convenient for integration into existing codebases.

### Run Script to Generate Image

```
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V/scripts/qwen_image

# Before running the script below, replace lightx2v_path and model_path in the script with actual paths
# Example: lightx2v_path=/home/user/LightX2V
# Example: model_path=/home/user/models/Qwen/Qwen-Image-2511
```

Text-to-Image Models
```
# Inference with 2512 text-to-image original model, default is 50 steps
bash qwen_image_t2i_2512.sh

# Inference with 2512 text-to-image step-distilled model, default is 8 steps, requires modify the lora_configs path in config_json file
bash qwen_image_t2i_2512_distill.sh

# Inference with 2512 text-to-image step-distilled + FP8 quantized model, default is 8 steps, requires modify the dit_quantized_ckpt path in config_json file
bash qwen_image_t2i_2512_distill_fp8.sh
```

Note 1: In the scripts qwen_image_t2i_2512_distill.sh、qwen_image_t2i_2512_distill_fp8.sh, the model_path parameter shall be consistent with that in qwen_image_t2i_2512.sh, and it refers to the local path of the Qwen-Image-2512 model for all these scripts.

Note 2: The config_json file to be modified is located in the directory LightX2V/configs/qwen_image, lora_configs and dit_quantized_ckpt respectively refer to the local paths of the distilled model being used.


Image Editing Models
```
# Inference with 2511 image editing original model, default is 40 steps
bash qwen_image_i2i_2511.sh

# Inference with 2511 image editing step-distilled model, default is 8 steps, requires modify the lora_configs path in config_json file
bash qwen_image_i2i_2511_distill.sh

# Inference with 2511 image editing step-distilled + FP8 quantized model, default is 8 steps, requires modify the dit_quantized_ckpt path in config_json file
bash qwen_image_i2i_2511_distill_fp8.sh
```
Note 1: The model_path parameter in all bash scripts shall be set to the path of the Qwen-Image-2511 model. The paths to be modified in the config_json file refer respectively to the paths of the distilled model being used.

Note 2: You need to modify the image path parameter image_path in the bash scripts, and you can pass in your own image to test the model.

Explanation of details

The content of qwen_image_t2i_2512.sh is as follows:
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
`source ${lightx2v_path}/scripts/base/base.sh` sets some basic environment variables.

`--model_cls qwen_image` specifies using the qwen_image model.

`--task t2i` specifies the t2i task.

`--model_path` specifies the model path.

`--config_json` specifies the config file path.

`--prompt` specifies the prompt.

`--negative_prompt` specifies the negative prompt.

The content of qwen_image_t2i_2512.json is as follows:
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
`infer_steps`: Number of inference steps.

`aspect_ratio` specifies the aspect ratio of the target image.

`prompt_template_encode` specifies the template used for prompt encoding.

`prompt_template_encode_start_idx` specifies the valid starting index of the prompt template.

`attn_type` specifies the attention layers inside the qwen model. Here, flash_attn3 is used, which is only supported on Hopper architecture GPUs (H100, H20, etc.). For other GPUs, use flash_attn2 instead.

`enable_cfg`: Whether to enable cfg. Set to true here, meaning two inferences will be performed: one with the prompt and one with the negative prompt, for better results but increased inference time.

`sample_guide_scale` specifies CFG guidance strength, controls the intensity of CFG effect.

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
`infer_steps` number of inference steps.This is a distilled model, and the inference steps have been distilled to 8 steps.

`enable_cfg` whether to enable CFG. For models that have undergone CFG distillation, set this parameter to false.

`lora_configs` specifies the LoRA weight configuration, and the path needs to be modified to the actual local path.

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
`dit_quantized`	whether to enable DIT quantization; setting it to True means performing quantization processing on the core DIT module of the model.

`dit_quantized_ckpt` specifies the DIT quantized weight path, which specifies the local path of the DIT weight file after FP8 quantization.

`dit_quant_scheme` the DIT quantization scheme, which specifies the quantization type as "fp8-sgl" (FP8 single-granularity quantization).

### Start Service to Generate Image

Start the service:
```
cd LightX2V/scripts/server

# Before running the script below, replace lightx2v_path and model_path in the script with actual paths
# Example: lightx2v_path=/home/user/LightX2V
# Example: model_path=/home/user/models/Qwen/Qwen-Image-2511
# Additionally: Set config_json to the corresponding model config path.
# Example: config_json ${lightx2v_path}/configs/qwen_image/qwen_image_t2i_2512.json

bash start_server_t2i.sh
```
Send a request to the server:

Here we need to open a second terminal as a user.
```
cd LightX2V/scripts/server

# Before running post.py, you need to modify the url in the script to url = "http://localhost:8000/v1/tasks/image/"
python post.py
```
After sending the request, you can see the inference logs on the server.

### Generate Image with Python Code


Running Step-Distilled + FP8 Quantized Model

Run the `qwen_2511_fp8.py` script, which uses a model optimized with step distillation and FP8 quantization:
```
cd examples/qwen_image/

# Environment variables need to be set before running.
export PYTHONPATH=/home/user/LightX2V

# Before running, you need to modify the paths in the script to the actual paths, including: model_path, dit_quantized_ckpt, image_path, save_result_path
python qwen_2511_fp8.py
```
This approach reduces inference steps through step distillation technology while using FP8 quantization to reduce model size and memory footprint, achieving faster inference speed.

Explanation of details

The content of qwen_2511_fp8.py is as follows:
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
Note 1: You can set runtime parameters by passing a config or by passing function arguments. Only one method can be used at a time. The script adopts the function parameter passing method and the config passing section is commented out, but it is recommended to use the config passing method instead. For GPUs such as A100-80G, 4090-24G, and 5090-32G, replace flash_attn3 with flash_attn2.

Note 2: Offload can be enabled for RTX 30/40/50 series GPUs to optimize VRAM usage.

Running Qwen-Image-Edit-2511 Model + Distilled LoRA

Run the qwen_2511_with_distill_lora.py script, which uses the Qwen-Image-Edit-2511 base model with distilled LoRA:

```
cd examples/qwen_image/

# Before running, you need to modify the paths in the script to the actual paths, including: model_path, path in pipe.enable_lora, image_path, save_result_path
python qwen_2511_with_distill_lora.py
```

This approach uses the complete Qwen-Image-Edit-2511 model and optimizes it through distilled LoRA, improving inference efficiency while maintaining model performance.
