---

# 模型转换工具（中文）

一个强大的模型权重转换工具，支持格式转换、量化、LoRA 合并等功能。

## 主要特性
- **格式转换**：支持 PyTorch（.pth）与 SafeTensors（.safetensors）互转  
- **模型量化**：支持 INT8、FP8、NVFP4、MXFP4、MXFP6、MXFP8，显著减小模型体积  
- **架构转换**：支持 LightX2V 与 Diffusers 架构互转  
- **LoRA 合并**：支持加载并合并多种 LoRA 格式  
- **多模型支持**：支持 Wan DiT、Qwen Image DiT、T5、CLIP 等  
- **灵活保存**：支持单文件、分 block、分 chunk 保存  
- **并行处理**：支持大模型转换的并行加速  

## 支持的模型类型
- `hunyuan_dit`：hunyuan DiT 1.5 模型  
- `wan_dit`：Wan DiT 系列模型（默认）  
- `wan_animate_dit`：Wan Animate DiT 模型  
- `qwen_image_dit`：Qwen Image DiT 模型  
- `wan_t5`：Wan T5 文本编码器  
- `wan_clip`：Wan CLIP 视觉编码器  

## 核心参数

### 基础参数
- `-s, --source`：输入路径（文件或目录）  
- `-o, --output`：输出目录  
- `-o_e, --output_ext`：输出格式，`.pth` 或 `.safetensors`（默认）  
- `-o_n, --output_name`：输出文件名（默认 `converted`）  
- `-t, --model_type`：模型类型（默认 `wan_dit`）  

### 架构转换参数
- `-d, --direction`：转换方向  
  - `None`：不转换（默认）  
  - `forward`：LightX2V → Diffusers  
  - `backward`：Diffusers → LightX2V  

### 量化参数
- `--quantized`：启用量化  
- `--bits`：量化比特（目前仅 8-bit）  
- `--linear_type`：线性层量化类型  
  - `int8`（torch.int8）  
  - `fp8`（torch.float8_e4m3fn）  
  - `nvfp4` / `mxfp4` / `mxfp6` / `mxfp8`  
- `--non_linear_dtype`：非线性层数据类型（`torch.bfloat16` / `torch.float16` / `torch.float32` 默认）  
- `--device`：量化设备 `cpu` 或 `cuda`（默认）  
- `--comfyui_mode`：ComfyUI 兼容模式（仅 int8、fp8）  
- `--full_quantized`：全量化模式（在 ComfyUI 模式下生效）  

> 对于 nvfp4、mxfp4、mxfp6、mxfp8，请按 `LightX2V/lightx2v_kernel/README.md` 安装算子。

### LoRA 参数
- `--lora_path`：LoRA 路径（可多个，空格分隔）  
- `--lora_strength`：LoRA 强度（可多个，默认 1.0）  
- `--alpha`：LoRA alpha（可多个）  
- `--lora_key_convert`：LoRA key 转换模式：`auto`（默认）/ `same` / `convert`  

### 保存参数
- `--single_file`：保存为单文件（大模型占用内存高）  
- `-b, --save_by_block`：按 block 保存（后向转换推荐）  
- `-c, --chunk-size`：分块大小（默认 100，0 为不分块）  
- `--copy_no_weight_files`：复制源目录中的非权重文件  

### 性能参数
- `--parallel`：开启并行（默认 True）  
- `--no-parallel`：关闭并行  

## 支持的 LoRA 格式
1. Standard：`{key}.lora_up.weight` / `{key}.lora_down.weight`  
2. Diffusers：`{key}_lora.up.weight` / `{key}_lora.down.weight`  
3. Diffusers V2：`{key}.lora_B.weight` / `{key}.lora_A.weight`  
4. Diffusers V3：`{key}.lora.up.weight` / `{key}.lora.down.weight`  
5. Mochi：`{key}.lora_B` / `{key}.lora_A`（无 .weight 后缀）  
6. Transformers：`{key}.lora_linear_layer.up.weight` / `{key}.lora_linear_layer.down.weight`  
7. Qwen：`{key}.lora_B.default.weight` / `{key}.lora_A.default.weight`  
此外支持 diff 文件：`.diff` / `.diff_b` / `.diff_m`  

## 使用示例

### 1. 模型量化
#### 1.1 Wan DiT → INT8  
多文件（按 DiT block 保存）
```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name wan_int8 \
    --linear_type int8 \
    --model_type wan_dit \
    --quantized \
    --save_by_block
```
单文件
```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name wan2.1_i2v_480p_int8_lightx2v \
    --linear_type int8 \
    --model_type wan_dit \
    --quantized \
    --single_file
```

#### 1.2 Wan DiT → FP8  
多文件（按 DiT block 保存）
```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name wan_fp8 \
    --linear_type fp8 \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_dit \
    --quantized \
    --save_by_block
```
单文件
```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v \
    --linear_type fp8 \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_dit \
    --quantized \
    --single_file
```
ComfyUI scaled_fp8
```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v_comfyui \
    --linear_type fp8 \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_dit \
    --quantized \
    --single_file \
    --comfyui_mode
```
ComfyUI full FP8
```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v_comfyui \
    --linear_type fp8 \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_dit \
    --quantized \
    --single_file \
    --comfyui_mode \
    --full_quantized
```
> 提示：其他 DiT 模型切换 `--model_type` 即可

#### 1.3 T5 编码器量化
INT8
```bash
python converter.py \
    --source /path/to/models_t5_umt5-xxl-enc-bf16.pth \
    --output /path/to/output \
    --output_ext .pth \
    --output_name models_t5_umt5-xxl-enc-int8 \
    --linear_type int8 \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_t5 \
    --quantized
```
FP8
```bash
python converter.py \
    --source /path/to/models_t5_umt5-xxl-enc-bf16.pth \
    --output /path/to/output \
    --output_ext .pth \
    --output_name models_t5_umt5-xxl-enc-fp8 \
    --linear_type fp8 \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_t5 \
    --quantized
```

#### 1.4 CLIP 编码器量化
INT8
```bash
python converter.py \
    --source /path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --output /path/to/output \
    --output_ext .pth \
    --output_name models_clip_open-clip-xlm-roberta-large-vit-huge-14-int8 \
    --linear_type int8 \
    --non_linear_dtype torch.float16 \
    --model_type wan_clip \
    --quantized
```
FP8
```bash
python converter.py \
    --source /path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --output /path/to/output \
    --output_ext .pth \
    --output_name models_clip_open-clip-xlm-roberta-large-vit-huge-14-fp8 \
    --linear_type fp8 \
    --non_linear_dtype torch.float16 \
    --model_type wan_clip \
    --quantized
```

#### 1.5 Qwen25_vl LLM 量化
INT8
```bash
python converter.py \
    --source /path/to/hunyuanvideo-1.5/text_encoder/llm \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name qwen25vl-llm-int8 \
    --linear_dtype torch.int8 \
    --non_linear_dtype torch.float16 \
    --model_type qwen25vl_llm \
    --quantized \
    --single_file
```
FP8
```bash
python converter.py \
    --source /path/to/hunyuanvideo-1.5/text_encoder/llm \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name qwen25vl-llm-fp8 \
    --linear_dtype torch.float8_e4m3fn \
    --non_linear_dtype torch.float16 \
    --model_type qwen25vl_llm \
    --quantized \
    --single_file
```

### 2. LoRA 合并
#### 2.1 单 LoRA 合并
```bash
python converter.py \
    --source /path/to/base_model/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name merged_model \
    --model_type wan_dit \
    --lora_path /path/to/lora.safetensors \
    --lora_strength 1.0 \
    --single_file
```
#### 2.2 多 LoRA 合并
```bash
python converter.py \
    --source /path/to/base_model/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name merged_model \
    --model_type wan_dit \
    --lora_path /path/to/lora1.safetensors /path/to/lora2.safetensors \
    --lora_strength 1.0 0.8 \
    --single_file
```
#### 2.3 LoRA 合并 + 量化
LoRA → FP8
```bash
python converter.py \
    --source /path/to/base_model/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name merged_quantized \
    --model_type wan_dit \
    --lora_path /path/to/lora.safetensors \
    --lora_strength 1.0 \
    --quantized \
    --linear_type fp8 \
    --single_file
```
LoRA → ComfyUI scaled_fp8
```bash
python converter.py \
    --source /path/to/base_model/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name merged_quantized \
    --model_type wan_dit \
    --lora_path /path/to/lora.safetensors \
    --lora_strength 1.0 \
    --quantized \
    --linear_type fp8 \
    --single_file \
    --comfyui_mode
```
LoRA → ComfyUI full FP8
```bash
python converter.py \
    --source /path/to/base_model/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name merged_quantized \
    --model_type wan_dit \
    --lora_path /path/to/lora.safetensors \
    --lora_strength 1.0 \
    --quantized \
    --linear_type fp8 \
    --single_file \
    --comfyui_mode \
    --full_quantized
```

#### 2.4 LoRA Key 转换模式
自动检测（推荐）
```bash
python converter.py \
    --source /path/to/model/ \
    --output /path/to/output \
    --lora_path /path/to/lora.safetensors \
    --lora_key_convert auto \
    --single_file
```
保持原 key
```bash
python converter.py \
    --source /path/to/model/ \
    --output /path/to/output \
    --direction forward \
    --lora_path /path/to/lora.safetensors \
    --lora_key_convert same \
    --single_file
```
按模型转换
```bash
python converter.py \
    --source /path/to/model/ \
    --output /path/to/output \
    --direction forward \
    --lora_path /path/to/lora.safetensors \
    --lora_key_convert convert \
    --single_file
```

### 3. 架构格式转换
#### 3.1 LightX2V → Diffusers
```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P \
    --output /path/to/Wan2.1-I2V-14B-480P-Diffusers \
    --output_ext .safetensors \
    --model_type wan_dit \
    --direction forward \
    --chunk-size 100
```
#### 3.2 Diffusers → LightX2V
```bash
python converter.py \
    --source /path/to/Wan2.1-I2V-14B-480P-Diffusers \
    --output /path/to/Wan2.1-I2V-14B-480P \
    --output_ext .safetensors \
    --model_type wan_dit \
    --direction backward \
    --save_by_block
```

### 4. 格式转换
#### 4.1 .pth → .safetensors
```bash
python converter.py \
    --source /path/to/model.pth \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name model \
    --single_file
```
#### 4.2 多个 .safetensors → 单文件
```bash
python converter.py \
    --source /path/to/model_directory/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name merged_model \
    --single_file
```
