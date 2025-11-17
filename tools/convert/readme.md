# Model Conversion Tool

A powerful model weight conversion tool that supports format conversion, quantization, LoRA merging, and more.

## Main Features

- **Format Conversion**: Support PyTorch (.pth) and SafeTensors (.safetensors) format conversion
- **Model Quantization**: Support INT8, FP8, NVFP4, MXFP4, MXFP6 and MXFP8 quantization to significantly reduce model size
- **Architecture Conversion**: Support conversion between LightX2V and Diffusers architectures
- **LoRA Merging**: Support loading and merging multiple LoRA formats
- **Multi-Model Support**: Support Wan DiT, Qwen Image DiT, T5, CLIP, etc.
- **Flexible Saving**: Support single file, block-based, and chunked saving methods
- **Parallel Processing**: Support parallel acceleration for large model conversion

## Supported Model Types

- `wan_dit`: Wan DiT series models (default)
- `wan_animate_dit`: Wan Animate DiT models
- `qwen_image_dit`: Qwen Image DiT models
- `wan_t5`: Wan T5 text encoder
- `wan_clip`: Wan CLIP vision encoder

## Core Parameters

### Basic Parameters

- `-s, --source`: Input path (file or directory)
- `-o, --output`: Output directory path
- `-o_e, --output_ext`: Output format, `.pth` or `.safetensors` (default)
- `-o_n, --output_name`: Output file name (default: `converted`)
- `-t, --model_type`: Model type (default: `wan_dit`)

### Architecture Conversion Parameters

- `-d, --direction`: Conversion direction
  - `None`: No architecture conversion (default)
  - `forward`: LightX2V → Diffusers
  - `backward`: Diffusers → LightX2V

### Quantization Parameters

- `--quantized`: Enable quantization
- `--bits`: Quantization bit width, currently only supports 8-bit
- `--linear_type`: Linear layer quantization type
  - `int8`: INT8 quantization (torch.int8)
  - `fp8`: FP8 quantization (torch.float8_e4m3fn)
  - `nvfp4`: NVFP4 quantization
  - `mxfp4`: MXFP4 quantization
  - `mxfp6`: MXFP6 quantization
  - `mxfp8`: MXFP8 quantization
- `--non_linear_dtype`: Non-linear layer data type
  - `torch.bfloat16`: BF16
  - `torch.float16`: FP16
  - `torch.float32`: FP32 (default)
- `--device`: Device for quantization, `cpu` or `cuda` (default)
- `--comfyui_mode`: ComfyUI compatible mode (only int8 and fp8)
- `--full_quantized`: Full quantization mode (effective in ComfyUI mode)
For nvfp4, mxfp4, mxfp6 and mxfp8, please install them fllowing LightX2V/lightx2v_kernel/README.md.

### LoRA Parameters

- `--lora_path`: LoRA file path(s), supports multiple (separated by spaces)
- `--lora_strength`: LoRA strength coefficients, supports multiple (default: 1.0)
- `--alpha`: LoRA alpha parameters, supports multiple
- `--lora_key_convert`: LoRA key conversion mode
  - `auto`: Auto-detect (default)
  - `same`: Use original key names
  - `convert`: Apply same conversion as model

### Saving Parameters

- `--single_file`: Save as single file (note: large models consume significant memory)
- `-b, --save_by_block`: Save by blocks (recommended for backward conversion)
- `-c, --chunk-size`: Chunk size (default: 100, 0 means no chunking)
- `--copy_no_weight_files`: Copy non-weight files from source directory

### Performance Parameters

- `--parallel`: Enable parallel processing (default: True)
- `--no-parallel`: Disable parallel processing

## Supported LoRA Formats

The tool automatically detects and supports the following LoRA formats:

1. **Standard**: `{key}.lora_up.weight` and `{key}.lora_down.weight`
2. **Diffusers**: `{key}_lora.up.weight` and `{key}_lora.down.weight`
3. **Diffusers V2**: `{key}.lora_B.weight` and `{key}.lora_A.weight`
4. **Diffusers V3**: `{key}.lora.up.weight` and `{key}.lora.down.weight`
5. **Mochi**: `{key}.lora_B` and `{key}.lora_A` (no .weight suffix)
6. **Transformers**: `{key}.lora_linear_layer.up.weight` and `{key}.lora_linear_layer.down.weight`
7. **Qwen**: `{key}.lora_B.default.weight` and `{key}.lora_A.default.weight`

Additionally supports diff formats:
- `.diff`: Weight diff
- `.diff_b`: Bias diff
- `.diff_m`: Modulation diff

## Usage Examples

### 1. Model Quantization

#### 1.1 Wan DiT Quantization to INT8

**Multiple safetensors, saved by dit blocks**
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

**Single safetensor file**
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

#### 1.2 Wan DiT Quantization to FP8

**Multiple safetensors, saved by dit blocks**
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

**Single safetensor file**
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

**ComfyUI scaled_fp8 format**
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

**ComfyUI full FP8 format**
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

> **Tip**: For other DIT models, simply switch the `--model_type` parameter

#### 1.3 T5 Encoder Quantization

**INT8 Quantization**
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

**FP8 Quantization**
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

#### 1.4 CLIP Encoder Quantization

**INT8 Quantization**
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

**FP8 Quantization**
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

### 2. LoRA Merging

#### 2.1 Merge Single LoRA

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

#### 2.2 Merge Multiple LoRAs

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

#### 2.3 LoRA Merging with Quantization

**LoRA Merge → FP8 Quantization**
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

**LoRA Merge → ComfyUI scaled_fp8**
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

**LoRA Merge → ComfyUI Full FP8**
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

#### 2.4 LoRA Key Conversion Modes

**Auto-detect mode (recommended)**
```bash
python converter.py \
    --source /path/to/model/ \
    --output /path/to/output \
    --lora_path /path/to/lora.safetensors \
    --lora_key_convert auto \
    --single_file
```

**Use original key names (LoRA already in target format)**
```bash
python converter.py \
    --source /path/to/model/ \
    --output /path/to/output \
    --direction forward \
    --lora_path /path/to/lora.safetensors \
    --lora_key_convert same \
    --single_file
```

**Apply conversion (LoRA in source format)**
```bash
python converter.py \
    --source /path/to/model/ \
    --output /path/to/output \
    --direction forward \
    --lora_path /path/to/lora.safetensors \
    --lora_key_convert convert \
    --single_file
```

### 3. Architecture Format Conversion

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

### 4. Format Conversion

#### 4.1 .pth → .safetensors

```bash
python converter.py \
    --source /path/to/model.pth \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name model \
    --single_file
```

#### 4.2 Multiple .safetensors → Single File

```bash
python converter.py \
    --source /path/to/model_directory/ \
    --output /path/to/output \
    --output_ext .safetensors \
    --output_name merged_model \
    --single_file
```
