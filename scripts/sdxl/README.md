# SDXL base 1.0 → LightX2V (A100)

目标为文生图模型 `stabilityai/stable-diffusion-xl-base-1.0`，revision
`462165984030d82259a11f4367a4eed129e94a7b`，不包含 refiner。
先在独立 Python 3.10 环境中执行 `bash scripts/sdxl/install_a100.sh`。

保持官方 Diffusers 权重目录布局：`unet/`、`vae/`、`text_encoder/`、`text_encoder_2/`、
`tokenizer/`、`tokenizer_2/`、`scheduler/`。优先读取 `.fp16.safetensors`，只访问本地文件。
Diffusers 是文件布局；运行时不实例化 Diffusers 的模型、pipeline 或 scheduler。
transformers 仅提供 tokenizer。

```bash
MODEL_PATH=/path/to/stable-diffusion-xl-base-1.0 \
PROMPT='A red apple on a wooden table, studio photograph.' \
SAVE_RESULT_PATH=save_results/sdxl/apple.png \
bash scripts/platforms/nvidia/run_sdxl.sh
```

默认 1024×1024、30 步 Euler、CFG=5、seed=42。支持 `--negative_prompt`、
`--size HEIGHT WIDTH`，尺寸至少 128 且为 64 的倍数。

A100 单卡已实现；MLU、MetaX、Ascend 脚本与 JSON 为明确占位。暂不支持
refiner、LoRA、量化、并行、offload 或 feature cache。

Both CLIP encoders use Transformers’ existing `CLIPTokenizerFast` implementation. Its Rust normalization matches the CLIP tokenizer used by the Transformers 5 CLIP tokenizer, including Unicode punctuation and literal HTML entities; the model does not maintain a separate tokenizer implementation.
