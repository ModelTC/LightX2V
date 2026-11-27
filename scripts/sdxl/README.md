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

A100 单卡已实现。MLU590 使用 torch.mlu 单卡入口，以 BF16 运行文本编码器和 UNet，VAE 与采样数值状态保留 FP32；使用 1024×1024、30 步 Euler、CFG=5 和 seed=42。运行
`PLATFORM=cambricon_mlu MLU_VISIBLE_DEVICES=0 PYTHON_BIN=/path/to/torch_mlu/python bash scripts/platforms/mlu/run_sdxl.sh`。
在已匹配 torch_mlu 与 torchvision 的环境中安装 `scripts/sdxl/requirements-mlu.txt`，不要运行 CUDA 专用的 `install_a100.sh`。Ascend 脚本与 JSON 仍为占位。MLU590 实机 smoke 验证前，不应视为已验收。暂不支持
refiner、LoRA、量化、并行、offload 或 feature cache。

Both CLIP encoders use Transformers’ existing `CLIPTokenizerFast` implementation. Its Rust normalization matches the CLIP tokenizer used by the Transformers 5 CLIP tokenizer, including Unicode punctuation and literal HTML entities; the model does not maintain a separate tokenizer implementation.

推理精度：两个 CLIP 文本编码器和 UNet 使用 BF16；归一化计算、时间频率、CFG 与 Euler 采样状态/更新、VAE 解码保留 FP32。检查点优先读取 `.fp16.safetensors`，其存储格式不限制加载后的计算精度。

## MetaX C500

Use a MetaX PyTorch/MACA environment and run `bash scripts/sdxl/install_metax.sh`. The installer preserves the installed MetaX PyTorch. Launch with `MACA_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=python3 bash scripts/platforms/metax/run_sdxl.sh`, using the same model and input environment variables documented above. The BF16/FP32 computation boundaries match the NVIDIA configuration.

Build an isolated image from this checkout with `docker build --build-arg METAX_BASE_IMAGE=<compatible-metax-development-image> -f dockerfiles/platforms/Dockerfile_metax_sdxl -t lightx2v-sdxl-metax .`. The validated runtime uses Python 3.10, MetaX PyTorch 2.8.0+metax3.5.3.9 and MACA SDK 3.5.3.18. Keep weights and generated outputs in external mounts.

## Ascend NPU

Use an isolated environment with matching PyTorch, torch_npu and CANN already
installed. Source the CANN `set_env.sh`, then run
`PYTHON_BIN=/path/to/python bash scripts/sdxl/install_ascend.sh`.
The installer keeps the installed accelerator PyTorch. Run the same requests with
`ASCEND_RT_VISIBLE_DEVICES=0 PYTHON_BIN=/path/to/python bash scripts/platforms/ascend_npu/run_sdxl.sh`
and the model/input environment variables documented above.

The original BF16 neural computation and FP32 numerical boundaries are preserved.
The NPU request disables HF32 convolution/matmul and restores the previous flags
afterwards. No weights or inputs are globally recast and no quantization is used.
