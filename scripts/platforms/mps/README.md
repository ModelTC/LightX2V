# MiniMax-H3 on Apple MPS

需要支持 BF16 的 PyTorch MPS 环境和 MiniMax-H3 diffusers 权重目录：

```text
MiniMax-H3/
├── transformer/
├── text_encoder/
├── tokenizer/
├── vae/
└── audio_vae/
```

各组件保留发布权重中的配置、safetensors 和索引文件，无需转换权重格式。

## 运行

在 LightX2V 根目录执行：

```bash
export MODEL_PATH=/path/to/MiniMax-H3
# 缓存脚本默认使用当前环境中的 python，也可设置 PYTHON=/path/to/python。

# 首次运行生成 AdaLN 缓存；已有匹配缓存时可跳过。
bash scripts/platforms/mps/run_cache_minimax_h3_adaln.sh
bash scripts/platforms/mps/run_minimax_h3_t2av.sh
```

默认配置为 `configs/platforms/mps/minimax_h3_t2av_4step_512_22.json`，生成 512×512、22 帧、4 步、BF16 的音视频，用于快速验证。输出为 `save_results/output_lightx2v_minimax_h3_t2av.mp4`。

推理脚本使用固定的本机仓库、模型、Python 路径和推理参数，调整时直接修改脚本。缓存脚本支持 `MODEL_PATH`、`PYTHON` 和 `CONFIG_JSON` 环境变量；生成缓存时请与推理脚本中的模型和配置保持一致。

缓存生成与推理必须使用相同模型和配置，尤其是步数与 flow shift。MPS 配置默认将缓存写入 `~/.cache/lightx2v/adaln/diffusers`；更换权重后需重新生成。

公共入口 `tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh` 默认使用 CUDA 和通用 H3 配置；MPS 请使用上面的专用入口。

## 内存配置与限制

DiT 和文本编码器逐层从磁盘读取权重，VAE 在解码阶段加载。当前文本编码器磁盘加载仅支持 `--model-variant fl2av --task t2av`，不支持图像条件、量化或张量并行。

DiT 磁盘逐层加载（`dit_disk_streaming=true`）不支持 LoRA，包括启动时加载和运行时切换。请勿设置 `lora_configs`、传入 `lora_path` 或启用 `lora_dynamic_apply`。

Mac 配置默认开启 `text_encoder_prefetch=true`：文本编码器复用两套层权重缓冲区，后台线程将下一层直接读入空闲的 MPS 共享缓冲区，与当前层计算重叠。该模式要求 `torch.mps._host_alias_storage` 可用、文件 dtype 与推理 dtype 一致。设为 `false` 可恢复单缓冲区同步加载，减少一层权重的内存占用；两种模式均在文本编码结束后按 `text_encoder_release_block_offload_buffers` 释放缓冲区。

默认开启 `dit_mps_shared_buffer=true`：两套 DiT 权重缓冲区交替计算和预读，在复用前等待 GPU 计算与磁盘读取完成。此模式要求：

- PyTorch 提供私有接口 `torch.mps._host_alias_storage`，已在 PyTorch 2.14.0 验证。
- `cpu_offload=true`、`offload_granularity="block"`、`dit_disk_streaming=true`。
- 非量化权重，文件 dtype 与推理 dtype 一致，并启用 AdaLN 缓存。

缺少共享视图接口时，可将 `dit_mps_shared_buffer` 设为 `false`，使用单缓冲区磁盘加载。

MPS 配置使用独立的 `attn_type="torch_sdpa_mps"`。`mps_sdpa_query_chunk_size` 控制该实现的 query 分块大小，配置默认为 512；每块仍访问完整的 key/value，设为 0 可关闭分块。普通 `torch_sdpa` 不使用该参数。

H3 的 `rope_compute_dtype` 控制 RoPE 计算精度，未设置时默认为 `"fp32"`。MPS 配置显式设为 `"bf16"`，保留此前的数值行为；使用融合 QKV norm/RoPE 算子时需保留 `"fp32"`。
