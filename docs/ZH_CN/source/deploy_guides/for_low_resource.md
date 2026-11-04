# Wan 低资源部署

本页介绍 Wan2.1 的量化、CPU/磁盘卸载与 VAE 优化。先完成[环境安装](../getting_started/quickstart.md)，再选择与模型、GPU 后端匹配的配置。显存和主机内存需求取决于权重、尺寸、帧数及算子；8GB 显存、16/32GB 内存不能作为所有配置的运行保证。

## 选择配置

| 模型与任务 | 配置 | 模板默认值 | 启动方式 |
| --- | --- | --- | --- |
| 14B I2V 480P | [480P 磁盘卸载](https://github.com/ModelTC/LightX2V/blob/main/configs/offload/disk/wan_i2v_phase_lazy_load_480p.json) | 40 步、CFG 开启、`[480, 832]` | 使用下方 CLI 示例 |
| 14B I2V 720P | [720P 磁盘卸载](https://github.com/ModelTC/LightX2V/blob/main/configs/offload/disk/wan_i2v_phase_lazy_load_720p.json) | 40 步、CFG 关闭、`[1280, 720]` | [720P 启动脚本](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan/inference/run_wan_i2v_lazy_load.sh) |
| 1.3B T2V 480P | [1.3B 配置](https://github.com/ModelTC/LightX2V/blob/main/configs/offload/block/wan_t2v_1_3b.json) | 4 步、CFG 开启、`[480, 832]` | CLI 使用 `--task t2v`，替换模型和配置，移除 `--image_path` |

尺寸均按 `[高, 宽]` 表示，720P 模板默认为竖屏。采样步数和 CFG 必须与权重匹配：4 步模板需配套蒸馏权重；使用原始权重时，应采用对应的采样配置。720P 脚本还固定了 FP16 和 `--size 1280 720`，切换模型或尺寸时需修改脚本，不能只替换 JSON。

## 运行 480P I2V

1. 准备完整的 Wan2.1-I2V-14B-480P 模型目录，以及配置要求的量化 DiT、T5、CLIP 和 Tiny VAE 权重。
2. 编辑上述 480P JSON，将 `dit_quantized_ckpt`、`t5_quantized_ckpt`、`clip_quantized_ckpt`、`tae_path` 替换为实际路径。`lazy_load=true` 需要按 block 保存的 DiT 权重目录，转换方法见[模型转换说明](https://github.com/ModelTC/LightX2V/blob/main/tools/convert/readme_zh.md)中的 `--save_by_block`。
3. 安装配置使用的 `sage_attn2` 和 `fp8-vllm` 等后端；若不支持当前 GPU，按[注意力](../method_tutorials/attention.md)和[量化指南](../method_tutorials/quantization.md)选择匹配的后端与权重。

在仓库根目录运行，将模型路径和 GPU 编号替换为实际值：

```bash
lightx2v_path="$PWD"
model_path="/path/to/Wan2.1-I2V-14B-480P"
export CUDA_VISIBLE_DEVICES=0
source scripts/base/base.sh

python -m lightx2v.infer \
  --model_cls wan2.1 \
  --task i2v \
  --model_path "$model_path" \
  --config_json configs/offload/disk/wan_i2v_phase_lazy_load_480p.json \
  --prompt "A white cat relaxing on a sunny beach." \
  --image_path assets/inputs/imgs/img_0.jpg \
  --save_result_path save_results/low_resource.mp4
```

尺寸和帧数沿用 JSON。推理完成后查看 `save_results/low_resource.mp4`。首次运行先检查峰值显存和主机内存，再增加尺寸或帧数。

## 按需调整

在完整 JSON 中修改以下字段；量化和卸载的详细配置以链接中的说明为准。

| 需求 | 设置与条件 |
| --- | --- |
| 降低权重占用 | 启用 `dit_quantized`、`t5_quantized` 或 `clip_quantized`，同时设置相应 scheme 和权重路径；仅设置 scheme 不会启用量化 |
| 降低 GPU 显存 | `cpu_offload=true`、`t5_cpu_offload=true`；主机内存需容纳卸载权重和缓冲区 |
| 降低 DiT 主机内存 | `lazy_load=true`，使用按 block 保存的权重和 SSD；结合磁盘吞吐量调整 `num_disk_workers` |
| 细化 DiT 卸载 | `offload_granularity="phase"`，按每个 block 内的 self-attention、cross-attention、FFN 阶段卸载 |
| 降低中间张量占用 | 按需启用 `clean_cuda_cache`、`rope_chunk` 并调整 `rope_chunk_size` |
| 降低 VAE 显存 | `use_tiling_vae=true`；或启用 `use_tae` 并设置 `tae_path`，Tiny VAE 需权衡画质 |
| 减少推理步数 | 使用匹配的[蒸馏权重与配置](../method_tutorials/step_distill.md)，不要仅降低原始模型的 `infer_steps` |

Tiny VAE 的 Wan2.1 权重见 [taew2_1.pth](https://github.com/madebyollin/taehv/raw/refs/heads/main/taew2_1.pth)。磁盘空间应覆盖基础模型、量化/蒸馏权重和输出文件。

## 继续阅读

- [参数卸载](../method_tutorials/offload.md)：CPU、磁盘卸载配置。
- [量化](../method_tutorials/quantization.md)：权重格式、scheme 和后端依赖。
- [Gradio-Lightx2v](https://github.com/ModelTC/Gradio-Lightx2v)：独立图形界面与部署说明。
