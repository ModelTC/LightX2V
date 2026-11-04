# Low-resource Wan deployment

This guide covers quantization, CPU/disk offloading, and VAE options for Wan2.1. Complete [environment setup](../getting_started/quickstart.md), then select a config compatible with your model and GPU backends. VRAM and host RAM requirements depend on weights, dimensions, frame count, and kernels; 8GB VRAM and 16/32GB RAM do not guarantee that every config will run.

## Choose a configuration

| Model and task | Configuration | Template defaults | Entry point |
| --- | --- | --- | --- |
| 14B I2V 480P | [480P disk offload](https://github.com/ModelTC/LightX2V/blob/main/configs/offload/disk/wan_i2v_phase_lazy_load_480p.json) | 40 steps, CFG on, `[480, 832]` | CLI example below |
| 14B I2V 720P | [720P disk offload](https://github.com/ModelTC/LightX2V/blob/main/configs/offload/disk/wan_i2v_phase_lazy_load_720p.json) | 40 steps, CFG off, `[1280, 720]` | [720P launcher](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan/inference/run_wan_i2v_lazy_load.sh) |
| 1.3B T2V 480P | [1.3B config](https://github.com/ModelTC/LightX2V/blob/main/configs/offload/block/wan_t2v_1_3b.json) | 4 steps, CFG on, `[480, 832]` | Use `--task t2v`, change the model and config, and omit `--image_path` |

Dimensions use `[height, width]`; the 720P template is portrait. Match sampling steps and CFG to the weights: the 4-step template needs matching distilled weights; original weights need their corresponding sampling config. The 720P launcher also fixes FP16 and `--size 1280 720`. Update the script when changing models or dimensions; changing only the JSON does not override these settings.

## Run 480P I2V

1. Prepare a complete Wan2.1-I2V-14B-480P model directory and the quantized DiT, T5, CLIP, and Tiny VAE weights required by the config.
2. Edit the 480P JSON above, replacing `dit_quantized_ckpt`, `t5_quantized_ckpt`, `clip_quantized_ckpt`, and `tae_path` with real paths. `lazy_load=true` requires a directory of DiT weights saved by block; see `--save_by_block` in the [conversion guide](https://github.com/ModelTC/LightX2V/blob/main/tools/convert/readme.md).
3. Install the configured backends, including `sage_attn2` and `fp8-vllm`. If your GPU does not support them, choose matching backends and weights using the [attention](../method_tutorials/attention.md) and [quantization](../method_tutorials/quantization.md) guides.

From the repository root, replace the model path and GPU ID, then run:

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

Dimensions and frame count come from the JSON. View `save_results/low_resource.mp4` after inference. Check peak VRAM and host RAM before increasing dimensions or frame count.

## Optional adjustments

Edit these fields in the complete JSON. Follow the linked guides for detailed quantization and offload settings.

| Goal | Settings and requirements |
| --- | --- |
| Reduce weight memory | Enable `dit_quantized`, `t5_quantized`, or `clip_quantized` with the matching scheme and weight path; a scheme alone does not enable quantization |
| Reduce VRAM | Set `cpu_offload=true` and `t5_cpu_offload=true`; host RAM must hold offloaded weights and buffers |
| Reduce DiT host RAM | Set `lazy_load=true`, use weights saved by block on an SSD, and tune `num_disk_workers` for disk throughput |
| Finer DiT offloading | Set `offload_granularity="phase"` to offload self-attention, cross-attention, and FFN phases within each block |
| Reduce intermediate memory | Enable `clean_cuda_cache` or `rope_chunk` and adjust `rope_chunk_size` as needed |
| Reduce VAE memory | Set `use_tiling_vae=true`; alternatively enable `use_tae` with `tae_path`, accounting for Tiny VAE's quality tradeoff |
| Reduce sampling steps | Use matching [distilled weights and configs](../method_tutorials/step_distill.md), rather than only reducing an original model's `infer_steps` |

Wan2.1 Tiny VAE weights are available as [taew2_1.pth](https://github.com/madebyollin/taehv/raw/refs/heads/main/taew2_1.pth). Allow disk space for base weights, quantized/distilled weights, and output files.

## Further reading

- [Offloading](../method_tutorials/offload.md): CPU and disk offload settings.
- [Quantization](../method_tutorials/quantization.md): weight formats, schemes, and backend dependencies.
- [Gradio-Lightx2v](https://github.com/ModelTC/Gradio-Lightx2v): standalone graphical interface and deployment instructions.
