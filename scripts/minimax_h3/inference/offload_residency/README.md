# MiniMax-H3 Resident Blocks

English | [简体中文](README_CN.md)

Run MiniMax-H3 text-to-audio-video on one device and compare a zero-residency baseline with resident DiT blocks. A separate tool measures one block's compute and transfer times and writes a resident count into a config. Inference reads that count and distributes resident blocks evenly across the stack.

## Before running

Prepare the complete BF16 Diffusers components following the [model guide](../../README.md). Set `lightx2v_path`, `model_path`, and the device index at the top of the scripts in this directory. The default task is `t2av` with the `fl2av` model weights, 124 frames, 768×1344 resolution, and 29 steps. Edit `--prompt` in the scripts; keep the prompt identical for estimation, baseline, and resident inference.

Generate a matching cache using the model's [AdaLN cache instructions](../../README.md#adaln-cache). The existing entry point is `tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh`: set the model path, platform, and the corresponding baseline config below, and select `--model-variant fl2av`. On the same platform, estimation and both inference modes share this cache. If an older cache fails validation, preserve its directory before regenerating it.

## NVIDIA GPU

Use `configs/minimax_h3/residency/baseline.json`. Set `CUDA_VISIBLE_DEVICES` in all three CUDA scripts, then run from the repository root:

```bash
bash scripts/minimax_h3/inference/offload_residency/estimate.sh
bash scripts/minimax_h3/inference/offload_residency/run.sh
bash scripts/minimax_h3/inference/offload_residency/run_baseline.sh
```

## Ascend NPU

Use `configs/minimax_h3/residency/baseline_npu.json`. Set `ASCEND_RT_VISIBLE_DEVICES` in all three NPU scripts:

```bash
bash scripts/minimax_h3/inference/offload_residency/estimate_npu.sh
bash scripts/minimax_h3/inference/offload_residency/run_npu.sh
bash scripts/minimax_h3/inference/offload_residency/run_baseline_npu.sh
```

The baseline runs independently. Resident inference requires estimation on the corresponding platform first.

## Outputs

All outputs go to `save_results/minimax_h3_residency/`. Repeated runs overwrite files with the same name.

| File | Contents |
| --- | --- |
| `resident.json` / `resident_npu.json` | Generated inference config containing `offload_resident_blocks` |
| `resident.profile.json` / `resident_npu.profile.json` | Block compute/transfer timings, resident count, and indices |
| `resident.mp4` / `resident_npu.mp4` | Audio-video output with resident blocks |
| `baseline.mp4` / `baseline_npu.mp4` | Audio-video output from the zero-residency baseline |

Estimation loads one real block, constructs text/audio/video activations matching the requested geometry, and loads the production AdaLN cache. Compute and transfer are timed separately, each with 3 warmups and 10 samples; the median is used. No full denoising, text encoding, or VAE decoding is performed, so estimation finishes faster than inference.

The count is estimated as `r = max(0, N - floor(N*c/d))`, where `N` is the block count and `c` and `d` are compute and transfer times. A result of `r=0` is valid. This aggregate estimate does not guarantee that every transfer is hidden, and the tool does not reduce the count to fit available device memory.

The baseline sets `r=0`: all DiT block master weights stay on the CPU and are copied into two device buffers as needed. Resident inference also keeps all CPU masters, but holds additional device copies of the selected `r` blocks across denoising steps and releases them before VAE decoding. Zero residency still requires device memory for execution.

## Changing settings and comparing runs

After changing frame count, resolution, steps, or operators in the baseline config, estimate again. Also re-estimate after changing hardware, weights, precision, or prompt length. Changes to steps, flow shifts, or weights require a matching AdaLN cache. You can manually lower `r` in the generated config; estimation overwrites that edit.

The default is `cpu_offload_layout="per_tensor"`. Contiguous layout is independent: to use it, set `"contiguous"` in the baseline config and re-estimate. Keep the same layout in both comparison runs.

These entry points use BF16, one device, eager execution, and `NoCaching`. Quantization, shared CPU weights, parallel execution, LoRA, compile, disk streaming, and fused QKV are unsupported. The estimator currently models `t2av` inputs; its count is not a measured result for other tasks.

Detailed profiling is disabled by default. To compare denoising time, set `PROFILING_DEBUG_LEVEL=1` in both inference scripts and compare `Run Dit every step` for identical inputs and seeds. This includes scheduling and model work; the first step also prepares resident weights. The estimator's single-block time is not a full denoising-step time.
