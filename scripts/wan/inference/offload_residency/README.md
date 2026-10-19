# Wan CPU Offload with Resident Weights

English | [简体中文](README_CN.md)

Run Wan2.1 image-to-video generation on a single device and compare two ways of managing block weights: `baseline` keeps all block weights in CPU memory and transfers them to device memory as needed; the residency strategy keeps some block weights in device memory throughout denoising to reduce repeated transfers.

Run `estimate` first to measure one block's computation and transfer times and generate a config with an estimated resident block count. Then run `run` for inference. Resident blocks are distributed evenly across the model's layers. Use `run_baseline` for comparison, or run it on its own.

Install the project dependencies for your platform, then run the commands below from the repository root.

## NVIDIA GPU

Use the FP8 weights for `Wan2.1-I2V-14B-720P-Lightx2v`. Set `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES` at the top of all three scripts, keeping the values consistent.

Set the following weight paths in `configs/wan/residency/baseline.json`. Absolute paths are recommended. These paths do not change automatically when you change `model_path` in the scripts.

- `dit_quantized_ckpt`: directory containing the FP8 DiT weights.
- `t5_quantized_ckpt`: quantized T5 file, such as `models_t5_umt5-xxl-enc-fp8.pth`.
- `clip_quantized_ckpt`: quantized CLIP file, such as `clip-fp8.pth`.

```bash
bash scripts/wan/inference/offload_residency/estimate.sh
bash scripts/wan/inference/offload_residency/run.sh
bash scripts/wan/inference/offload_residency/run_baseline.sh
```

## Ascend NPU

Use the original BF16 weights for `Wan2.1-I2V-14B-720P`, including the text encoder, CLIP, and VAE. Set `lightx2v_path`, `model_path`, and `ASCEND_RT_VISIBLE_DEVICES` at the top of all three NPU scripts. The base config is `configs/wan/residency/baseline_npu.json`.

```bash
bash scripts/wan/inference/offload_residency/estimate_npu.sh
bash scripts/wan/inference/offload_residency/run_npu.sh
bash scripts/wan/inference/offload_residency/run_baseline_npu.sh
```

## Changing inputs and settings

Set the input image with `--image-path` in the estimation script and `--image_path` in both inference scripts. Use the same image in all three scripts. Keep the positive prompt, negative prompt, and `seed` consistent between the two inference scripts as well.

The defaults are 81 frames and 40 inference steps. To change the frame count, dimensions, or step count, edit `num_frames`, `size`, and `infer_steps` in your platform's base config, then rerun estimation to generate a matching residency config.

`cpu_offload_layout` defaults to `per_tensor`. To use contiguous weight transfers, change it to `contiguous` in the base config and rerun estimation. Use the same layout for the baseline and the residency strategy.

## Results and comparison

Files are saved in `save_results/wan_residency/`. Running the scripts again overwrites results with the same names.

| Output | NVIDIA GPU | Ascend NPU |
| --- | --- | --- |
| Generated residency config | `resident.json` | `resident_npu.json` |
| Measurement report | `resident.profile.json` | `resident_npu.profile.json` |
| Video with resident weights | `resident.mp4` | `resident_npu.mp4` |
| Baseline video | `baseline.mp4` | `baseline_npu.mp4` |

In the generated config, `offload_resident_blocks` is the number of resident blocks. In the measurement report, `resident_blocks` and `streamed_blocks` are the numbers of blocks kept on the device and transferred as needed, respectively. `compute_ms` and `transfer_ms` are one block's computation and transfer times in milliseconds. A result of zero resident blocks is valid: the tool estimates that the current input does not require additional resident weights.

To compare timings, set `PROFILING_DEBUG_LEVEL` to `1` in both inference scripts and run them sequentially on the same device. Check the per-step `Run Dit every step` timings in the logs. The first step includes initialization overhead; use subsequent steps to compare steady-state performance.

Keeping more blocks resident uses more device memory. The tool does not automatically reduce the count based on available memory. To reduce memory use, lower `offload_resident_blocks` in the generated config. Rerunning estimation overwrites manual changes.

Rerun estimation after changing the device, model weights, quantization, attention implementation, input image, frame count, dimensions, or transfer layout. The estimate selects a resident block count; use inference comparisons to measure the actual speedup.
