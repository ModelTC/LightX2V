# Qwen Image CPU Offload with Resident Weights

English | [简体中文](README_CN.md)

Run Qwen Image text-to-image generation on a single device and compare two ways of managing block weights: `baseline` keeps all block weights in CPU memory and transfers them to device memory as needed; the residency strategy keeps some block weights in device memory throughout denoising to reduce repeated transfers.

Run `estimate` first to measure one block's computation and transfer times and generate a config with an estimated resident block count. Then run `run` for inference. Resident blocks are distributed evenly across the model's layers. Use `run_baseline` for comparison, or run it on its own.

Install the project dependencies for your platform, then run the commands below from the repository root.

## NVIDIA GPU

Use the BF16 weights for `Qwen-Image-2512`. The model directory should contain `transformer`, `text_encoder`, `tokenizer`, `scheduler`, and `vae`. Set `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES` at the top of all three scripts, keeping the values consistent.

The base config is `configs/qwen_image/residency/baseline.json`.

```bash
bash scripts/qwen_image/inference/offload_residency/estimate.sh
bash scripts/qwen_image/inference/offload_residency/run.sh
bash scripts/qwen_image/inference/offload_residency/run_baseline.sh
```

## Ascend NPU

Use the BF16 weights for the same model. Set `lightx2v_path`, `model_path`, and `ASCEND_RT_VISIBLE_DEVICES` at the top of all three NPU scripts. The base config is `configs/qwen_image/residency/baseline_npu.json`.

```bash
bash scripts/qwen_image/inference/offload_residency/estimate_npu.sh
bash scripts/qwen_image/inference/offload_residency/run_npu.sh
bash scripts/qwen_image/inference/offload_residency/run_baseline_npu.sh
```

## Changing inputs and settings

The defaults are a coffee shop prompt, a 16:9 aspect ratio, 50 steps, and `seed=42`. When changing the positive or negative prompt, update the estimation script and both inference scripts: estimation constructs inputs based on prompt length. Keep `seed` consistent between the two inference scripts.

To change the aspect ratio or step count, edit `aspect_ratio` and `infer_steps` in your platform's base config, then rerun estimation to generate a matching residency config.

`cpu_offload_layout` defaults to `per_tensor`. To use contiguous weight transfers, change it to `contiguous` in the base config and rerun estimation. Use the same layout for the baseline and the residency strategy.

These scripts are for text-to-image generation on a single device. They do not support image editing or layered generation.

## Results and comparison

Files are saved in `save_results/qwen_residency/`. Running the scripts again overwrites results with the same names.

| Output | NVIDIA GPU | Ascend NPU |
| --- | --- | --- |
| Generated residency config | `resident.json` | `resident_npu.json` |
| Measurement report | `resident.profile.json` | `resident_npu.profile.json` |
| Image with resident weights | `resident.png` | `resident_npu.png` |
| Baseline image | `baseline.png` | `baseline_npu.png` |

In the generated config, `offload_resident_blocks` is the number of resident blocks. In the measurement report, `resident_blocks` and `streamed_blocks` are the numbers of blocks kept on the device and transferred as needed, respectively. `compute_ms` and `transfer_ms` are one block's computation and transfer times in milliseconds. A result of zero resident blocks is valid: the tool estimates that the current input does not require additional resident weights.

To compare timings, set `PROFILING_DEBUG_LEVEL` to `1` in both inference scripts and run them sequentially on the same device. Check the per-step `infer_main` timings in the logs. These include both positive and negative DiT passes and weight transfers, but exclude text encoding and VAE decoding. The first step includes initialization overhead; use subsequent steps to compare steady-state performance.

Keeping more blocks resident uses more device memory. The tool does not automatically reduce the count based on available memory. To reduce memory use, lower `offload_resident_blocks` in the generated config. Rerunning estimation overwrites manual changes.

Rerun estimation after changing the device, model weights, quantization, attention implementation, prompts, image dimensions, or transfer layout. The estimate selects a resident block count; use inference comparisons to measure the actual speedup.
