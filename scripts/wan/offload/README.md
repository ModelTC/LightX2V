# Wan 2.1 CPU Block Offload Guide

English | [简体中文](README_CN.md)

Launcher: `run_wan_block_shared_offload.sh`.
Configuration: `configs/offload/block/wan_block_shared.json`.
Defaults: **I2V, host sharing, 8 GPUs, Ulysses SP8**.

Edit paths, GPU selection, and inference arguments directly in the script; edit parallel settings in the JSON file. The script sets its arguments explicitly, does not infer the GPU count, and does not forward arguments appended to `bash script.sh`. Environment variables such as `TASK`, `CONFIG_JSON`, `SHARED_CPU_WEIGHT_SCOPE`, and `WAN_*` do not override this script. To change options without editing files, run the full `python -m torch.distributed.run ... -m lightx2v.infer ...` command directly.

## Setup and launch

1. Activate a Python environment with the project dependencies installed.
2. Set `lightx2v_path` and `model_path` at the top of the script to your project and checkpoint directories.
3. Set `dit_quantized_ckpt`, `t5_quantized_ckpt`, `clip_quantized_ckpt`, and `vae_path` in the JSON to the actual checkpoint locations. Changing only `model_path` in the script does not update these paths.
4. Run from the project root:

```bash
cd /path/to/LightX2V
bash scripts/wan/offload/run_wan_block_shared_offload.sh
```

The default configuration requires model configuration files, FP8-vLLM DiT `block_*.safetensors`, FP8 T5/CLIP weights, and `Wan2.1_VAE.pth`. It uses FP16 inference and FlashAttention 3. Relative checkpoint paths in the JSON are resolved from the working directory, so launch from the project root as shown above, or use absolute paths.

## Change the GPU count

Update `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` in the script, and `parallel.seq_p_size` in the JSON. All three must specify the same GPU count:

| GPUs | `CUDA_VISIBLE_DEVICES` | `--nproc_per_node` | `parallel.seq_p_size` | Example `parallel.vae_parallel` |
| --- | --- | --- | --- | --- |
| 8 (default) | `0,1,2,3,4,5,6,7` | `8` | `8` | `true` |
| 4 | `0,1,2,3` | `4` | `4` | `false` |
| 2 | `2,5` | `2` | `2` | `false` |
| 1 | `0` | `1` | `1` | `false` |

For example, to use GPUs 2 and 5, set `export CUDA_VISIBLE_DEVICES=2,5` and `--nproc_per_node=2` in the script, and replace the JSON `parallel` object with:

```json
{
  "seq_p_size": 2,
  "seq_p_attn_type": "ulysses",
  "cfg_p_size": 1,
  "vae_parallel": false
}
```

Keep the other configuration fields. Launch with the same `bash` command afterward. Edit the GPU list inside the script; setting it in the environment before calling the script will not override it.

The default 14B model has 40 attention heads. Ulysses SP must divide 40, giving GPU counts of **1, 2, 4, 5, 8, 10, 20, or 40**. These are head-divisibility constraints, not a claim that every layout has been validated. This launcher runs on a single node and requires enough visible GPUs, host memory, and GPU memory. This configuration does not use TP or CFG parallelism; CFG guidance remains enabled.

The non-8-GPU examples set `parallel.vae_parallel=false` to start with serial VAE execution. This is an example setting, not an 8-GPU restriction on the VAE. Enabling VAE parallelism also depends on the spatial partitioning of the image and latent dimensions.

## Choose host or NUMA sharing

The Python command in the script defaults to `--shared_cpu_weight_scope host`. To use NUMA sharing, replace that argument with the following fragment; it is not a standalone command:

```bash
  --shared_cpu_weight_scope numa \
```

The JSON does not need this field. The CLI argument takes precedence over the same setting in an older JSON file.

Host scope shares one copy of compatible DiT block CPU weights within the same host and IPC namespace. NUMA scope creates copies for the NUMA domains of the participating GPUs.

## Change the task and inputs

The default task is `--task i2v`, with the reference image specified by `--image_path`. Edit `--prompt`, `--negative_prompt`, and `--save_result_path` to change the prompts and output location.

To use T2V:

1. Change `--task i2v` to `--task t2v` and remove the `--image_path` line.
2. Set `model_path` to the corresponding T2V model directory.
3. Prepare a matching T2V shared offload JSON and update `--config_json`. You can copy the default JSON, replace the DiT, T5, and VAE paths, and remove the I2V-specific CLIP settings.
4. Update the prompts and output filename.

The shared adapter requires FP8-vLLM block weights. Changing only the task name while keeping the I2V checkpoint is insufficient. The GPU-count list above applies to the 14B model; full T2V generation must be checked with the actual T2V weights.

## Other options

| Where to edit | Option | Default / purpose |
| --- | --- | --- |
| Shell script | `lightx2v_path`, `model_path` | Project and model directories |
| Inference command | `--image_path`, `--prompt`, `--negative_prompt` | First frame and prompts |
| Inference command | `--save_result_path`, `--seed` | Output file and random seed (default 42) |
| Inference command | `--shared_cpu_weight_scope` | `host` (default) or `numa` |
| Inference command | `--config_json` | Configuration file path |
| JSON | `infer_steps`, `num_frames`, `size` | 40 steps, 81 frames, `[480, 832]` in height/width order |

Sharing requires Linux SysV shared memory and CUDA pinned memory. Strict NUMA binding failures raise an error. The current shared path does not support TP, LoRA, or lazy loading. CPU weight sharing is not enabled for T5, CLIP, or the VAE. GPU activations and work buffers remain private to each process.
