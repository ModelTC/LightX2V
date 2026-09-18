# Hunyuan Image 3.0 CPU Block Offload Guide

English | [简体中文](README_CN.md)

Launcher: `run_hunyuan_image3_block_shared_offload.sh`.
Configuration: `configs/hunyuan_image3/offload/hunyuan_image3_block_shared.json`.
Defaults: **T2I, host sharing, 8 GPUs, TP2 × SP2 × CFG2**.

Edit paths, GPU selection, and inference arguments directly in the script; edit parallel settings in the JSON file. The script sets its arguments explicitly, does not infer the GPU count, and does not forward arguments appended to `bash script.sh`. Environment variables such as `TASK`, `CONFIG_JSON`, `SHARED_CPU_WEIGHT_SCOPE`, and `HUNYUAN_IMAGE3_MODEL_PATH` do not override the script. Set the upstream code path using `HUNYUAN_IMAGE3_REPO_PATH` inside the script. To change options without editing files, run the full `python -m torch.distributed.run ... -m lightx2v.infer ...` command directly.

## Setup and launch

1. Activate a Python environment with the project dependencies installed.
2. Set the three paths at the top of the script:
   - `lightx2v_path`: the LightX2V project directory.
   - `model_path`: the original BF16 HunyuanImage-3-Instruct checkpoint directory containing `model.safetensors.index.json`.
   - `HUNYUAN_IMAGE3_REPO_PATH`: the upstream HunyuanImage-3.0 code directory providing tokenizer, image processing, VAE, and vision modules.
3. Run from the project root:

```bash
cd /path/to/LightX2V
bash scripts/hunyuan_image3/offload/run_hunyuan_image3_block_shared_offload.sh
```

The defaults enable FlashInfer MoE, KV caching, 50 inference steps, and `think_recaption`. The environment needs FlashInfer's `cutlass_fused_moe`, `ActivationType`, and an autotune interface that accepts cache arguments.

## Change the GPU count

Update `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` in the script together with the JSON `parallel` settings. The process count must equal both the visible GPU count and TP×SP×CFG.

| GPUs / processes | Example GPU list | `tensor_p_size` | `seq_p_size` | `cfg_p_size` | `cfg_mode` |
| --- | --- | --- | --- | --- | --- |
| 1 | `0` | `1` | `1` | `1` | `"serial"` |
| 2 | `2,5` | `2` | `1` | `1` | `"serial"` |
| 4 | `0,1,2,3` | `2` | `2` | `1` | `"serial"` |
| 8 (default) | `0,1,2,3,4,5,6,7` | `2` | `2` | `2` | `"parallel"` |
| 16 | `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15` | `2` | `4` | `2` | `"parallel"` |

For example, to use GPUs 2 and 5, set `export CUDA_VISIBLE_DEVICES=2,5` and `--nproc_per_node=2` in the script, and replace the JSON `parallel` object with:

```json
{
  "pipeline_parallel": false,
  "tensor_p_size": 2,
  "seq_p_size": 1,
  "cfg_p_size": 1,
  "seq_p_attn_type": "ulysses",
  "cfg_mode": "serial"
}
```

Keep the other settings, including `enable_cfg=true`. With CFG=1, the two guidance branches run sequentially. Launch with the same `bash` command afterward. Edit the GPU list inside the script; setting it in the environment before calling the script will not override it.

Also set `flashinfer_autotune_cache` to a separate path for the task and parallel layout, such as `save_results/hunyuan_image3_flashinfer_autotune_t2i_tp2_sp1_cfg1.json` for two-GPU T2I. A new cache may require tuning on the first run.

The **1, 2, 4, 8, and 16 GPU** layouts above satisfy head-divisibility constraints; they are not a claim that every layout has been validated. The model has 32 Q heads and 8 KV heads, and Ulysses requires TP×SP to divide both counts. This launcher runs on a single node and needs the corresponding number of visible GPUs, with enough GPU memory for block buffers, MoE workspaces, KV caches, and other model components.

## Choose host or NUMA sharing

The Python command in the script defaults to `--shared_cpu_weight_scope host`. To use NUMA sharing, replace that argument with the following fragment; it is not a standalone command:

```bash
  --shared_cpu_weight_scope numa \
```

The JSON does not need this field. The CLI argument takes precedence over the same setting in an older JSON file.

Host scope shares compatible TP shards of transformer block CPU weights within the same host and IPC namespace. NUMA scope creates the corresponding copies for the NUMA domains of the participating GPUs.

## Switch to TI2I reference-image editing

1. Change `--task t2i` to `--task ti2i` in the launch command.
2. Add the reference image argument:

```bash
  --image_path "${HUNYUAN_IMAGE3_REPO_PATH}/assets/demo_instruct_imgs/input_0_0.png" \
```

3. Set `--prompt` to your editing instruction and update `--save_result_path`.

With the JSON unchanged and no `--size` argument, TI2I still outputs **1024×1024**; it does not automatically adopt the reference image dimensions. To align the output size with the reference, you can remove `size` from the JSON and add `"image_size": "auto"` and `"align_image_size": true`.

Use a separate `flashinfer_autotune_cache` path for TI2I, such as `save_results/hunyuan_image3_flashinfer_autotune_ti2i_tp2_sp2_cfg2.json` for the default eight-GPU layout. This keeps task-specific tuning caches separate; the task itself is selected by `--task`.

To return to default T2I, change `--task` back to `t2i` and remove `--image_path`. If you changed the JSON, remove `image_size` and `align_image_size`, restore `"size": [1024, 1024]`, and select the T2I cache path.

## Other options

| Where to edit | Option | Default / purpose |
| --- | --- | --- |
| Shell script | Three path variables | Project, checkpoint, and upstream code directories |
| Shell script | `OMP_NUM_THREADS` | `1` |
| Inference command | `--prompt`, `--save_result_path`, `--seed` | Prompt, output file, random seed (default 42) |
| Inference command | `--shared_cpu_weight_scope` | `host` (default) or `numa` |
| Inference command | `--config_json` | Path to another configuration file |
| JSON | `infer_steps`, `size` | 50 steps, default T2I size `[1024, 1024]` in height/width order |
| JSON | `bot_task`, `max_new_tokens` | `"think_recaption"`, `2048`; longer reasoning text increases runtime |
| JSON | `flashinfer_autotune_cache` | Cache file matching the task and TP/SP/CFG layout |

To limit reasoning text, set `max_new_tokens` in the JSON to a suitable value such as `512`, or add `--max_new_tokens 512` to the script's Python command. Run from the project root. The program resolves relative `flashinfer_autotune_cache` paths against the LightX2V project root.

Sharing requires Linux SysV shared memory and CUDA pinned memory. Strict NUMA binding failures raise an error. Pre/post weights, the VAE, vision encoder, and GPU KV caches/activations remain private to each process. The current shared path does not support quantization, LoRA, lazy loading, compile, CUDA Graph, or pipeline execution across GPUs.
