# Qwen-Image-2512 CPU Block Offload Guide

English | [简体中文](README_CN.md)

Launcher: `qwen_image_2512_block_shared_offload.sh`.
Configuration: `configs/qwen_image/offload/qwen_image_2512_block_shared.json`.
Defaults: **T2I, host sharing, 8 GPUs, Ulysses SP8**.

Edit paths, GPU selection, and inference arguments directly in the script; edit parallel settings in the JSON file. The script sets its arguments explicitly, does not infer the GPU count, and does not forward arguments appended to `bash script.sh`. Environment variables such as `TASK`, `CONFIG_JSON`, `SHARED_CPU_WEIGHT_SCOPE`, and `QWEN_*` do not override this script. To change options without editing files, run the full `python -m torch.distributed.run ... -m lightx2v.infer ...` command directly.

## Setup and launch

1. Activate a Python environment with the project dependencies installed. The default attention backend is FlashAttention 3.
2. Set `lightx2v_path` and `model_path` at the top of the script to your project and model directories. Use the original BF16 Diffusers weights, including the transformer, text encoder, tokenizer, and VAE components.
3. Run from the project root:

```bash
cd /path/to/LightX2V
bash scripts/qwen_image/offload/qwen_image_2512_block_shared_offload.sh
```

## Change the GPU count

Update `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` in the script, and `parallel.seq_p_size` in the JSON. All three must specify the same GPU count:

| GPUs | `CUDA_VISIBLE_DEVICES` | `--nproc_per_node` | `parallel.seq_p_size` |
| --- | --- | --- | --- |
| 8 (default) | `0,1,2,3,4,5,6,7` | `8` | `8` |
| 4 | `0,1,2,3` | `4` | `4` |
| 2 | `2,5` | `2` | `2` |
| 1 | `0` | `1` | `1` |

For example, to use GPUs 2 and 5, set `export CUDA_VISIBLE_DEVICES=2,5` and `--nproc_per_node=2` in the script, and replace the JSON `parallel` object with:

```json
{
  "seq_p_size": 2,
  "seq_p_attn_type": "ulysses",
  "cfg_p_size": 1
}
```

Keep the other configuration fields. Launch with the same `bash` command afterward. Edit the GPU list inside the script; setting it in the environment before calling the script will not override it.

The model has 24 attention heads. Ulysses SP must divide 24, giving GPU counts of **1, 2, 3, 4, 6, 8, 12, or 24**. These are head-divisibility constraints, not a claim that every layout has been validated. This launcher runs on a single node and requires enough visible GPUs, host memory, and GPU memory. This configuration does not use TP or CFG parallelism; CFG guidance remains enabled.

## Choose host or NUMA sharing

The Python command in the script defaults to `--shared_cpu_weight_scope host`. To use NUMA sharing, replace that argument with the following fragment; it is not a standalone command:

```bash
  --shared_cpu_weight_scope numa \
```

The JSON does not need this field. The CLI argument takes precedence over the same setting in an older JSON file.

Host scope shares one copy of compatible CPU block weights within the same host and IPC namespace. NUMA scope creates copies for the NUMA domains of the participating GPUs.

## Task and other options

The current BF16 shared-weight adapter supports only `--task t2i`. Changing the task name does not enable `i2i` or layered generation; image editing requires a separate integration.

| Where to edit | Option | Default / purpose |
| --- | --- | --- |
| Shell script | `lightx2v_path`, `model_path` | Project and model directories |
| Inference command | `--prompt`, `--negative_prompt` | Prompt and negative prompt |
| Inference command | `--save_result_path`, `--seed` | Output file and random seed (default 42) |
| Inference command | `--shared_cpu_weight_scope` | `host` (default) or `numa` |
| Inference command | `--config_json` | Configuration file path |
| JSON | `infer_steps` | `50` |
| JSON | `aspect_ratio` | `"16:9"` |
| JSON | `sample_guide_scale` | `4.0` |

Sharing requires Linux SysV shared memory and CUDA pinned memory. Strict NUMA binding failures raise an error. The current shared entry supports original BF16 weights, block granularity, and `NoCaching`; it does not support TP, quantization, LoRA, layered generation, or lazy loading. The text encoder and VAE are loaded separately by each process. GPU activations and work buffers also remain private to each process.
