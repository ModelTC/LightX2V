# MiniMax-H3 CPU Block Offload Guide

English | [简体中文](README_CN.md)

Launcher: `run_minimax_h3_block_shared_offload.sh`.
Configuration: `configs/minimax_h3/offload/minimax_h3_block_shared_offload.json`.
Defaults: **T2AV, host sharing, 8 GPUs, Ulysses SP8**.

Edit paths, GPU selection, and inference arguments directly in the script; edit sharing scope and parallel settings in the JSON file. The script sets its arguments explicitly, does not infer the GPU count, and does not forward arguments appended to `bash script.sh`. Environment variables such as `TASK`, `SHARED_CPU_WEIGHT_SCOPE`, and `MINIMAX_H3_*` do not override this inference script. To change command-line options without editing the script, run the full `python -m torch.distributed.run ... -m lightx2v.infer ...` command directly.

## Setup and launch

1. Activate a Python environment with the project dependencies installed.
2. Set `lightx2v_path` and `model_path` at the top of the script to your project and checkpoint directories.
3. Generate a matching AdaLN cache as described below, unless one already exists.
4. Run from the project root:

```bash
cd /path/to/LightX2V
bash scripts/minimax_h3/offload/run_minimax_h3_block_shared_offload.sh
```

Base tasks require `transformer/`, `text_encoder/`, `tokenizer/`, `processor/`, `vae/`, and `audio_vae/` in the model directory. The dedicated Ref2AV variant also requires `transformer_ref/`. The default setup uses original BF16 weights, SageAttention2, SGL, and Triton kernels.

## Prepare the AdaLN cache

The current H3 shared block offload integration requires a matching AdaLN cache. Run the following from the project root, using the same model path as the inference script:

```bash
CUDA_VISIBLE_DEVICES=0 \
MINIMAX_H3_MODEL_PATH=/path/to/MiniMax-H3 \
MINIMAX_H3_CACHE_TASK=fl2av \
MINIMAX_H3_CONFIG="$PWD/configs/minimax_h3/offload/minimax_h3_block_shared_offload.json" \
bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
```

The cache tool accepts these environment variables and uses one GPU. Match the cache to `--model-variant`: `fl2av` weights use an `fl2av` cache, and `ref2av` weights use a `ref2av` cache. Set `MINIMAX_H3_CACHE_TASK=ref2av` to generate the latter.

Defaults are 29 steps, video/audio flow shifts of 12/3, and cache directory `~/.cache/lightx2v/adaln`. Cache generation and inference must use matching weights, `infer_steps`, flow shifts, and `adaln_cache_dir`. Changing only the GPU count or host/NUMA scope does not require rebuilding the cache.

## Change the GPU count

Update `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` in the script, and `parallel.seq_p_size` in the JSON. All three must specify the same GPU count.

| GPUs | Script `CUDA_VISIBLE_DEVICES` | `--nproc_per_node` | JSON `parallel.seq_p_size` | Example JSON `vae_decode_parallel` |
| --- | --- | --- | --- | --- |
| 8 (default) | `0,1,2,3,4,5,6,7` | `8` | `8` | `true` |
| 4 | `0,1,2,3` | `4` | `4` | `false` |
| 2 | `2,5` | `2` | `2` | `false` |
| 1 | `0` | `1` | `1` | `false` |

For example, to use GPUs 2 and 5, set `export CUDA_VISIBLE_DEVICES=2,5` and `--nproc_per_node=2` in the script, then update these JSON fields:

```json
{
  "parallel": {
    "seq_p_size": 2,
    "seq_p_attn_type": "ulysses"
  },
  "vae_decode_parallel": false
}
```

This is a partial configuration; keep the other fields. Launch with the same `bash` command afterward. The script sets its own GPU list, so edit that list instead of prefixing the command with a different `CUDA_VISIBLE_DEVICES` value.

H3 has 56 attention heads. Ulysses SP must divide 56, giving GPU counts of **1, 2, 4, 7, 8, 14, 28, or 56**. These are head-divisibility constraints, not a claim that every layout has been validated. This launcher runs on a single node and requires enough visible GPUs, host memory, and GPU memory.

The non-8-GPU examples disable VAE parallelism to start with serial decoding. The current VAE distributes spatiotemporal tiles and is not limited to 8 GPUs: `vae_decode_parallel` can remain `true` on multiple GPUs, and the runner disables it automatically for a single GPU.

## Choose host or NUMA sharing

The JSON configuration defaults to `"shared_cpu_weight_scope": "host"`. To use NUMA sharing, change this field in the same JSON file:

```json
{
  "shared_cpu_weight_scope": "numa"
}
```

Keep the other JSON fields unchanged and run the same launcher. Sharing scope is a startup configuration setting, not a command-line or per-request option.

Host scope shares one copy of compatible CPU weights within the same host and IPC namespace. NUMA scope creates copies for the NUMA domains of the participating GPUs. Both use the same launcher.

## Change the task and inputs

Edit `--task` and `--model-variant` in the script's Python command, and add the required input arguments:

| `--task` | `--model-variant` | Input arguments |
| --- | --- | --- |
| `t2av` (default) | `fl2av` | None |
| `i2av` | `fl2av` | `--image_path /path/to/first.png` |
| `l2av` | `fl2av` | `--last_frame_path /path/to/last.png` |
| `fl2av` | `fl2av` | Both `--image_path` and `--last_frame_path` |
| `ref2av` | `ref2av` | `--image_path` or `--video_path`, optionally with `--audio_path` |

The Ref2AV example uses the dedicated `transformer_ref/` weights. The runner also allows `--model-variant fl2av --task ref2av` with the base transformer. That combination requires the `fl2av` cache: do not choose the cache by task name alone.

For Ref2AV, replace the corresponding arguments in the launch command with the following, keeping the remaining arguments:

```bash
  --model-variant ref2av \
  --task ref2av \
  --image_path "${lightx2v_path}/assets/inputs/imgs/img_0.jpg" \
```

Ref2AV accepts comma-separated paths for multiple references of the same type. Audio references must accompany images or video. The script does not supply a reference image automatically. Adjust `--prompt` and `--save_result_path` when changing tasks.

## Other options

| Where to edit | Option | Default / purpose |
| --- | --- | --- |
| Shell script | `lightx2v_path`, `model_path` | Project and checkpoint directories |
| Inference command | `--prompt`, `--seed`, `--save_result_path` | Prompt, random seed (default 42), output file |
| JSON | `shared_cpu_weight_scope` | `host` (default) or `numa` |
| Inference command | `--config_json` | Path to another configuration file |
| JSON | `infer_steps` | `29` |
| JSON | `size` | `[544, 960]`, in height/width order |
| JSON | `num_frames`, `fps` | `124` frames, `24` fps |
| JSON | `adaln_cache_dir` | `~/.cache/lightx2v/adaln` |

The shared configuration covers the DiT, text backbone, and video VAE, with block offload enabled for each. Sharing requires Linux SysV shared memory and CUDA pinned memory. GPU activations and work buffers remain private to each process. The current shared path does not support TP, quantization, LoRA, compile, or lazy loading.

The shared video VAE uses `vae_encoder_conv_mode="torch"`. Channels-last and FP8 Conv3D layouts are not represented in its shared manifest and cannot be used with this shared loading path.
