# NeoPP: LightLLM integration and KV replay

NeoPP uses LightX2V as the image generation backend of LightLLM. LightLLM
encodes the text and input images and supplies the required KV cache and
position offsets. Start production inference and end-to-end debugging through
the [SenseNova-U1 deployment instructions](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/deployment.md).

The [LightLLM adapter](https://github.com/ModelTC/LightLLM/blob/fb7838ce86217a7c86625d71b15b72f2f05e0ac2/lightllm/server/x2i_server/lightx2v/adapter.py)
uses these LightX2V interfaces:

- `LightX2VPipeline(..., support_tasks=["t2i", "i2i"])` initializes the backend.
- `modify_config({"save_result_for_debug": False})` selects in-memory output.
  LightLLM replaces `process_images_after_vae_decoder` to return encoded image
  bytes, and wraps `_run_infer_step` to check cancellation.
- `runner.set_kvcache(...)` injects conditioning; `set_inference_params(...)`
  supplies matching position offsets, CFG settings and output format.
- `generate(seed=None, save_result_path="", target_shape=[height, width])`
  continues the RNG state restored by LightLLM for later images in a session.
  An explicit integer seed starts a seeded generation.

Default image editing also uses `set_kvcache`: LightLLM includes image
conditioning in the two KV branches. The optional `image_guidance_scale != 1`
mode uses a separate three-branch interface that is not implemented by this
LightX2V runner; see the [upstream routing](https://github.com/ModelTC/LightLLM/blob/fb7838ce86217a7c86625d71b15b72f2f05e0ac2/lightllm/server/httpserver/manager.py#L527).

The Python files in this directory replay KV dumps for backend development and
performance debugging. Replace their `/path/to/...` placeholders with the
checkpoint, KV files and output directory for your model and request. The
filenames and position offsets in the examples describe particular dumps;
update both to match your captured request. Each turn calls `load_kvcache(...)`
and `set_inference_params(...)` before `generate(...)`;
`save_result_for_debug=True` saves the image to the requested file.

### KV file layout

Online `set_kvcache(...)` accepts `[layers, 2, sequence_length, kv_heads, head_dim]`.
The existing replay files use `[layers, 2, kv_heads, sequence_length, head_dim]`;
`load_kvcache(...)` swaps dimensions 2 and 3 when loading them. To save an online
KV tensor in this replay format:

```python
torch.save(past_kv_cache.transpose(2, 3).cpu(), "/path/to/cond.pt")
```

Apply the same conversion to the unconditional KV when CFG is enabled. Existing
replay files already in this layout can be loaded unchanged. Record the position
offsets passed to `set_inference_params(...)` for the same request and use those
values for replay; do not infer them from the tensor dimensions.

### Run the replay examples

Create your chosen output directory and run from `examples/neopp`, since the
examples use relative config paths. Replace the paths in these commands too:

```bash
mkdir -p /path/to/save_results
cd /path/to/LightX2V/examples/neopp
```

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python neopp_dense_1k.py
```

The same launcher applies to `neopp_dense_2k.py`, `neopp_dense_1k_fp8.py`,
`neopp_dense_2k_fp8.py`, `neopp_dense_2k_8steps.py`, `neopp_moe_1k.py` and
`neopp_moe_2k.py`. Use the matching checkpoint and KV dumps for each variant.

Two GPUs with CFG parallelism (`cfg_p_size=2`):

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 neopp_dense_1k_parallel_cfg.py
```

Four GPUs with CFG and sequence parallelism (`cfg_p_size=2`, `seq_p_size=2`):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 neopp_dense_1k_parallel_cfg_seq.py
```

Use `neopp_dense_2k_parallel_cfg_seq.py` with the same four-GPU launcher for the
2K replay. The process count must match the parallel sizes in the config.

### Shell launchers

The `scripts/neopp/run_neopp_*.sh` launchers use `replay_kv.py` to generate one
image from an explicit KV dump. Edit the model path, KV paths, matching offsets,
seed, output size and output path directly in the shell script. The launchers
retain the shared environment setup in `scripts/base/base.sh`.

| Launcher | Replay mode | GPUs |
| --- | --- | --- |
| `run_neopp_dense_t2i_1k.sh` | Dense, 1024 × 1024 | 1 |
| `run_neopp_dense_i2i_1k.sh` | Dense image editing, 1024 × 1024 | 1 |
| `run_neopp_dense_t2i_1k_cfg2.sh` | Dense with CFG parallelism, 1024 × 1024 | 2 |
| `run_neopp_moe_t2i_1k.sh` | MoE, 1024 × 1024 | 1 |
| `run_neopp_moe_t2i_1k_cfg2.sh` | MoE with CFG parallelism, 1024 × 1024 | 2 |
| `run_neopp_moe_t2i_512.sh` | MoE, 512 × 512 | 1 |

After editing the paths and request values:

```bash
bash /path/to/LightX2V/scripts/neopp/run_neopp_dense_t2i_1k.sh
```

For image editing, capture KV from a LightLLM request that includes the input
images. CFG settings come from the selected JSON. In `replay_kv.py`, an explicit
seed overrides the JSON seed; if both are omitted, the default is 42.

`run_neopp_dense_i2i_1k_cfg3.sh` has been removed because the current runner does
not implement three-branch image guidance. Use the supported two-branch image
editing path above with `image_guidance_scale=1` in LightLLM.
