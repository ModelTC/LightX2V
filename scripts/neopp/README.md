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
  NeoPP converts this existing LightLLM argument to `size`; new LightX2V calls
  use `size=[height, width]`. Omitting `seed` also preserves the current RNG state;
  an explicit integer seed starts a seeded generation.

NeoPP defaults an omitted `task` to `t2i`; both tasks share the same generation
path with conditioning supplied through KV. LightLLM calls remain unchanged.

An explicit constructor `task`, such as `LightX2VPipeline(..., task="t2i")`, is the
default for calls that omit it. Passing `generate(task="i2i", ...)` selects a task
for that request without changing the default.

Default image editing also uses `set_kvcache`: LightLLM includes image
conditioning in the two KV branches. The optional `image_guidance_scale != 1`
mode uses a separate three-branch interface that is not implemented by this
LightX2V runner; see the [upstream routing](https://github.com/ModelTC/LightLLM/blob/fb7838ce86217a7c86625d71b15b72f2f05e0ac2/lightllm/server/httpserver/manager.py#L527).

The replay tools below are for backend development with KV captured from LightLLM.
Use the checkpoint, KV branches and position offsets from the same captured request.
Set `task="t2i"` or `task="i2i"` to match that request. Image editing KV must include
the input image conditioning. `save_result_for_debug=True` writes generated images.

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

### Single-request and multi-turn replay

Run from the LightX2V repository root after installing the package. Create the output
directory and replace all `/path/to/...` placeholders before running.

- [replay_kv.py](replay_kv.py) accepts explicit model/config/KV paths, offsets, task,
  seed and output size. It is used by the Shell launchers below. Sampling parameters
  come from the JSON; omitted `set_inference_params` fields retain the runner defaults
  (in particular, `cfg_norm="global"`).
- [replay_kv_multiturn.py](replay_kv_multiturn.py) loads one pipeline and replays three
  requests. Edit `model_path`, `config_path`, `kv_dir`, `output_dir`, `output_prefix`,
  `size` and `turns`. Results use `<output_dir>/<output_prefix>_<turn>.png`.
  It preserves the dense 1K example's explicit `cfg_norm="none"`, CFG interval
  `(-1, 2)`, scale `4.0`, shift `3.0` and seeds `200, 201, 202`.

```bash
python scripts/neopp/replay_kv_multiturn.py
```

Select a matching checkpoint and one of these existing configurations:

| Mode | Config under `configs/neopp/` | GPUs |
| --- | --- | --- |
| Dense | [neopp_dense.json](../../configs/neopp/neopp_dense.json) | 1 |
| Dense FP8 | [neopp_dense_fp8.json](../../configs/neopp/neopp_dense_fp8.json) | 1 |
| Dense CFG | [neopp_dense_parallel_cfg.json](../../configs/neopp/neopp_dense_parallel_cfg.json) | 2 |
| Dense CFG + sequence parallel | [neopp_dense_parallel_cfg_seq.json](../../configs/neopp/neopp_dense_parallel_cfg_seq.json) | 4 |
| Dense 8-step LoRA | [neopp_dense_8steps.json](../../configs/neopp/neopp_dense_8steps.json) | 1 |
| MoE | [neopp_moe.json](../../configs/neopp/neopp_moe.json) | 1 |

For parallel replay, set `config_path` in the multi-turn file, then use `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/neopp/replay_kv_multiturn.py
```

CFG + sequence parallel uses four processes instead. The process count must match
the product of the parallel sizes in the selected JSON.

To reproduce the previous 2K dense replay, set `size=[2048, 2048]`, select the 2K KV
directory, and use these turns. The offsets are captured values, not filename-derived:

```python
turns = [
    ("to_x2v_cond_kv_0_298.pt", "to_x2v_uncond_kv_0_9.pt", 298, 9, 200),
    ("to_x2v_cond_kv_1_360.pt", "to_x2v_uncond_kv_1_12.pt", 366, 12, None),
    ("to_x2v_cond_kv_2_439.pt", "to_x2v_uncond_kv_2_15.pt", 441, 15, None),
]
```

The original BF16 2K replay continued RNG state on turns 1 and 2; the FP8 and parallel
2K examples explicitly seeded them with `201` and `202`. Preserve the intended seed
policy when switching configurations. Set a distinct `output_dir` or `output_prefix`
for each run to keep previous results.

The 8-step example uses the matching LoRA config, size `[2048, 2048]`, and only
`("to_x2v_cond_kv_0_298.pt", None, 298, None, 200)` in `turns`; CFG is disabled.
The previous MoE examples use one turn with `to_x2v_cond_kv_325.pt`,
`to_x2v_uncond_kv_9.pt`, offsets `325` / `9`, seed `200`, and size `[1024, 1024]`
or `[2048, 2048]`. Set `cfg_interval=(0, 1)` and retain `cfg_norm="none"` in
`set_inference_params` for those examples. These explicit overrides differ from
the defaults of the existing MoE JSON and single-request launcher.

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
`--seed` sets the seed; omitting it preserves the current RNG state.

`run_neopp_dense_i2i_1k_cfg3.sh` has been removed because the current runner does
not implement three-branch image guidance. Use the supported two-branch image
editing path above with `image_guidance_scale=1` in LightLLM.
