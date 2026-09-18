# Existing Model Adapter Case Studies

English | [简体中文](model-adapters.md)

These notes describe the current repository's implementation boundaries and entry points. They do not replace source inspection and runtime validation for a new task. See [implementation-patterns_en.md](implementation-patterns_en.md) for common contracts. Do not turn one model's quantization, task, or cache restrictions into limitations of the common backend.

## Choose the closest implementation

| Case | Checkpoint and sharing scope | Integration questions it helps answer |
|---|---|---|
| Wan | FP8-vllm DiT files split by block; non-block weights remain private | Quantized dtype/scale semantics, block checkpoints, shared adapters |
| Qwen Image | Sharded Diffusers BF16 weights; transformer blocks shared, pre/post and other private weights separate | Index/header consistency, manifests spanning shards, strict operator transpose validation |
| Hunyuan Image 3.0 | Original indexed BF16 checkpoint; shared arena contains all storage-TP slices | TP/SP/CFG combinations, fused MoE buffers, heterogeneous block slots, logical KV-cache layer indices |
| MiniMax H3 DiT | Original safetensors; selected transformer blocks shared | AdaLN cache filtering, GPU buffer release/rebuild, multicomponent initialization order |
| MiniMax H3 text | Qwen3-VL text-prefix weights, including embeddings and selected layers | Additional shared components, event slots, restoring shared sources |
| MiniMax H3 video VAE | Selected static encoder/decoder parameters shared | Native nn.Module integration, mixed runtime dtypes, decoder blocks, multiple tiles |

Weight-sharing scope and block-offload scope can differ within one pipeline. In particular, distinguish the H3 video VAE encoder from its decoder.

## Wan: FP8-vllm adapter

Source entry points:

- [model.py](../../../../lightx2v/models/networks/wan/model.py): `WanModel._load_shared_cpu_weights()`.
- [shared_block_weights.py](../../../../lightx2v/models/networks/wan/shared_block_weights.py): `WanFp8VllmSharedBlockAdapter`.
- `lightx2v/models/networks/wan/weights/transformer_weights.py`: block weights and CUDA buffers.
- `lightx2v/models/networks/wan/infer/offload/transformer_infer.py`: `infer_with_blocks_offload()`.

Call order: the adapter checks configuration and `block_*.safetensors` and constructs a manifest; the model loads private `non_block.safetensors` weights, coordinates preflight errors, calls `materialize()` and `build_weight_map()`, then performs common weight binding and validation.

Key functions:

- `_discover_block_files()` and `_inspect_checkpoint()` check layer indices, schemas, content signatures, and runtime dtypes.
- `_target_dtype()` determines final CPU storage types; `_populate()` reproduces existing loading and postprocessing semantics.
- `materialize()` passes scope, strict NUMA policy, and registration size to the common coordinator.

Preserve FP8 scale semantics. The current baseline first casts released FP32 checkpoint scales to the inference dtype, then stores them in FP32 staging tensors. Shared population must reproduce this rounding. Copying original FP32 scales directly into shared FP32 tensors changes results. FP8 weights themselves remain FP8.

The adapter requires `cpu_offload=true`, `offload_granularity=block`, `dit_quantized=true`, and `dit_quant_scheme=fp8-vllm`, with matching inference and sensitive dtypes. `_validate_config()` defines its TP, lazy-loading, online-quantization, and LoRA/adapter restrictions. FP8 checkpoint format and inference dtype are separate concepts: default BF16 inference does not establish support for sharing unquantized BF16 checkpoints or for every Wan family member.

The entry point is [scripts/wan/offload](../../../../scripts/wan/offload/), with `run_wan_block_shared_offload.sh` and `configs/offload/block/wan_block_shared.json`. Defaults are I2V, host sharing, and eight GPUs with SP8. Change `--shared_cpu_weight_scope host` to `numa` inside the script to switch modes. When changing GPU count, edit the GPU list, process count, and JSON parallel layout together. T2V also needs a matching T2V checkpoint, configuration, and input arguments.

The current script inherits the BF16 default from `base.sh`; `SENSITIVE_LAYER_DTYPE=None` follows the main dtype. Existing environment values can change these settings. Old FP16 validation records cover only that precision. Full BF16 inference after the default change requires separate evidence.

T5/CLIP quantization and DiT sharing are different capabilities in this example. Encoder quantization does not establish CPU weight sharing for those encoders.

## Qwen Image: Diffusers BF16 adapter

Source entry points:

- [model.py](../../../../lightx2v/models/networks/qwen_image/model.py): `_load_shared_cpu_weights()`, `_validate_shared_cpu_weights()`.
- [shared_block_weights.py](../../../../lightx2v/models/networks/qwen_image/shared_block_weights.py): `QwenBf16SharedBlockAdapter`, `validate_qwen_shared_block_views()`.
- `lightx2v/models/networks/qwen_image/infer/offload/transformer_infer.py`: block scheduling loop.

`_inspect_checkpoint()` builds a manifest from the safetensors index and shard headers, checking layer indices, completeness, duplicate keys, and block schemas. `load_private_weights()` loads only selected private weights. `_populate()` fills shared blocks. `materialize()` and `build_weight_map()` connect to the common protocol.

Current requirements include T2I, unquantized BF16 weights, both main and sensitive dtypes resolving to BF16, `feature_caching=NoCaching`, and block offload. The `base.sh` defaults of `DTYPE=BF16` and `SENSITIVE_LAYER_DTYPE=None` already satisfy the dtype requirement; both environment values do not need to be the literal string `BF16`. Layered generation, TP, lazy loading, LoRA/diff/adapter weights, and some checkpoint-path overrides are unsupported. Do not silently switch to T2I or remove an adapter to satisfy these constraints. Implement out-of-scope capabilities separately or report the limitation explicitly.

Beyond common address validation, `validate_qwen_shared_block_views()` uses operator `base_attrs` to verify the required transpose orientation. This catches layout errors that shape checks alone would miss for square matrices.

The entry point is [scripts/qwen_image/offload](../../../../scripts/qwen_image/offload/), with `qwen_image_2512_block_shared_offload.sh` and `configs/qwen_image/offload/qwen_image_2512_block_shared.json`. Defaults are T2I, host sharing, and eight GPUs with SP8. Select NUMA with `--shared_cpu_weight_scope` in the command. This shared adapter currently does not support I2I.

The example shares DiT blocks. It does not automatically enable sharing for the encoder or VAE. Each additional component requires its own loader, owner, lifecycle, and evidence.

## Hunyuan Image 3.0: TP slices and MoE block slots

Source entry points:

- [model.py](../../../../lightx2v/models/networks/hunyuan_image3/model.py): `_validate_offload_config()`, `_load_shared_cpu_weights()`, `_init_offload_manager()`, `close_shared_cpu_weights()`.
- [shared_block_weights.py](../../../../lightx2v/models/networks/hunyuan_image3/shared_block_weights.py): `HunyuanImage3SharedBlockAdapter`, `validate_hunyuan_shared_views()`.
- [offload.py](../../../../lightx2v/models/networks/hunyuan_image3/offload.py): `block_signature()`, `HunyuanImage3BlockSlot`, `HunyuanImage3BlockOffload`.

The adapter builds the same manifest containing every storage-TP slice from `model.safetensors.index.json` and shard headers. Different TP coordinates select their own regions; SP/CFG ranks with the same TP coordinate adopt the same CPU views. The signature includes TP size, micro-shard layout, and runtime dtypes. Leader population preserves checkpoint conversion order: router weights pass through the baseline loading dtype before their final FP32 conversion.

`block_signature()` groups GPU buffers by layout and MoE semantics, allocating two slots per compatible block family. Expert weights copy directly into each slot's fused MoE pack; no GPU pack is retained for every logical layer. Computation continues to address KV caches by logical layer index, never by reusable slot index. Ready/free events protect slots, and a completion event protects consecutive calls. Close the GPU offload manager before closing the shared owner.

The entry point is [scripts/hunyuan_image3/offload](../../../../scripts/hunyuan_image3/offload/), using `configs/hunyuan_image3/offload/hunyuan_image3_block_shared.json`. Defaults are T2I, host sharing, eight GPUs with TP2 × SP2 × CFG2, FlashInfer MoE, and KV caching. TI2I requires changing `--task` and adding `--image_path`. Keeping the JSON `size` setting still produces 1024×1024 output; matching the reference image requires changing size settings. Use distinct FlashInfer autotune caches for tasks and parallel layouts.

The runner resolves the upstream code directory from `HUNYUAN_IMAGE3_REPO_PATH` to provide tokenizer, VAE, and vision modules. The launcher does not need to append it to `PYTHONPATH`. Pre/post weights, the VAE, vision encoder, and GPU KV caches/activations remain private to each rank.

The current shared entry uses original BF16 weights. Quantization, LoRA, lazy loading, compile, AR CUDA Graph, and pipeline execution across GPUs are outside this path. Valid TP/SP/CFG combinations also depend on head divisibility and available GPU memory. Do not copy another model's no-TP restriction here or treat example layouts as runtime evidence.

## MiniMax H3: shared loading and memory management across phases

### DiT

Source entry points:

- [shared_block_weights.py](../../../../lightx2v/models/networks/minimax_h3/shared_block_weights.py): `load_h3_shared_weights()`, `load_shared_dit()`.
- [model.py](../../../../lightx2v/models/networks/minimax_h3/model.py): shared-loading override, `release_block_offload_buffers()`, `ensure_block_offload_buffers()`.
- `lightx2v/models/networks/minimax_h3/weights/transformer_weights.py` and `infer/offload/transformer_infer.py`: weight buffers and computation loop.

Argument contract for `load_h3_shared_weights()`:

| Argument | Purpose |
|---|---|
| `expected` | Validate exact selected checkpoint tensor names, shapes, and source dtypes |
| `runtime_dtypes` | Specify runtime dtypes in final shared CPU storage |
| `include` | Filter checkpoint tensors unused by this inference path |
| `shared` | Divide selected tensors into shared and private sets |
| `validate` | Check and coordinate component-specific conditions before materialization |

The helper constructs a manifest with meta tensors, loads private weights separately, and has the leader write shared tensors directly into destination views in `populate()`. It remains an H3-specific helper. New families should prefer the common coordinator; do not promote the H3 loader into a common API just to standardize documentation.

`load_shared_dit()` currently requires block offload and an AdaLN cache. It uses `model.remove_keys` to omit unused AdaLN/time and other weights and shares only `transformer_blocks.*`. TP, quantization, LoRA, lazy loading, compile, and other combinations are restricted by the H3 adapter.

DiT GPU buffers may be released during text encoding or VAE phases and rebuilt through `ensure_block_offload_buffers()` before denoising. Rebuilding uses CPU-source shapes, dtypes, and strides without reloading the whole checkpoint or closing the shared arena.

### Text encoder

The loading entry point is `load_shared_text_weights()` in [shared_weights.py](../../../../lightx2v/models/input_encoders/hf/minimax_h3/shared_weights.py); scheduling lives in `lightx2v/models/input_encoders/hf/minimax_h3/qwen3vl.py`.

Select embeddings and used layers through the text-prefix expected schema, then load them into BF16 shared storage. After `backbone.load()`, restore CPU sources, validate them, and retain `shared_cpu_weight_owner`. The current path requires unquantized text block offload without text TP.

Event-slot scheduling protects each layer's GPU buffers, and a completion event protects reuse across requests. After temporarily activating embeddings on GPU, restore their original pinned sources. Releasing GPU block buffers retains shared CPU weights. The vision tower is a separate component; its whole-module offload is not part of text weight sharing.

### Video VAE

Entry points are `load_shared_video_vae()` in [weights.py](../../../../lightx2v/models/video_encoders/hf/minimax_h3/weights.py), `from_pretrained()`/`_activate()` in `video_vae.py`, and `VideoVAEDecoderOffload` in [offload.py](../../../../lightx2v/models/video_encoders/hf/minimax_h3/offload.py).

Obtain expected checkpoint specifications from a meta model, then determine runtime dtypes with `_prepare_inference_weights(use_channels_last_encoder=False)`. Materialize target types such as FP16/FP32 directly in shared storage. After explicit Parameter binding, validate and retain the owner, avoiding a second weight copy caused by converting a full CPU model.

Static video VAE sharing may cover both encoder and decoder weights. Current block offload uses two `NativeModuleBlockSlot` instances for the decoder. Encoding activates the encoder and related projections as whole modules; decoding activates necessary non-block weights and executes decoder blocks in sequence. A completion event orders multiple tiles, and release restores original CPU sources. Shared loading requires `vae_encoder_conv_mode="torch"`; channels-last and FP8 Conv3D need corresponding shared manifests and cannot be substituted directly. Check `video_vae.py` for additional device, quantization, and compile restrictions before making changes.

Whole-module audio VAE offload remains a separate private path. Video VAE sharing does not establish sharing for every VAE.

### Runner and launch entry point

`load_model()`, `init_run()`, `_offload_transformer()`, and `run_main()` in `lightx2v/models/runners/minimax_h3/minimax_h3_runner.py` determine component-loading order and phase transitions. Every rank must enter shared component initialization in the same order. Do not create shared text/VAE modules only on rank 0.

The entry point is [scripts/minimax_h3/offload](../../../../scripts/minimax_h3/offload/), with `run_minimax_h3_block_shared_offload.sh` and `configs/minimax_h3/offload/minimax_h3_block_shared_offload.json`. Select `t2av/i2av/l2av/fl2av/ref2av` with `--task` in the script and supply the corresponding inputs. Defaults are `t2av`, `--model-variant fl2av`, host sharing, and eight GPUs with SP8. Select NUMA with `--shared_cpu_weight_scope`, and change the GPU list, process count, and JSON layout together.

Match the AdaLN cache to `--model-variant` and actual transformer weights, not only the task name. Variant `fl2av` uses `transformer/` and an fl2av cache; the dedicated `ref2av` variant uses `transformer_ref/` and a ref2av cache. Steps, flow shifts, and cache directory must also match. The cache tool has been restored to upstream's direct-file launch style:

```bash
python "${lightx2v_path}/tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py" \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3/offload/minimax_h3_block_shared_offload.json" \
  --model-variant fl2av
```

Set project/model paths and source `scripts/base/base.sh` before using this fragment. If using the cache tool's own shell script, edit its paths, GPU, configuration, and variant directly. It currently does not read `MINIMAX_H3_MODEL_PATH`, `MINIMAX_H3_CONFIG`, or `MINIMAX_H3_CACHE_TASK` overrides. Follow the tool's source rather than instructions for the old wrapper.

The CLI selects the task; configuration filenames alone do not establish the effective task. The ordinary `scripts/minimax_h3/run_minimax_h3_i2av.sh` also does not replace the shared launcher in the offload directory. Check the configuration each script actually references.

The H3 example uses separate component flags: `text_encoder_shared_cpu_weights`, `video_vae_shared_cpu_weights`, their block-offload settings, `text_encoder_release_block_offload_buffers`, and `dit_release_block_offload_buffers`. Add flags to new models only for component paths that are actually implemented.

Historical commits or external validation records are background only. Current offload READMEs are usage guides, not experimental reports. Base conclusions on runs with the current code, assets, hardware, and configuration. Do not reuse old memory measurements or output hashes as results for the current task.
