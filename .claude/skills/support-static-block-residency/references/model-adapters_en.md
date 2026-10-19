# Wan and Qwen Image Integration Examples

English | [简体中文](model-adapters.md)

These examples locate the current implementation. Distinguish validator coverage, launcher examples, and actual validation coverage; a launcher does not establish hardware acceptance. See [implementation patterns](implementation-patterns_en.md) for the common algorithm and lifetime rules.

## Wan2.1

### Call chain and estimation inputs

- [model.py](../../../../lightx2v/models/networks/wan/model.py): `WanModel` initialization and the call to `validate_wan_residency_config()`.
- [transformer_weights.py](../../../../lightx2v/models/networks/wan/weights/transformer_weights.py): `WanTransformerAttentionBlock` and device slots.
- [offload/transformer_infer.py](../../../../lightx2v/models/networks/wan/infer/offload/transformer_infer.py): `WanOffloadTransformerInfer`, `infer_with_blocks_offload()`, and `release_resident_blocks()`.
- [wan_runner.py](../../../../lightx2v/models/runners/wan/wan_runner.py): cleanup in `run_segment()` and `_run_warmup()`.
- [Estimator](../../../../tools/estimate_offload_residency.py): `build_wan_case()`, loading prefix `blocks.0.`.
- [infer/utils.py](../../../../lightx2v/models/networks/wan/infer/utils.py): `image_latent_size()`, matching production I2V sizing rules.

Current validation accepts `model_cls="wan2.1"` with `t2v` or `i2v`; the example launchers use I2V. Other Wan variants do not inherit static-residency support merely by sharing a base class.

I2V estimation reads the actual aspect ratio of `--image-path` and combines it with config `size`, VAE stride, and patch size to construct the latent grid. Treating the JSON size as the final image dimensions can mismatch production inference. T2V uses config dimensions. The temporal dimension follows normalized `num_frames`.

The case uses production Wan blocks, scheduler state, and RoPE construction, with synthetic hidden state, modulation inputs, and context. I2V context includes image-conditioning tokens when the image encoder is enabled. The tool explicitly rejects unimplemented Sol-Attn Morton input reordering; do not silently change the operator to bypass this limit.

Quantized checkpoint lookup uses `dit_quantized_ckpt`; the unquantized path uses `dit_original_ckpt`, falling back to `model_path`. Preserve production loading dtype and sensitive-layer conversion rules, especially FP8 scale rounding. FP8 checkpoint storage and computation dtype are separate settings.

### Launchers and configs

Entry points are in [Wan offload_residency](../../../../scripts/wan/inference/offload_residency/). See the [English guide](../../../../scripts/wan/inference/offload_residency/README.md) / [Chinese guide](../../../../scripts/wan/inference/offload_residency/README_CN.md).

| Platform | Weights and base config | Default generation conditions |
|---|---|---|
| CUDA | `Wan2.1-I2V-14B-720P-Lightx2v` FP8; [baseline.json](../../../../configs/wan/residency/baseline.json) | 81 frames, 40 steps, config `size=[480,832]`; the input image determines the actual I2V aspect ratio |
| Ascend NPU | Original `Wan2.1-I2V-14B-720P` BF16; [baseline_npu.json](../../../../configs/wan/residency/baseline_npu.json) | The same frame, step, and size settings, with platform-appropriate NPU operators |

CUDA config paths `dit_quantized_ckpt`, `t5_quantized_ckpt`, and `clip_quantized_ckpt` are independent of the launcher's `model_path`; check each. NPU launchers contain a model-path placeholder that must be filled with complete model, encoder, and VAE assets.

Use the same image for estimator `--image-path` and inference `--image_path`. Match positive/negative prompts and seed between inference launchers. Regenerate the residency config from the base config after changing frame count, dimensions, or steps.

Outputs are under `save_results/wan_residency/`: `resident.json`, `resident.profile.json`, `resident.mp4`, and `baseline.mp4`. NPU counterparts are `resident_npu.json`, `resident_npu.profile.json`, `resident_npu.mp4`, and `baseline_npu.mp4`.

## Qwen Image

### Call chain and estimation inputs

- [model.py](../../../../lightx2v/models/networks/qwen_image/model.py): `QwenImageTransformerModel` initialization and config validation.
- [transformer_weights.py](../../../../lightx2v/models/networks/qwen_image/weights/transformer_weights.py): `QwenImageTransformerAttentionBlock` and device slots.
- [offload/transformer_infer.py](../../../../lightx2v/models/networks/qwen_image/infer/offload/transformer_infer.py): `validate_qwen_residency_config()`, `QwenImageOffloadTransformerInfer`, and cleanup entry points.
- [qwen_image_runner.py](../../../../lightx2v/models/runners/qwen_image/qwen_image_runner.py): `run()` and `_run_warmup()` release resident weights before VAE decoding.
- [Estimator](../../../../tools/estimate_offload_residency.py): `build_qwen_case()`, loading prefix `transformer_blocks.0.`.
- [infer/utils.py](../../../../lightx2v/models/networks/qwen_image/infer/utils.py): `get_qwen_image_size()`, shared by estimation and the runner.

Current scope is `model_cls="qwen_image"`, `task="t2i"`, excluding layered generation, editing tasks, Magi compile, `zero_cond_t`, and `use_layer3d_rope`. Common static-residency checks also apply. Ordinary sequential CFG passes differ from the rejected multi-device CFG parallel mode.

The image grid follows effective dimensions, VAE scale, and patch rules. `Qwen2Tokenizer` computes lengths using the production prompt template and subtracts its prefix. Update estimation and inference together when changing positive or negative prompts. With CFG enabled, the current case selects the shorter text branch because it usually provides less compute budget; this representative-input choice does not prove a timing bound for arbitrary operators.

The case uses production `QwenEmbedRope` and operator-specific frequency/position preparation, with synthetic image, text, and modulation inputs. `run_block()` receives image/text and returns text/image; preserve that state ordering when connecting `run_blocks()`. Check the text branch as well as the final output so correctness in one branch cannot hide an error in the other.

Checkpoint lookup prefers the applicable quantized/original override, then `transformer_model_path`, and finally `model_path/transformer`. New quantized formats still need production-loading and operator-contract checks; BF16 example results do not establish coverage for all formats.

### Launchers and configs

Entry points are in [Qwen offload_residency](../../../../scripts/qwen_image/inference/offload_residency/). See the [English guide](../../../../scripts/qwen_image/inference/offload_residency/README.md) / [Chinese guide](../../../../scripts/qwen_image/inference/offload_residency/README_CN.md).

| Platform | Weights and base config | Default generation conditions |
|---|---|---|
| CUDA | `Qwen-Image-2512` BF16; [baseline.json](../../../../configs/qwen_image/residency/baseline.json) | Coffee shop prompt, 16:9, 50 steps, seed 42 |
| Ascend NPU | The same BF16 model; [baseline_npu.json](../../../../configs/qwen_image/residency/baseline_npu.json) | The same generation conditions, with platform-appropriate NPU operators |

The model directory contains `transformer`, `text_encoder`, `tokenizer`, `scheduler`, and `vae`. Text-encoder/VAE offload settings act independently of DiT residency. Do not attribute the entire pipeline's memory change to `r`.

Outputs are under `save_results/qwen_residency/`: `resident.json`, `resident.profile.json`, `resident.png`, and `baseline.png`. NPU counterparts are `resident_npu.json`, `resident_npu.profile.json`, `resident_npu.png`, and `baseline_npu.png`.

## Platform usage conventions

Both directories contain `estimate.sh`, `run.sh`, and `run_baseline.sh`, with `_npu.sh` counterparts. CUDA launchers use `PLATFORM=cuda` and `CUDA_VISIBLE_DEVICES`; Ascend uses `PLATFORM=ascend_npu` and `ASCEND_RT_VISIBLE_DEVICES`, while runtime code selects the device API through `AI_DEVICE="npu"`. Do not confuse platform and device names.

Launchers set `lightx2v_path` and `model_path`, source [base.sh](../../../../scripts/base/base.sh), and use `"${lightx2v_path}/..."` for config, input, output, and `mkdir` paths. They keep the caller's working directory and omit `set -e`. In `base.sh`, `DTYPE` defaults to BF16 and `SENSITIVE_LAYER_DTYPE` to None, following the main dtype. Existing external values are retained; record effective values for comparisons.

Launchers inherit `PROFILING_DEBUG_LEVEL` from `base.sh`, currently `2`. Adjust the shared setting there when needed and keep it identical for baseline and resident runs. Exporting a value before invoking a launcher does not override the explicit assignment in `base.sh`. See [validation and debugging](validation-and-debugging_en.md) for timing labels and coverage.

Existing CUDA validation and NPU static checks cover their respective conditions only. New NPU validation must actually execute estimation and inference entry points. Without hardware, retain an explicit untested status rather than hiding limitations by changing precision or inputs.
