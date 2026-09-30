# Wan and Qwen Integration Examples

English | [简体中文](model-examples.md)

These examples describe reuse points and differences in the current implementation. Use the source version, checkpoint, and effective configuration to determine precision, model size, block counts, and operator selection.

## Wan

Trace block definitions from `register_offload_group` in [TransformerWeights](../../../../lightx2v/models/networks/wan/weights/transformer_weights.py). Checkpoint prefixes are `blocks.{i}.`, and the two existing `offload_block_cuda_buffers` serve as device slots.

Inference uses the existing [offload TransformerInfer](../../../../lightx2v/models/networks/wan/infer/offload/transformer_infer.py); common code handles loading, layout, and transfers. The former Wan-specific `models/networks/wan/weights/block_layout.py` has been replaced by the common loader. Do not restore it based on stale IDE tabs or historical paths.

- [CUDA continuous configuration](../../../../configs/wan/layout/continuous.json): the current example uses LightX2V FP8-vLLM weights and CUDA attention. T5 and CLIP also have their own quantization settings.
- [NPU continuous configuration](../../../../configs/wan/layout/continuous_npu.json): original floating-point weights, explicit NPU operators, and component offload.
- The [T5 weight implementation](../../../../lightx2v/models/input_encoders/hf/wan/t5/model.py) accepts `rms_norm_type`, selected through the runner's `t5_rms_norm_type`, while preserving the original CUDA default. See [WanRunner](../../../../lightx2v/models/runners/wan/wan_runner.py); other callers must pass consistent arguments too.
- The [CUDA entry point](../../../../scripts/wan/offload_layout/run_contiguous.sh) and [NPU entry point](../../../../scripts/wan/offload_layout/run_contiguous_npu.sh) each have a corresponding baseline script. Check paths, environment settings, and actual output names before running them.

A contiguous DiT layout does not mean T5, CLIP, and VAE share that layout. The current GPU and NPU examples differ in quantization and component residency policies, so they cannot directly measure layout benefits across platforms.

## Qwen Image

[QwenImageTransformerWeights](../../../../lightx2v/models/networks/qwen_image/weights/transformer_weights.py) registers `transformer_blocks.{i}.`. Each block contains four phases: image attention, text attention, joint attention, and FFN. They share one layout; do not treat each phase as a complete block.

Inference uses the existing [QwenImageOffloadTransformerInfer](../../../../lightx2v/models/networks/qwen_image/infer/offload/transformer_infer.py) with the same common group declaration, storage contracts, and transfer logic. There is no Qwen-specific layout loader.

- [CUDA continuous configuration](../../../../configs/qwen_image/layout/continuous.json): the current example uses original Qwen-Image-2512 BF16 weights and `flash_attn3`. Read weights/inference code for other defaults.
- [NPU continuous configuration](../../../../configs/qwen_image/layout/continuous_npu.json): BF16, `npu_flash_attn`, and PyTorch real RoPE, norm, and modulation.
- [TransformerInfer](../../../../lightx2v/models/networks/qwen_image/infer/transformer_infer.py) uses `modulate_type` to decide whether to import Qwen Triton kernels, preserving the default CUDA kernels.
- The [text encoder](../../../../lightx2v/models/input_encoders/hf/qwen25/qwen25_vlforconditionalgeneration.py) and [VAE](../../../../lightx2v/models/video_encoders/hf/qwen_image/vae.py) are each offloaded as whole components. Contiguous blocks apply only to the DiT.
- Both [CUDA](../../../../scripts/qwen_image/offload_layout/run_contiguous.sh) and [NPU](../../../../scripts/qwen_image/offload_layout/run_contiguous_npu.sh) entry points and their baseline counterparts exist. NPU launchers accept `MODEL_PATH`, locate the repository automatically, and write outputs under it. Do not assume all older scripts support the same environment overrides.

The weight directory should contain the original model's `transformer/`, `text_encoder/`, `tokenizer/`, `vae/`, and related files. A single quantized DiT file cannot replace the full pipeline. The existing distilled `fp8-sgl` entry point does not establish that its operators implement the storage contracts needed here.

## Deciding where to integrate a new model

| Finding | Where to make the change |
|---|---|
| Native block offload and operator contracts already exist | Register the model group, add entry points, and validate |
| Multiple block structures | Keep compatible blocks/slots in one group; heterogeneous components need independent weight containers, managers, and corresponding scheduling |
| Operators have extra scales, biases, or device scalars | Operator storage descriptions and any required binding |
| The target platform lacks the required memory capabilities | Platform backend, with CUDA isolation checks |
| The text encoder or VAE still has CUDA-only dependencies | That component's configuration or platform operators; DiT layout cannot fix it |
| No native block offload, or an incompatible native nn.Module representation | Analyze gaps in block weights, slots, and scheduling before deciding the change scope |

Use the two implementations as references for mechanisms. Do not turn their layer indices, parameter counts, precision, shapes, or paths into common constants. See the [Wan guide](../../../../scripts/wan/offload_layout/README.md) and [Qwen guide](../../../../scripts/qwen_image/offload_layout/README.md) for existing commands, adapting them to the target machine before execution.
