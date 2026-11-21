# LTX Audio and Video Generation

[English](README.md) | [简体中文](README_zh.md)

## Prepare

Complete the [environment setup](../../docs/EN/source/getting_started/quickstart.md), then choose an entry for the model version and task:

| Version or task | Entry |
| --- | --- |
| LTX-2 text to audio-video | [run_ltx2_t2av.sh](run_ltx2_t2av.sh) |
| LTX-2 image to audio-video | [run_ltx2_i2av.sh](run_ltx2_i2av.sh) |
| LTX-2.3 text / image to audio-video | [run_ltx2_3_t2av.sh](ltx2_3/run_ltx2_3_t2av.sh) / [run_ltx2_3_i2av.sh](ltx2_3/run_ltx2_3_i2av.sh) |
| LTX-2.3 audio, keyframes, and motion conditioning | [ltx2_3](ltx2_3) |
| LTX-2.5 distilled, eight GPUs | [run_ltx2_5_t2av_distilled.sh](ltx2_5/run_ltx2_5_t2av_distilled.sh) / [run_ltx2_5_i2av_distilled.sh](ltx2_5/run_ltx2_5_i2av_distilled.sh) |

Download the corresponding [LTX-2](https://huggingface.co/Lightricks/LTX-2), [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3), or [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) weights. In the selected Shell script, set `lightx2v_path`, `model_path`, and `CUDA_VISIBLE_DEVICES`, then edit the prompt and input paths.

- **LTX-2 / LTX-2.3:** set `dit_original_ckpt` in the selected JSON to the matching checkpoint. Point `gemma_original_ckpt` to a complete [Gemma 3](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) HF directory containing its weights, tokenizer, and processor. When this setting is omitted, the loader searches under `model_path` for Gemma assets.
- **LTX-2.5:** set the component checkpoint paths in the selected JSON, including the Transformer, text encoder, video/audio VAE, duration head, and upsampler. The launchers use [ltx2_5_distilled_8gpu.json](../../configs/ltx2/ltx2_5_distilled_8gpu.json).

Prepare the model architecture config before startup. For LTX-2, the official `transformer/config.json` is supported. For integrated LTX-2.3 checkpoints without a standalone config, export the matching Transformer and audio/video startup settings from that checkpoint metadata once into a `config.json` under `model_path`. Keep each model directory tied to its checkpoint version; do not reuse an LTX-2 config for LTX-2.3. The existing directory loader reads these settings, with explicit deployment settings applied last. Gemma remains a separate asset selected by `gemma_original_ckpt`.

For LTX-2 / LTX-2.3 Gemma 3, set `gemma_cpu_offload=true` and `gemma_offload_granularity="block"` to keep the vocabulary on CPU, transfer only selected token embeddings, and stage decoder layers, final norm, projection, and video/audio connectors separately. This setting is independent of the DiT `offload_granularity`. Transfers are synchronous and retain the original CPU weight storage; this mode does not pin or prefetch weights. The default `"model"` retains whole-module offload. Calls sharing one encoder must be serial; concurrent shared-encoder calls through the direct API are unsupported. LTX-2.5 rejects Gemma block offload before loading.

## Run

For the basic LTX-2 example, update `dit_original_ckpt` in [ltx2.json](../../configs/ltx2/ltx2.json), then run from the repository root:

```bash
bash scripts/ltx2/run_ltx2_t2av.sh
```

The basic text/image scripts save results to `save_results/output_lightx2v_ltx2_t2av.mp4` or `save_results/output_lightx2v_ltx2_i2av.mp4`. Change `--save_result_path` to choose another location. Set resolution, frame count, and sampling parameters in the selected JSON under [configs/ltx2](../../configs/ltx2).

LTX2 and LTX2.5 services can start without `--task`. Each native request selects `t2av` or `i2av`, sharing the loaded model. CLI inference still requires `--task`. LTX2's `ltx2_s2v` and `v2av` paths use dedicated deployment configs and keep their task selected at startup.

For multi-GPU LTX-2 inference, choose the launchers ending in `_cfg_parallel.sh`, `_ulysses.sh`, or `_tp.sh`, and keep GPU count consistent with `--nproc_per_node` and the JSON parallel settings.

## Python API

| LTX-2 weights | Text to audio-video | Image to audio-video |
| --- | --- | --- |
| Base | [ltx2_t2av.py](inference/ltx2_t2av.py) | [ltx2_i2av.py](inference/ltx2_i2av.py) |
| Distilled FP8 | [ltx2_t2av_distilled_fp8.py](inference/ltx2_t2av_distilled_fp8.py) | [ltx2_i2av_distilled_fp8.py](inference/ltx2_i2av_distilled_fp8.py) |

Edit the model directory, checkpoint, input, and output paths in the selected example, then run:

```bash
python scripts/ltx2/inference/ltx2_t2av.py
```

See the [Python API guide](../../docs/EN/source/getting_started/python_api.md) for shared usage.
