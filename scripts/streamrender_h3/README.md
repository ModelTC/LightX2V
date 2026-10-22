# StreamRender-H3 semantic-reference video

`streamrender_h3` is a LightX2V-native runner for the released two-step checkpoint. It loads the Qwen3-VL conditioner, TAE encoder, H3 DiT, prefix KV cache, and causal video decoder from LightX2V modules and model weights. Inference does not import or launch the Sol-Engine source checkout. The request takes a 24 fps semantic video, a first-frame reference image, and a prompt; it publishes an RGB MP4. The model also computes audio latents, but this video route does not publish audio.

Download the [checkpoint bundle](https://huggingface.co/Efficient-Large-Model/StreamRender-H3) and its assets into one model directory. Set `model_path` in [refv2v_2step.json](../../configs/streamrender_h3/refv2v_2step.json), then run:

```bash
bash scripts/streamrender_h3/run_refv2v_2step.sh
```

Set `SEMANTIC_VIDEO`, `REFERENCE_IMAGE`, `OUTPUT_PATH`, `CONFIG_PATH`, `NUM_FRAMES`, `CUDA_VISIBLE_DEVICES`, or `PYTHON_BIN` to change the example inputs and runtime. The checkpoint bundle's `assets.json` supplies the default image and prompt.

The released float schedule uses video timesteps `1, 6/7` and audio timesteps `1, 3/5`. Both DiT steps run. Each timestep keeps a separate prefix KV bank, and later windows read the stored physical prefix directly from the shared LightX2V KV pool. On a 47-frame real example, the native BF16 output versus the official worker's published frames measured 28.93 dB mean RGB PSNR and 0.8972 SSIM. This establishes visual alignment, not bitwise parity.

[refv2v_2step_aot_graph.json](../../configs/streamrender_h3/refv2v_2step_aot_graph.json) enables AOT compilation, CUDA Graph, and warmup for repeated same-shape requests. Block weight offload and KV offload can run together. In a 47-frame test with both offloads, the first request captured 200 graphs and five subsequent requests added no captures; all requests produced identical MP4 bytes. The warm pipeline took 12.54 seconds, excluding model loading and Qwen conditioning. The resident eager config is preferable when there is only one request.

[refv2v_2step_fp8_f16_accum_sm120.json](../../configs/streamrender_h3/refv2v_2step_fp8_f16_accum_sm120.json) uses converted FP8 weights and fused QKV. Its SM120 accumulation kernel still needs real SM120 validation. Convert the checkpoint without materializing a merged state dict:

```bash
python -m tools.convert.quant.h3_fp8_f16_accum \
  --source /path/to/StreamRender-H3 \
  --output /path/to/StreamRender-H3-fp8-f16-accum
```
