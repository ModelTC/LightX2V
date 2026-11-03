# Depth Anything 3 MONO-LARGE

Use either the official DA3MONO-LARGE `model.safetensors` checkpoint from
[`depth-anything/DA3MONO-LARGE`](https://huggingface.co/depth-anything/DA3MONO-LARGE),
or the ComfyUI-format `depth_anything_3_mono_large.safetensors` checkpoint with
LightX2V's native depth runner. Set up the LightX2V Python
environment first, then run from any directory:

```bash
bash /path/to/LightX2V/scripts/depth_anything3/run_da3mono_large.sh \
  /path/to/geometry_estimation \
  /path/to/input.png \
  /path/to/depth.png
```

For a video, use an MP4 output path:

```bash
bash /path/to/LightX2V/scripts/depth_anything3/run_da3mono_large.sh \
  /path/to/geometry_estimation \
  /path/to/input.mp4 \
  /path/to/depth.mp4
```

The first argument may point directly to either safetensors file or to its parent
directory. If both filenames are present, pass the desired file explicitly.
The script uses the active `python` and the repository's
[`da3mono_large.json`](../../configs/depth_anything3/da3mono_large.json).
Set `process_res` (default `504`) and `process_res_method` in that config to
control the internal inference size. The four methods match the official DA3
input processor:

| Method | First resize | Align width and height to multiples of 14 |
| --- | --- | --- |
| `upper_bound_resize` (default) | Longest side to `process_res` | Resize to nearest multiple |
| `lower_bound_resize` | Shortest side to `process_res` | Resize to nearest multiple |
| `upper_bound_crop` | Longest side to `process_res` | Center crop down to multiple |
| `lower_bound_crop` | Shortest side to `process_res` | Center crop down to multiple |

For example, a 640×360 input at `process_res=504` becomes 504×280 internally
with `upper_bound_resize`, or 896×504 with `lower_bound_resize`. The preview
is still resized back to the original input size. `lower_bound` may use much
more GPU memory for wide or tall images; CUDA Graph capture follows the
processed frame size.
Set `attn_type` in that config to select a LightX2V dense-attention backend.
`torch_sdpa` is the default; `flash_attn2`, `flash_attn3` and `sage_attn2`
also work when their optional packages are installed. Flash and Sage can
produce slightly different depth values from the default backend.
`backbone_dtype` defaults to `auto`: the ViT backbone uses BF16 on CUDA and
FP32 elsewhere, while the depth head stays in FP32. Set it to `fp32` to
retain the original checkpoint precision. BF16 can alter preview pixels,
especially at the sky mask boundary; explicit `bf16` requires CUDA.
`norm_type` can be `torch` (default) or `Triton`; the latter uses LightX2V's
2D LayerNorm kernel on flattened token rows and requires Triton. It is an
optional platform choice, not a faster default on H100.
For long, fixed-size videos, set `use_cuda_graph` to `true` to capture the
model once and replay it for subsequent frames. Capture adds startup time and
holds extra GPU memory; single-image requests always use eager execution.
Only one input shape is captured per runner. Frames of another size use eager
execution, and `runner.close()` releases the graph with the model.
The supplied checkpoint still uses unquantized linear weights. FP8
`mm_type` selection needs a DA3-specific quantized checkpoint and weight/scale
loader; it is not enabled by changing `attn_type`.

The output is an 8-bit grayscale PNG at the input image size. Near regions are
bright, far regions are dark, and detected sky is black. It is a relative-depth
preview, not a metric-depth map. MP4 output is an H.264 grayscale preview.
Video frames are decoded, inferred and encoded
incrementally with a bounded writer queue; the source audio is muxed after
encoding. Video output uses the source's average frame rate (variable-rate
inputs become constant-rate), with per-frame contrast normalization. H.264
requires even dimensions, so an odd input width or height gains one repeated
edge pixel in the output. Only the `mono_large` variant is supported.
Video encoding and audio muxing require `ffmpeg` on `PATH`.

The optional fourth and fifth script arguments save numeric outputs alongside
the preview:

```bash
bash /path/to/LightX2V/scripts/depth_anything3/run_da3mono_large.sh \
  /path/to/geometry_estimation /path/to/input.png /path/to/depth.png \
  /path/to/depth.npy /path/to/sky_mask.png
```

For videos, provide directories instead of files. Frames are written as
`000000.npy`, `000001.npy`, ... and `000000.png`, `000001.png`, ... while the
MP4 is encoded. A directory must be new or empty so a shorter rerun cannot
leave stale frames. Pass an empty fourth argument to save only the mask.
The NPY holds non-normalized float32 relative depth at the internal inference
resolution, after the mono sky-depth correction; it is not metric depth.
The mask PNG uses 255 for sky (`sky >= 0.5`) and 0 otherwise, also at the
internal inference resolution. The official mono sky-depth correction uses a
different threshold (`sky >= 0.3`): sky pixels receive the 99th percentile
of non-sky depth when both regions have more than ten pixels. This runner
computes the percentile over all non-sky pixels for deterministic results;
the official implementation randomly samples 100,000 pixels for larger images.
