# Depth Anything 3 MONO-LARGE

Use the ComfyUI-format `depth_anything_3_mono_large.safetensors` checkpoint with
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

The first argument may point directly to the safetensors file or to its parent
directory. The script uses the active `python` and the repository's
[`da3mono_large.json`](../../configs/depth_anything3/da3mono_large.json).

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
