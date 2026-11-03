# Depth Anything 3 MONO-LARGE

Use the ComfyUI-format `depth_anything_3_mono_large.safetensors` checkpoint with
LightX2V's native single-image depth runner. Set up the LightX2V Python
environment first, then run from any directory:

```bash
bash /path/to/LightX2V/scripts/depth_anything3/run_da3mono_large.sh \
  /path/to/geometry_estimation \
  /path/to/input.png \
  /path/to/depth.png
```

The first argument may point directly to the safetensors file or to its parent
directory. The script uses the active `python` and the repository's
[`da3mono_large.json`](../../configs/depth_anything3/da3mono_large.json).

The output is an 8-bit grayscale PNG at the input image size. Near regions are
bright, far regions are dark, and detected sky is black. It is a relative-depth
preview, not a metric-depth map. This first release handles one RGB image per
request and only the `mono_large` variant.
