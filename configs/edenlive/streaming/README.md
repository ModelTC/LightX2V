# Wan2.2 5B AR streaming

`wan22_5b_ar_fp8_lighttae_sync.json` is the current 1280×704, 24 FPS preset:
FP8 DiT, compile, warmup, four denoising steps, and synchronous LightTAE chunk
decoding. `num_frames=null` follows the recording length.

The preset selects `sage_attn2_sm90_fused` for H100/H200 (SM90). It uses the same
SageAttention 2 attention and FP8 value-quantization kernels, fuses key centering
with INT8 quantization, and reuses the history's transposed value layout during
the four denoising steps and KV commit. Raw KV cache contents remain unchanged.
This backend requires head dimension 64 or 128. For another GPU architecture,
set the three attention types to a supported backend such as `sage_attn2`.

The other presets retain their configured attention backends. The resident AR
path also reuses rotated history keys and text K/V. Its FP Ring cache copies
logical fragments directly into fixed-capacity attention workspaces through
`k_cache(..., out=...)` and `v_cache(..., out=...)`. Storage addresses stay fixed
across chunks; each new chunk refreshes history and its relative RoPE, then the
denoising steps overwrite only the current part. CPU/KV offload and quantized KV
retain the existing cache materialization path. Request cleanup clears these
extra caches.

An earlier 16 FPS H200 run on 2026-10-16 produced all 477 frames in 40 chunks: 33.18 generation
FPS, 360.97 ms mean output interval, and 396.33 ms P95. The output MP4 is byte
identical to the pre-optimization output. Extra cache workspaces increased peak
allocated memory from 25.91 to 31.39 GiB.

Profile, timings, saved video, and reproduction commands:
[/data/nvme6/gushiqiao/edenlive_benchmarks/2026-10-16_profile_opt/README.md](/data/nvme6/gushiqiao/edenlive_benchmarks/2026-10-16_profile_opt/README.md).

A 24 FPS check of persistent workspaces produced all 717 frames in 60 chunks.
Before/after latents and MP4 bytes match exactly; mean output interval changed
from 360.96 to 348.47 ms in one paired run. See
[the workspace validation report](/data/nvme6/gushiqiao/edenlive_benchmarks/2026-10-16_persistent_kv_workspace/README.md).
