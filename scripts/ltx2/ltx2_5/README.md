# LTX-2.5 distilled inference

This directory contains the native LightX2V profiles for the official
LTX-2.5 distilled two-stage audio/video pipeline. The implementation reuses
the LTX-2.3 DiT, audio VAE, latent upsampler, and sequence-parallel execution,
while adding the LTX-2.5 Gemma 4 text encoder, duration head, first-stage
ancestral sampler, keyframe position embedding, and diffusion video decoder.

Supported entry points:

- text-to-audio-video (`t2av`)
- first-frame image-to-audio-video (`i2av`)
- one GPU with block offload, or eight GPUs with Ulysses SP8
- the released BF16 distilled Transformer and default DiffVAE

The supplied profiles intentionally implement the distilled pipeline first.
The LTX-2.5 base/dev guider pipeline, quantized checkpoints, prompt enhancement,
generated keyframes, and DiffVAE tiling are not enabled by these profiles.

## Model layout

By default the scripts use:

```text
/data/nvme0/gushiqiao/models/LTX-2.5/
  diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
  vae/ltx-2.5-video-vae-bf16.safetensors
  vae/ltx-2.5-audio-vae-bf16.safetensors
  model_patches/ltx-2.5-duration-head-bf16.safetensors
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
```

Set `MODEL_PATH` to relocate the whole split pack. The supplied source-parity
baseline intentionally uses the locally available LTX-2.5 x2 upsampler 1.0 in
both LightX2V and `ltx_pipelines`; no LTX-2.3 download is required.

The header-only checker does not load the large tensors:

```bash
python tools/convert/check_ltx25_checkpoint.py \
  /data/nvme0/gushiqiao/models/LTX-2.5
```

## Text-to-audio-video

The launch scripts follow the same flat style as the Wan examples. Paths,
visible GPUs, prompt, seed and output are written near the top of each script;
edit those lines directly when needed. The supplied command is fixed to SP8:

```bash
scripts/ltx2/ltx2_5/run_ltx2_5_t2av_distilled.sh
```

The command does not pass `--num_frames`, so DurationHead predicts a length in
the configured 1--20 second range and snaps it to the causal VAE grid `8k+1`.
To force a length, add `--num_frames 121` to the command in the script.

## First-frame image-to-audio-video

Set `image_path` near the top of the script, then run:

```bash
scripts/ltx2/ltx2_5/run_ltx2_5_i2av_distilled.sh
```

Final two-stage height and width must be divisible by 64. An explicit frame
count must satisfy `num_frames = 8k+1`.

## Source-alignment profile

The JSON profiles preserve the released eight-step schedule exactly:

- stage 1 sigmas: `1, .99375, .9875, .98125, .975, .909375, .725, .421875, 0`
- stage 1 ancestral noise: `eta=1`, `s_noise=1`, generator seed `seed+10000`
- stage 2 sigmas: `.909375, .725, .421875, 0`
- image-conditioning CRF: 18
- Gemma 4 maximum sequence length: 1024
- DiT attention: PyTorch SDPA, matching the supplied upstream `.venv`
- spatial upsampler: local LTX-2.5 x2 BF16 1.0 on both sides
- output sink: float DiffVAE frames -> limited-range BT.709 YUV420p, CRF 19,
  preset `veryfast`

At the supplied 121-frame 1024x1536 shape, upstream `AUTO_TILING` resolves to
one full DiffVAE tile. LightX2V therefore keeps `use_tiling_vae=false`; this is
numerically equivalent for that profile. True multi-tile DiffVAE decode for
larger/longer custom requests is not yet exposed.

For numerical comparisons, use the same GPU count, attention backend,
spatial-upscaler checkpoint, dimensions, frame count, prompt, image, and seed
in LightX2V and the upstream `ltx_pipelines.distilled` command.

## License

The model weights are governed by the LTX-2.x Community License embedded in
the checkpoints and distributed with the upstream LTX-2 repository. Review
that license, including its commercial-use conditions, before redistribution
or commercial deployment. Native decoder code ported from upstream retains
its Apache-2.0 notices.
