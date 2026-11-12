# H3 video PDMD LoRA transferred to H3-Image

Checked on 2026-11-11. This records the published artifacts and inference
implementations, not a new image-training result.

## Provenance

The [akhaliq model card](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step)
describes transferring the existing video PDMD adapter to H3-Image. It does
not claim to have trained a new image PDMD adapter. The source adapter is
[pdmd2026/pdmd_2NFE_lora](https://huggingface.co/pdmd2026/pdmd_2NFE_lora).

The two repositories' Hugging Face LFS metadata independently confirms that
their `lora_model_0.safetensors` files are identical:

| Field | Value |
| --- | --- |
| akhaliq revision | `ef1e3b9a237fc7a57ba4d07174ecca397d695d8a` |
| upstream revision | `3bc1bf125da77483a5d7819dae6b745b0404dadf` |
| file size, both repositories | 1,383,680,648 bytes |
| SHA256, both repositories | `3179e630c3fbd96b9ad7aec3a54fe23fa0ddc63e4afb64607f10cdb3d05c862a` |

Sources: [akhaliq file metadata](https://huggingface.co/api/models/akhaliq/minimax-h3-image-pdmd-2step?blobs=true)
and [upstream file metadata](https://huggingface.co/api/models/pdmd2026/pdmd_2NFE_lora?blobs=true).
Only model cards, code, metadata, and HTTP range reads of safetensors headers
were needed; the large weights were not downloaded for this audit.

The upstream card identifies the training base as the joint video/audio
MiniMax-H3 model. Its reported recipe is PDMD, two sampling segments,
critic:student updates 6:1, student/critic learning rates 5e-5/1e-5, and
checkpoint step 4000. PDMD projects the DMD update perpendicular to the
student–critic endpoint residual. These are the upstream authors' reported
training settings, not measurements of this repository's training run.

## What the adapter contains

The actual safetensors header agrees with the released
[sidecar metadata](https://huggingface.co/pdmd2026/pdmd_2NFE_lora/blob/main/lora_model_0.safetensors.json):

- 624 BF16 tensors, comprising 312 LoRA A/B pairs.
- Rank 128 and alpha 128, so the adapter multiplier is 1 before user strength.
- Six adapted matrices per block: Q, K, V, attention output, and two FFN
  projections. This covers 50 main blocks and two token-refiner blocks.
- No AdaLN or final output-head LoRA.

The native/ComfyUI conversion contains the same updates in a different
layout. Its fused QKV representation uses rank 384 and alpha 384 because
three rank-128 updates are concatenated with a block-diagonal B matrix.
The attention output and FFN projections remain rank/alpha 128. This is a
representation change, not a separately trained rank-384 adapter. Its header
also records the FFN gate/value permutation.

## Two different inference paths

The repository offers both of these; they should not be conflated.

| Path | Transformer output | Other components |
| --- | --- | --- |
| `render_grid.py`, Diffusers | Original single `final_layer.video_out` mapped to `proj_out`; no fused PDD head file loaded | Pruna P-H3-Image's Qwen3-VL-4B plus connector and video VAE |
| ComfyUI merged checkpoint | Six-head bank retained in a stacked output matrix; PDD sigma metadata retained | Comfy-Org H3-Image repack and its image-serving path |

The [Diffusers script](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step/blob/main/render_grid.py)
downloads the upstream PDMD adapter, converts only the native transformer
shards, merges the adapter, and samples with shift 12. Its scheduler is
called with three grid points for two denoising evaluations. The card
explicitly separates this early 3D-transformer implementation from the
new native `MiniMaxH3ImageTransformer2DModel` integration.

The [ComfyUI merge script](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step/blob/main/merge_comfy.py)
merges 208 native-layout updates into Comfy-Org's checkpoint and preserves
its metadata. Header inspection confirms output weights of shape
`[576,5376]` and a bias of shape `[576]`, together with the original six-step
PDD sigma grid.

ComfyUI's [`FinalLayer` and `_pdd_head`](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ldm/minimax/model.py)
use both the current and next sampling sigma. A step spanning several
original PDD intervals takes a duration-weighted mean of the corresponding
heads. Its compact checkpoint representation stores the first full head
and subsequent offsets. This differs from selecting one head solely from
the current sigma.

## Implications for our code

PDMD and the released H3-Image PDD head bank are separate mechanisms. The
video adapter is trained with PDMD. The image checkpoint's per-step heads
and fixed six-step schedule are the official PDD inference path. Transferring
the adapter does not make these two methods identical.

The original image checkpoint still includes a single FP32 output layer:
`final_layer.video_out.weight [96,5376]` and bias `[96]`, located through
`transformer/model.safetensors.index.json` in shard
`model-00001-of-00012.safetensors`. Separately,
`pdd_fused_heads.safetensors` holds weights `[6,96,5376]` and biases `[6,96]`.
The current Diffusers image checkpoint replaces the original output with
these six heads. See the [official model card](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image).

Our earlier PDMD experiment selected the head at the start of the enclosing
released interval for off-grid score queries. That experimental extension
reproduces neither of the two community paths above. It should remain
explicitly identified in existing experiments; changing it silently would
change the experiment and checkpoint semantics.

For a new single-head experiment, restore the original output projection
from the native shard and keep the image model's official text conditioning,
packing, RoPE and VAE. This permits testing the transferred adapter without
an interval-dependent output selection. It still does **not** establish
that the released, already distilled image backbone is an officially
validated continuous-score teacher: MiniMax does not state that guarantee
for the preserved original head. Training a new PDD or PDMD model from it
requires validation of the resulting trajectories and images.

The published samples support trying this transfer. They do not establish
that any video LoRA transfers equally well, or that the Pruna-based sample
quality and timings will carry over unchanged to our native image stack.
