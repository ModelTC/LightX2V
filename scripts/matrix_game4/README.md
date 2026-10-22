# Matrix-Game-4

Native Wan2.2 5B Control autoregressive rerendering from low-poly RGB video and an independent sink image. The game engine handles actions, HUD and audio outside the model. LightX2V runs its own weights, operators, scheduler and shared Wan AR lifecycle.

See the [Chinese deployment and cache guide](README_zh.md). The local Castle release is at `/data/nvme3/gushiqiao/models/Matrix-Game-4.0`.

```bash
VIDEO_PATH=/path/to/castle-recording.mp4 bash scripts/matrix_game4/run_matrix_game4.sh
```

Set `VIDEO_PATH` to a local Castle recording when launching, or export it once before trying the other profiles below. The public interface is `--model_cls matrix_game4 --model-variant castle --task v2v`, with `--video_path`, `--image_path` and a scene prompt. Output preserves the source frame rate and frame count.

[quality.json](../../configs/matrix_game4/quality.json) uses the original Wan VAE, two denoising steps, 480×864, FA3, and enables compile, CUDA Graph and warmup together. FA2 and Sage2 use the shared SF physical-ring cache backends. MG4 refreshes only the fixed-relative sink from its raw key; recent keys retain absolute RoPE positions. Reading KV does not concatenate or reorder the window.

[realtime.json](../../configs/matrix_game4/realtime.json) uses native TAEHV for both input encoding and output decoding, with a single `[1000]` denoising step. DiT compile, CUDA Graph and warmup remain enabled; codec execution is eager.

The full-video validation scene is Castle. Racing requires its own checkpoint and [quality_racing.json](../../configs/matrix_game4/quality_racing.json), because its reference is the first output frame and its sink retains absolute RoPE positions. Single-GPU resident caches remain the default without CFG. KV offload, weight block streaming, SP and TP are available through the profiles below; MG4 shares KV across denoising steps.

## Initial native-integration validation

H200, PyTorch 2.11.0+cu130, Castle BF16, 480×864, 241 frames at 24 FPS, diffusion seed 42. Both LightX2V presets enable compile, CUDA Graph and warmup; timings average three requests after the first full request. DiT time covers denoising and clean KV commit and excludes reference prefill.

| Backend / preset | Reference DiT | LightX2V DiT | DiT time change | Video PSNR |
| --- | ---: | ---: | ---: | ---: |
| FA3 quality | 8.982 s | 7.865 s | −12.43% | 19.75 dB |
| FA3 realtime | 6.003 s | 2.648 s | −55.88% | 18.13 dB |
| Sage2 quality | 8.168 s | 8.944 s | +9.50% | 16.70 dB |
| Sage2 realtime | 5.914 s | 3.386 s | −42.75% | 18.18 dB |

FA3 references use the official native FA3 path. Sage2 references use the same official model with dense Sage2 self/cross attention on H200; they are not the official RTX5090 FP8 preset. Sage2 quality is slower in this measured shape and asynchronous Wan VAE setup. FA3 remains the default.

Both FA3 presets produce exactly matching conditioning and first 20 latent chunks. Physical ring order changes attention reduction rounding after wrapping; later explosions, smoke and lighting differ. Full videos are not bit-identical. The `torch_opaque` RMSNorm backend preserves the native mean-square reduction under compile, improving FA3 quality PSNR from 18.30 to 19.75 dB. Existing RMSNorm registrations and SF/Eden defaults are unchanged.

Native public-request wall times are 14.06 s for FA3 quality and 7.36 s for realtime, including source decoding, control/sink encoding, T5, DiT and output decoding. Official push timings start after input preprocessing and text/sink encoding, so their wall times cannot be compared as equal-scope speedups. Stage CUDA events overlap with asynchronous VAE work and must not be summed.

Four requests per preset produced identical latent hashes, with zero new graph captures after warmup; 135 integration/regression tests passed. H200 is the only hardware validated with full videos. FA2 has cache/kernel regression coverage, not a full MG4 video validation. The measured resident FA3 pipeline peaks at 33.69 GB allocated for quality and 26.83 GB for realtime; 24 GB cards are not validated. T5/VAE symlinks require their targets when moving the local model bundle.


Cache ownership and backend selection live in Transformer Infer. MG4 reuses the SF rolling pool and FA/Sage attention backends; `rope.py` only retains its raw sink and position policy. Use `sage_attn2` for Sage; architecture-specific Sage backend names are not accepted by MG4.


## Offload and distributed execution

`realtime_offload.json` enables both pinned CPU KV storage and weight block streaming while retaining resident input VAE and text encoders so the conditioning matches the resident profile. Only committed KV tokens are prefetched until the window fills; current slot writes and refreshed sink keys are persisted before staging is reused. Reference prefill, growing/full windows and clean commits use per-layer CUDA Graph variants over the shared weight buffers; CPU/GPU transfers stay outside the graph.

`realtime_sp2.json` stores every valid token with head-sharded KV. RoPE and sink positions remain global, and sequence padding is excluded from storage. `realtime_tp2.json` reuses Wan weight partitioning: self KV shards heads across TP and SP; cross KV shards only across TP. MG4 Torch RMSNorm preserves the cast before multiplying its learned weight. Reference prefill additionally preserves native TP reduction and post-reduction rounding inside compiled blocks; denoising retains its prior compiled arithmetic.

All profiles retain compile, CUDA Graph and warmup. Resident SP/TP uses the shared block-stack graph; combining either mode with weight/KV staging uses per-block graphs. Offload trades transfer time for memory, and two GPUs need not outperform one GPU for this 5B shape. Distributed and offload video validation uses H200; consumer GPU coverage remains pending.

```bash
CONFIG_PATH=configs/matrix_game4/realtime_offload.json bash scripts/matrix_game4/run_matrix_game4.sh
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 CONFIG_PATH=configs/matrix_game4/realtime_sp2.json bash scripts/matrix_game4/run_matrix_game4.sh
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 CONFIG_PATH=configs/matrix_game4/realtime_tp2.json bash scripts/matrix_game4/run_matrix_game4.sh
```


## Historical offload and parallel validation

The earlier offload integration used the same Castle BF16, 480×864, 241-frame case and compared against its preceding native LightX2V implementation. Resident, SP2 and combined KV/weight offload produce byte-identical full latent and decoded video tensors. Its TP2-versus-single-GPU comparison had video PSNR 17.62 dB and later smoke/fire differences. This historical cross-mode metric does not describe current same-mode TP2 precision after the prefill RMS fix below.

A paired H200 run with identical CPU affinity, one excluded request and three formal requests measured resident FA3 DiT at 2.632 → 2.622 s and public-request wall time at 7.832 → 7.643 s. These small changes establish retained throughput. MG4 launches VAE work from the caller thread onto a separate CUDA stream; avoiding a competing Python worker preserves dispatch speed for its short chunks. Other model defaults are unchanged.

Single-formal-request smoke measurements were 3.314 s DiT for SP2 and 29.484 s for combined offload. Combined offload reduces peak allocated memory from 26.83 to 14.57 GB. These modes add distributed execution and lower-memory operation; neither outperforms resident single-GPU execution for this 5B case. Resident/SP2 reuse four captures, and offload reuses sixty per-layer captures, without formal-request recaptures. FP64 RoPE, dynamic sink updates and odd-token SP padding have regression coverage.


## Current same-mode validation (2026-10-21)

MG4 is absent from upstream LightX2V main, so this regression uses the frozen preceding MG4 implementation. TP2 is compared with that implementation in **TP2**, rather than against single-GPU output. The phase-specific RMS hook preserves the original native reference-prefill rounding inside compiled blocks and leaves denoising arithmetic unchanged. The shared graph variant includes the reference-prefill flag for both whole-stack and staged execution.

Castle realtime, FA3, 480×864, 241 frames, TP2 resident: complete latent SHA is identical for two requests on both ranks; full float-video SHA is identical for request 0 on both ranks. Each request adds **0 captures and 184 replays** with compile, Graph and warmup enabled. The one formal rank-0 measurement per version is DiT 3.581 → 2.021 s and pipeline 8.422 → 6.511 s; these are regression measurements, not stable throughput estimates.

SP2 with weight block + KV offload also matches the preceding same-mode complete latent/video output; its steady timing is essentially unchanged, within 0.5%, while cold warmup is longer. The final TP2 + weight block/KV offload check also has exact complete latent and float-video SHA for two requests on both ranks against the same correct TP baseline. Each request adds 0 captures/compilations and 3690 per-block replays; peak allocated memory is about 6.80 GB per rank, with CPU weight sources and pinned matrix buffers verified. This is a functional/memory check, not a speed comparison with resident TP. Input VAE weights remain resident. FA2/Sage2 full-video TP equivalence, alternative TP RMS backends, mixed SP×TP and RTX 40/50 hardware are outside this validation.
