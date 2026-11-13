# TBDub Student: native Wan2.2 5B v2v

The runner inherits `Wan22DenseRunner`, extends Wan's native DiT with audio
conditioning, and uses native HuBERT and a small Wan2.2 VAE adapter. It does not
import the upstream TBDub/DiffSynth pipeline.

## Input contract

Supply a **25-FPS cropped-face video** and its driving audio. The crop is resized
to 512×512; audio is decoded to mono 16 kHz. Speaker selection, face tracking,
cropping and paste-back remain the caller's responsibility (e.g. video_dubbing).
The output duration follows the audio; `--num_frames` optionally caps it. Short
reference videos are extended using the source pipeline's ping-pong rule.
The model uses the released null prompt embedding, not a user text prompt.

## Weights and dependencies

From the repository root:

```bash
python -m pip install -e '.[tbdub]'
export MODEL_PATH=/path/to/tbdub
hf download TaoLiveAIGC/TBDub tbdub_student.safetensors null_prompt_emb.pt --local-dir "$MODEL_PATH"
hf download KlingTeam/X-Dub Wan2.2_VAE.safetensors --local-dir "$MODEL_PATH"
hf download facebook/hubert-large-ll60k config.json preprocessor_config.json pytorch_model.bin \
  --local-dir "$MODEL_PATH/hubert-large-ll60k"
```

FFmpeg must be on PATH. Explicit `dit_original_ckpt`, `vae_path`,
`audio_encoder_path` and `prompt_embedding_path` in the config override the
default paths under `MODEL_PATH`.

## Inference

```bash
export CUDA_VISIBLE_DEVICES=0
export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16
export VIDEO_PATH=/path/to/cropped_25fps.mp4
export AUDIO_PATH=/path/to/translated.wav
export OUTPUT_PATH=/path/to/dubbed_crop.mp4
bash scripts/wan22/run_tbdub_v2v.sh
```

The same runner is available through the existing LightX2V server:

```bash
python -m lightx2v.server --model_cls tbdub --task v2v \
  --model_path "$MODEL_PATH" --config_json configs/wan22/tbdub.json \
  --host 127.0.0.1 --port 8000
```

Request fields: `video_path`, `audio_path`, `seed`, `num_frames`, `size`,
`save_result_path`, `return_result_tensor`. Media paths must be accessible to the
server. No separate TBDub service or custom request protocol is introduced.

## Implementation conventions

- Student: two Euler steps, shift 1, no CFG. Teacher inference is not included.
- Each 77-frame window has five history frames and advances by 72 frames.
  Two generated latent frames condition the next window. The latent sequence
  is stitched before VAE decoding; the initial five-frame preroll is discarded.
- DiT/VAE use BF16; HuBERT uses FP32. The supplied Torch norm/modulation and
  complex RoPE settings retain the source rounding conventions.
- Model/block offload and Wan sequence parallelism are wired through native
  lifecycle hooks. Sequence parallelism requires an evenly divisible token
  count; set `parallel.seq_p_size` and launch matching torchrun ranks.
- The VAE adapter preserves TBDub's encoder-head cache, normalization and scale
  conventions without modifying the shared Wan VAE.
- Phase offload, tensor parallelism, CFG, feature caching and tiled/streaming VAE
  are not supported by this port.

This migration has only been checked statically in this checkout; numerical
parity, performance and parallel execution have not been revalidated here.
