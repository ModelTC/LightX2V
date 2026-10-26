# Wan AR codec configuration

Self-Forcing, Lingbot Fast v2, Realtime-Video, EdenLive AR and Matrix-Game-v4 share codec loading and compile configuration.

| Setting | Scope |
| --- | --- |
| `use_compile` | DiT compilation |
| `vae_use_compile` | Compilation of the selected VAE/TAE components, independent of `use_compile`; omitted or `false` keeps codec execution eager |
| `async_vae_decode` | Top-level asynchronous chunk decode switch; omitted or `false` keeps decode synchronous |
| `use_tae`, `tae_path` | Select the compatible Wan2.1/Wan2.2 tiny output codec and its weights |
| `tae_decoder_path` | Optional compatible decoder-only replacement; used by MG4 to retain its original input/sink encoder |

VAE/TAE compilation uses [WanRunner._configure_vae_compile](../../lightx2v/models/runners/wan/wan_runner.py). TAE weight loading and optional decoder replacement use the same file's `load_tiny_vae_decoder()`. [WanStreamingVAERunnerMixin](../../lightx2v/models/runners/wan/streaming_vae.py) shares causal decoder selection and chunk decoding. Models keep their required input encoders and normalization; sharing the loader does not make Wan2.1 and Wan2.2 checkpoints interchangeable. Krea keeps its original input encoder for pixel feedback, and MG4 keeps its Castle input/sink encoder when only the output decoder changes.

EdenLive and MG4 expose `--task v2v`, with an independent `image_path` and conditioning `video_path`. Eden accepts only `v2v`. Lingbot Fast remains `i2v`; Self-Forcing and Krea remain `t2v`.

Saved benchmark configurations describe the measured codec, dtype and compile state. Changing a current JSON does not update historical measurements: in particular, Lingbot's earlier FP32 full-VAE results remain FP32 results and must not be relabeled as new default-dtype or parallel-encoder results. Native Wan2.2 and TAE encoders do not gain spatial multi-GPU execution merely from a shared loader or DiT SP/TP; actual codec parallelism must be checked separately.

## Shared autoregressive file output

Wan AR and H3 AR runners inherit the request output lifecycle in
[AutoregressiveRunnerMixin](../../lightx2v/models/runners/autoregressive_runner.py).
A local `.mp4` `save_result_path` automatically selects incremental output on
rank 0. No extra JSON switch is needed. Models send decoded chunks through
`record_video_chunk()`; AV models provide their final waveform with
`set_stream_audio()`.

[StreamingVideoWriter](../../lightx2v/utils/streaming_video.py) converts one
chunk to CPU RGB uint8, feeds a bounded queue, and writes fragmented H.264 to a
temporary file while inference continues. One-second GOPs/fragments and packet
flushes ensure payload reaches disk before the request ends. Successful video
output is renamed to the requested path; audio is added using video stream-copy
and audio-only trim/padding, preserving every video frame. Encoding failures
propagate to the request and its worker is drained during cleanup.

With `return_result_tensor=false`, connected chunk loops avoid collecting a
whole GPU video. With `return_result_tensor=true`, the original tensor return
contract remains available alongside file output. Existing network/RTMP output
keeps its model-specific transport. Codec layout/range conversion remains
explicit: Wan uses BCTHW in [-1,1], H3 uses BCTHW in [0,1] or uint8, and TaoMate
applies its exact native-to-publication frame mapping before submitting frames.
