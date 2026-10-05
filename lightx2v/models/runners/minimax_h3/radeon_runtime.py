"""Runtime features for MiniMax-H3 multi-GPU runs on gfx1201 (``radeon_gfx1201_h3``), installed by ``install``.

Each feature is on by default and disabled with RADEON_CORESW_<NAME>=0:
- FASTLOAD: radeon_weight_load (zero-copy offload views, pinned staging ring, parallel VAE prepare) and, with
  sequence parallelism, IDLE_EVICT (free idle VAE / text-encoder memory during the DiT steps).
- VAE_FUSED: fused HIP video VAE decoder blocks (radeon_vae_fused).
- TE_RANK0: run the text encoder on rank 0 only and broadcast its outputs.
- INPUT_OVERLAP: (ref2av, with TE_RANK0) the last rank VAE-encodes the references while rank 0 runs the text encoder,
  then broadcasts the latents (same tiles and ops as the parallel encode, so the latents are bitwise identical).
- AUDIO_OVERLAP: with parallel VAE decode, the last rank decodes the audio while rank 0 decodes video rows.
- STREAM_SAVE: rank 0 encodes the .mp4 while the video VAE decodes: tiles are dealt round-robin over the ranks
  (radeon_vae_stream), each finished clip's frames go to a PyAV encoder thread, and the audio is written after the
  last frame as before. Frames are bitwise identical to the whole-video decode; the encoder settings are unchanged.
- FP32_CONV_ROCBLAS: FP32 convolutions (VAE im2col GEMMs) on rocBLAS; hipBLASLt picks a per-process FP32 kernel,
  which makes outputs differ from run to run.
- WARMUP: initialize the BLAS libraries and the Triton cache key on a side thread.
"""

import os
import sys

_installed = []


def _enabled(name):
    return os.environ.get("RADEON_CORESW_" + name, "1") != "0"


def _patch_fp32_conv_rocblas(module):
    import threading

    import torch
    import torch.nn.functional as F

    default = torch.backends.cuda.preferred_blas_library()
    lock = threading.Lock()
    depth = [0]

    def wrap(native):
        def conv(input, *args, **kwargs):
            if input.dtype != torch.float32 or not input.is_cuda:
                return native(input, *args, **kwargs)
            with lock:
                if depth[0] == 0:
                    torch.backends.cuda.preferred_blas_library("cublas")
                depth[0] += 1
            try:
                return native(input, *args, **kwargs)
            finally:
                with lock:
                    depth[0] -= 1
                    if depth[0] == 0:
                        torch.backends.cuda.preferred_blas_library(default)

        return conv

    for name in ("conv1d", "conv2d", "conv3d"):
        setattr(F, name, wrap(getattr(F, name)))


def _patch_te_rank0(module):
    cls = module.MiniMaxH3Runner
    native = cls.run_text_encoder

    cls._radeon_text_encoder = native

    def run_text_encoder(self, input_info, keyframes=None, references=None):
        import torch
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_world_size() == 1:
            return native(self, input_info, keyframes=keyframes, references=references)
        out = native(self, input_info, keyframes=keyframes, references=references) if dist.get_rank() == 0 else None
        meta = [None if out is None else {k: (tuple(v.shape), v.dtype) for k, v in out.items()}]
        dist.broadcast_object_list(meta, src=0)
        device = torch.device("cuda", torch.cuda.current_device())
        if out is None:
            out = {k: torch.empty(shape, dtype=dtype, device=device) for k, (shape, dtype) in meta[0].items()}
        for key in sorted(out):
            dist.broadcast(out[key], src=0)
        return out

    cls.run_text_encoder = run_text_encoder


class _Pending:
    """Rank-0 stand-in for the audio decoded on another rank; replays [i]/.float()/.cpu() after the NCCL receive."""

    def __init__(self, resolve, ops=()):
        self._resolve, self._ops = resolve, ops

    def __getitem__(self, key):
        return _Pending(self._resolve, self._ops + (("getitem", key),))

    def float(self):
        return _Pending(self._resolve, self._ops + (("float", None),))

    def cpu(self):
        return _Pending(self._resolve, self._ops + (("cpu", None),))

    def materialize(self):
        value = self._resolve()
        for op, arg in self._ops:
            value = value[arg] if op == "getitem" else getattr(value, op)()
        return value


class _LazyAudio:
    def __init__(self, pending, sampling_rate):
        self._pending, self.sampling_rate, self._value = pending, sampling_rate, None

    @property
    def waveform(self):
        if self._value is None:
            self._value = self._pending.materialize()
        return self._value


_DTYPES = ("float32", "bfloat16", "float16", "float64")


def _broadcast_tensors(tensors, src):
    """Broadcast a list of tensors (shapes/dtypes from src); returns them on the current device."""
    import torch
    import torch.distributed as dist

    meta = [None if tensors is None else [(tuple(t.shape), str(t.dtype).split(".")[-1]) for t in tensors]]
    dist.broadcast_object_list(meta, src=src)
    device = torch.device("cuda", torch.cuda.current_device())
    out = []
    for index, (shape, dtype) in enumerate(meta[0]):
        tensor = tensors[index].to(device) if tensors is not None else torch.empty(shape, dtype=getattr(torch, dtype), device=device)
        dist.broadcast(tensor, src=src)
        out.append(tensor)
    return out


def _patch_input_overlap(module):
    cls = module.MiniMaxH3Runner
    native = cls._run_input_encoder_local_h3

    def _run_input_encoder_local_h3(self):
        import torch.distributed as dist

        from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2

        if (
            self.input_info.task != "ref2av"
            or not dist.is_initialized()
            or dist.get_world_size() == 1
            or getattr(cls, "_radeon_text_encoder", None) is None
            or self.loaded_transformer_partition != "transformer_ref"
        ):
            return native(self)
        with ProfilingContext4DebugL2("Run Input Encoder"):
            self.clear_conditioning_state()
            self._resolve_request_geometry()
            self.prepared_references = references = self._prepare_references()
            rank, src = dist.get_rank(), dist.get_world_size() - 1
            text_encoder_output, encoded = None, None
            if rank == 0:
                text_encoder_output = cls._radeon_text_encoder(self, self.input_info, references=references)
            elif rank == src:
                vae, parallel = self.video_vae, self.video_vae.encode_parallel
                vae.encode_parallel = False
                try:
                    with ProfilingContext4DebugL1("Run VAE Encoder"):
                        encoded = self._encode_references(references)
                finally:
                    vae.encode_parallel = parallel
            keys = sorted(text_encoder_output) if rank == 0 else None
            keys = [keys]
            dist.broadcast_object_list(keys, src=0)
            values = _broadcast_tensors(None if rank != 0 else [text_encoder_output[k] for k in keys[0]], 0)
            text_encoder_output = dict(zip(keys[0], values))
            counts = [None if encoded is None else (len(encoded[0]), len(encoded[1]))]
            dist.broadcast_object_list(counts, src=src)
            latents = _broadcast_tensors(None if encoded is None else [*encoded[0], *encoded[1]], src)
            video_latents = [t.cpu() for t in latents[: counts[0][0]]]
            audio_latents = [t.cpu() for t in latents[counts[0][0] :]]
            if rank == src:
                video_latents, audio_latents = encoded
            else:
                # the attributes _encode_references records on each reference
                videos, audios = iter(video_latents), iter(audio_latents)
                for reference in references:
                    if reference.kind != "audio":
                        latent = next(videos)
                        reference.num_latent_frames = latent.shape[2]
                        reference.latent_height, reference.latent_width = latent.shape[3:]
                    if reference.has_audio:
                        reference.num_audio_latents = next(audios).shape[-1]
            self.condition_video_latents, self.condition_audio_latents = video_latents, audio_latents
            tags = text_encoder_output["text_token_tags"]
            if tags.ndim != 1:
                raise ValueError("MiniMax-H3 conditioner token tags must be one-dimensional")
            self.maybe_empty_cache()
            return {"text_encoder_output": text_encoder_output}

    cls._run_input_encoder_local_h3 = _run_input_encoder_local_h3
    cls._run_input_encoder_local_t2av = cls._run_input_encoder_local_i2av = native


def _patch_audio_overlap(module):
    cls = module.MiniMaxH3Runner
    native_decode, native_post, real_audio = cls.run_vae_decoder, cls.process_images_after_vae_decoder, module.Audio

    def audio_factory(*args, **kwargs):
        waveform = kwargs.get("waveform", args[0] if args else None)
        if isinstance(waveform, _Pending):
            return _LazyAudio(waveform, kwargs.get("sampling_rate", args[1] if len(args) > 1 else None))
        return real_audio(*args, **kwargs)

    module.Audio = audio_factory

    def run_vae_decoder(self, video_rows, audio_rows):
        import torch
        import torch.distributed as dist

        if not (self.video_vae.decode_parallel and dist.is_initialized() and dist.get_world_size() > 1):
            return native_decode(self, video_rows, audio_rows)
        rank, src = dist.get_rank(), dist.get_world_size() - 1
        if rank == 0:
            cache = {}

            def resolve():
                if "audio" not in cache:
                    device = torch.device("cuda", torch.cuda.current_device())
                    meta = torch.empty(6, dtype=torch.int64, device=device)
                    dist.recv(meta, src)
                    ndim, code, *dims = meta.tolist()
                    cache["audio"] = torch.empty(dims[:ndim], dtype=getattr(torch, _DTYPES[code]), device=device)
                    dist.recv(cache["audio"], src)
                return cache["audio"]

            decode, self.audio_vae.decode = self.audio_vae.decode, lambda latents: _Pending(resolve)
            try:
                return native_decode(self, video_rows, audio_rows)
            finally:
                self.audio_vae.decode = decode
        out = native_decode(self, video_rows, audio_rows)
        if rank == src:
            from lightx2v.utils.profiler import ProfilingContext4DebugL1

            latents = module.unpack_audio_tokens(audio_rows[self.scheduler.num_condition_audio_rows :], self.scheduler.num_audio_latents)
            with ProfilingContext4DebugL1("Run Audio VAE Decoder (overlapped)"):
                audio = self.audio_vae.decode(latents).contiguous()
            meta = torch.zeros(6, dtype=torch.int64, device=audio.device)
            meta[0], meta[1] = audio.ndim, _DTYPES.index(str(audio.dtype).split(".")[-1])
            meta[2 : 2 + audio.ndim] = torch.tensor(audio.shape)
            dist.send(meta, 0)
            dist.send(audio, 0)
            torch.cuda.synchronize()
        return out

    def process_images_after_vae_decoder(self):
        if isinstance(getattr(self, "gen_audio", None), _Pending) and self.input_info.return_result_tensor:
            self.gen_audio = self.gen_audio.materialize()
        return native_post(self)

    cls.run_vae_decoder = run_vae_decoder
    cls.process_images_after_vae_decoder = process_images_after_vae_decoder


def _patch_stream_save(module):
    import queue
    import threading

    cls = module.MiniMaxH3Runner
    native_decode, native_post = cls.run_vae_decoder, cls.process_images_after_vae_decoder

    def active(self):
        path = self.input_info.save_result_path
        return bool(path) and not self.input_info.return_result_tensor and os.path.splitext(path)[1].lower() == ".mp4" and self.video_vae.use_tiling

    class _AudioSlot:
        """Audio for encode_video, filled by the main thread once the last frame is decoded (read after the video)."""

        def __init__(self, sampling_rate):
            self.sampling_rate, self._ready, self._waveform = sampling_rate, threading.Event(), None

        def set(self, waveform):
            self._waveform = waveform
            self._ready.set()

        @property
        def waveform(self):
            self._ready.wait()
            return self._waveform

    def encoder(self, frames, audio, failure):
        def chunks():
            while (item := frames.get()) is not None:
                yield item

        try:
            module.encode_video(
                video=chunks(),
                fps=int(self.config.get("fps", 24)),
                audio=audio,
                output_path=self.input_info.save_result_path,
                video_chunks_number=1,
                video_codec_options=self.config.get("video_codec_options"),
            )
        except BaseException as error:  # noqa: BLE001  # re-raised on the main thread
            failure.append(error)
            while frames.get() is not None:
                pass

    def run_vae_decoder(self, video_rows, audio_rows):
        import torch
        import torch.distributed as dist

        from lightx2v.models.video_encoders.hf.minimax_h3.radeon_vae_stream import stream_decode
        from lightx2v.utils.profiler import ProfilingContext4DebugL1

        if not active(self):
            return native_decode(self, video_rows, audio_rows)
        parallel = self.video_vae.decode_parallel and dist.is_initialized() and dist.get_world_size() > 1
        rank, world = (dist.get_rank(), dist.get_world_size()) if parallel else (0, 1)
        video_rows = video_rows[self.scheduler.num_condition_video_rows :]
        audio_rows = audio_rows[self.scheduler.num_condition_audio_rows :]
        video_latents = module.unpatchify_video_tokens(
            video_rows,
            self.scheduler.num_latent_frames,
            self.scheduler.latent_height,
            self.scheduler.latent_width,
            channels=int(self.config.get("in_channels", 24)),
            patch_size=tuple(self.config.get("patch_size", (1, 2, 2))),
        )
        audio_latents = module.unpack_audio_tokens(audio_rows, self.scheduler.num_audio_latents)
        self.set_vae_decode_tile_shape()
        if rank == 0:
            path = self.input_info.save_result_path
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._stream_frames, self._stream_failure = queue.Queue(maxsize=4), []
            self._stream_audio = _AudioSlot(self.audio_vae.sampling_rate)
            self._stream_thread = threading.Thread(target=encoder, args=(self, self._stream_frames, self._stream_audio, self._stream_failure), name="radeon_stream_save", daemon=True)
            self._stream_thread.start()
        try:
            with ProfilingContext4DebugL1("Run Video VAE Decoder"):
                for chunk in stream_decode(self.video_vae, video_latents, rank, world):
                    self._stream_frames.put(self._video_to_uint8_frames(chunk))
                    del chunk
        finally:
            if rank == 0:
                self._stream_frames.put(None)
        source = world - 1
        if rank == source:
            with ProfilingContext4DebugL1("Run Audio VAE Decoder"):
                audio = self.audio_vae.decode(audio_latents).contiguous()
            if world > 1:
                meta = torch.zeros(6, dtype=torch.int64, device=audio.device)
                meta[0], meta[1] = audio.ndim, _DTYPES.index(str(audio.dtype).split(".")[-1])
                meta[2 : 2 + audio.ndim] = torch.tensor(audio.shape)
                dist.send(meta, 0)
                dist.send(audio, 0)
                torch.cuda.synchronize()
        if rank == 0:
            if world > 1:
                device = torch.device("cuda", torch.cuda.current_device())
                meta = torch.empty(6, dtype=torch.int64, device=device)
                dist.recv(meta, source)
                ndim, code, *dims = meta.tolist()
                audio = torch.empty(dims[:ndim], dtype=getattr(torch, _DTYPES[code]), device=device)
                dist.recv(audio, source)
            self._stream_audio.set(audio[0].float().cpu())
        return None, None

    def process_images_after_vae_decoder(self):
        from lightx2v.utils.profiler import ProfilingContext4DebugL2

        thread = getattr(self, "_stream_thread", None)
        if thread is None:
            if active(self):
                return {"video": None, "audio": None}
            return native_post(self)
        self._stream_thread = None
        with ProfilingContext4DebugL2("Save Audio-Video Output"):
            thread.join()
        if self._stream_failure:
            raise self._stream_failure[0]
        print(f"[radeon_runtime] streamed output saved to {self.input_info.save_result_path}", flush=True)
        return {"video": None, "audio": None}

    cls.run_vae_decoder = run_vae_decoder
    cls.process_images_after_vae_decoder = process_images_after_vae_decoder


def _warm_libraries():
    import threading

    import torch

    device = torch.cuda.current_device()

    def run():
        try:
            torch.cuda.set_device(device)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for dtype in (torch.float32, torch.bfloat16):
                    a = torch.randn(64, 192, dtype=dtype, device="cuda")
                    w = torch.randn(256, 192, dtype=dtype, device="cuda")
                    b = torch.randn(256, dtype=dtype, device="cuda")
                    torch.mm(a, w.t()), torch.mm(a, w.t().contiguous()), torch.mm(a.t().contiguous().t(), w.t())
                    torch.addmm(b, a, w.t()), torch.addmm(b, a, w.t().contiguous()), torch.nn.functional.linear(a, w, b)
            stream.synchronize()
            from triton.runtime.cache import triton_key

            triton_key()
        except Exception as error:  # noqa: BLE001  # warm-up only
            print(f"[radeon_runtime] warm-up skipped: {error!r}", file=sys.stderr, flush=True)

    threading.Thread(target=run, name="radeon_warmup", daemon=True).start()


def _patch_vae(module):
    from lightx2v.models.video_encoders.hf.minimax_h3 import radeon_vae_fused

    cls = module.MiniMaxH3VideoVAE
    native = cls.from_pretrained.__func__

    def from_pretrained(klass, *args, **kwargs):
        vae = native(klass, *args, **kwargs)
        print(f"[radeon_runtime] fused VAE decoder blocks installed={radeon_vae_fused.install(vae)}", flush=True)
        return vae

    cls.from_pretrained = classmethod(from_pretrained)


def install(config):
    """Install the enabled features once per process; call after init_parallel and before build_runner."""
    if _installed:
        return
    _installed.append(True)
    import importlib

    from lightx2v.models.runners.minimax_h3 import radeon_weight_load as weight_load

    runner = importlib.import_module("lightx2v.models.runners.minimax_h3.minimax_h3_runner")
    video_vae = importlib.import_module("lightx2v.models.video_encoders.hf.minimax_h3.video_vae")
    if _enabled("FASTLOAD"):
        weight_load.install_runner(runner)
        for name, function in (
            ("lightx2v.common.ops.utils", weight_load.install_ops_utils),
            ("lightx2v.common.ops.tensor.tensor", weight_load.install_tensor),
            ("lightx2v.common.ops.embedding.embedding_weight", weight_load.install_embedding),
            ("lightx2v.models.networks.minimax_h3.infer.offload.transformer_infer", weight_load.install_offload_infer),
            ("lightx2v.common.offload.event_manager", weight_load.install_event_manager),
        ):
            function(importlib.import_module(name))
        weight_load.install_video_vae(video_vae)
        if config.get("seq_parallel", False) and _enabled("IDLE_EVICT"):
            weight_load.install_residency(runner)
    if _enabled("VAE_FUSED"):
        _patch_vae(video_vae)
    if _enabled("TE_RANK0"):
        _patch_te_rank0(runner)
        if _enabled("INPUT_OVERLAP"):
            _patch_input_overlap(runner)
    if _enabled("AUDIO_OVERLAP"):
        _patch_audio_overlap(runner)
    if _enabled("STREAM_SAVE"):
        _patch_stream_save(runner)
    if _enabled("FP32_CONV_ROCBLAS"):
        _patch_fp32_conv_rocblas(runner)
    if _enabled("WARMUP"):
        _warm_libraries()
