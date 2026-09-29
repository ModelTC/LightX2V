"""Runtime features for MiniMax-H3 multi-GPU runs on gfx1201 (``radeon_gfx1201_h3``), installed by ``install``.

Each feature is on by default and disabled with RADEON_CORESW_<NAME>=0:
- FASTLOAD: radeon_weight_load (zero-copy offload views, pinned staging ring, parallel VAE prepare) and, with
  sequence parallelism, IDLE_EVICT (free idle VAE / text-encoder memory during the DiT steps).
- VAE_FUSED: fused HIP video VAE decoder blocks (radeon_vae_fused).
- TE_RANK0: run the text encoder on rank 0 only and broadcast its outputs.
- AUDIO_OVERLAP: with parallel VAE decode, the last rank decodes the audio while rank 0 decodes video rows.
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
    if _enabled("AUDIO_OVERLAP"):
        _patch_audio_overlap(runner)
    if _enabled("FP32_CONV_ROCBLAS"):
        _patch_fp32_conv_rocblas(runner)
    if _enabled("WARMUP"):
        _warm_libraries()
