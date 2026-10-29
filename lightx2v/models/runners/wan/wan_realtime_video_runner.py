"""Krea Realtime context reconstruction on the shared Wan causal runtime."""

import threading
from collections import deque
from pathlib import Path

import torch

from lightx2v.models.networks.wan.realtime_video_model import WanRealtimeVideoModel
from lightx2v.models.runners.wan.wan_sf_runner import WanSFRunner
from lightx2v.models.schedulers.wan.realtime_video.scheduler import WanRealtimeVideoScheduler
from lightx2v.models.video_encoders.hf.wan.realtime_video_vae import configure_realtime_video_vae
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


def select_clean_context(history, *, segment_index, context_frames=3, keep_first_frame=False, reencoded_first=None):
    """Match GenerationSession.get_clean_context_frames in the released code."""
    if history.shape[1] < context_frames:
        raise ValueError("Krea context refill requires a complete previous chunk")
    if keep_first_frame or (segment_index - 1) * 3 < context_frames:
        return torch.cat((history[:, :1], history[:, 1:][:, -context_frames + 1 :]), dim=1)
    if reencoded_first is None:
        raise ValueError("A rolling Krea context requires the oldest decoded frame to be reencoded")
    return torch.cat((reencoded_first.to(history), history[:, 1:][:, -context_frames + 1 :]), dim=1)


@RUNNER_REGISTER("wan2.1_realtime")
class WanRealtimeVideoRunner(WanSFRunner):
    _SUPPORTS_AUTOREGRESSIVE_WARMUP = True
    _WARMUP_TASKS = ("t2v",)

    def load_transformer(self):
        if self.config.get("lora_configs"):
            raise NotImplementedError("Krea Realtime's released checkpoint does not require a runtime LoRA")
        return WanRealtimeVideoModel(model_path=self.config["model_path"], config=self.config, device=self.init_device)

    def init_scheduler(self):
        self.scheduler = WanRealtimeVideoScheduler(self.config)

    def load_vae_encoder(self):
        path = self.config.get("vae_path", str(Path(self.config["model_path"]) / "Wan2.1_VAE.pth"))
        return configure_realtime_video_vae(
            WanVAE(
                vae_path=path,
                dtype=self._resolve_vae_dtype(self.config.get("vae_dtype"), torch.float16),
                device=AI_DEVICE,
                parallel=self.get_vae_parallel(),
                cpu_offload=self.config.get("vae_cpu_offload", False),
                dummy_model=self.config.get("dummy_model", False),
            )
        )

    def load_vae(self):
        if self.config.get("lightvae_path"):
            raise ValueError("Select Krea's alternative decoder with use_tae and tae_path")
        encoder = self.load_vae_encoder()
        if self.config.get("use_tae", False):
            # Pixel feedback always uses the released Wan encoder. The tiny
            # decoder supplies both displayed pixels and the feedback frames.
            self._configure_vae_compile(encoder, components=("encoder",))
            return encoder, self.load_tiny_vae_decoder()
        # The resident VAE has independent encode/decode caches and immutable
        # weights. Reuse it instead of retaining two full copies of the model.
        decoder = self.load_vae_encoder() if self.config.get("vae_cpu_offload", False) else encoder
        self._configure_vae_compile(encoder, components=("encoder",))
        return encoder, self._configure_vae_compile(decoder, components=("decoder",))

    def init_run(self):
        super().init_run()
        self._realtime_frames = deque(maxlen=1 + (self.scheduler.context_frames - 1) * 4)
        self._realtime_decode_ready = [threading.Event() for _ in range(self.video_segment_num)]
        self._realtime_decode_events = {}
        self._realtime_decode_errors = {}

    def decode_segment_latents(self, segment_idx, segment_latents):
        try:
            pixels = self.vae_decoder.cached_decode_withflag(segment_latents.to(self.vae_decoder.dtype), segment_idx == 0, segment_idx == self.video_segment_num - 1)
            pixels = pixels.float().clamp_(-1, 1)
            self._realtime_frames.extend(pixels.unbind(dim=2))
            if segment_idx == 0:
                # The official streaming session retains all nine feedback
                # frames but drops the first three from the returned video.
                pixels = pixels[:, :, 3:]
            ready = torch.cuda.Event()
            ready.record()
            self._realtime_decode_events[segment_idx] = ready
            return pixels
        except Exception as error:
            self._realtime_decode_errors[segment_idx] = error
            raise
        finally:
            self._realtime_decode_ready[segment_idx].set()

    def _reencode_oldest_frame(self, previous_segment):
        # Wait only for the producer to enqueue its ready event, then express
        # the required pixel-to-context dependency on the inference stream.
        self._realtime_decode_ready[previous_segment].wait()
        if previous_segment in self._realtime_decode_errors:
            raise RuntimeError("The preceding Krea decode failed") from self._realtime_decode_errors[previous_segment]
        torch.cuda.current_stream().wait_event(self._realtime_decode_events[previous_segment])
        frame = self._realtime_frames[0]
        video = frame.transpose(0, 1).unsqueeze(0).to(self.vae_encoder.dtype)
        return self.vae_encoder.encode(video)

    def run_segment(self, segment_idx=0):
        scheduler = self.scheduler
        infer = self.model.transformer_infer
        infer.reset_context_cache()
        if segment_idx:
            history = scheduler.latents[:, : segment_idx * scheduler.num_frame_per_chunk]
            keep_first = bool(self.config["ar_config"].get("keep_first_frame", False))
            first = self._reencode_oldest_frame(segment_idx - 1) if segment_idx > 1 and not keep_first else None
            context = select_clean_context(history, segment_index=segment_idx, context_frames=scheduler.context_frames, keep_first_frame=keep_first, reencoded_first=first)
            scheduler.prepare_context(context, segment_idx)
            try:
                self.model.infer(self.inputs)
            finally:
                scheduler.is_context_prefill = False
                scheduler.is_rerun = False
        return super().run_segment(segment_idx)

    def run_cache_commit(self, segment_idx):
        # Official Krea rebuilds the clean context before the next chunk.
        # A second SF-style post-chunk clean commit would be redundant.
        pass
