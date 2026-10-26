"""Native StreamRender-H3 RGB-to-RGB autoregressive request path."""

import json
from functools import partial
from itertools import islice
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger
from torch.nn.attention import SDPBackend, sdpa_kernel

from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl import MiniMaxH3Qwen3VLTextEncoder
from lightx2v.models.networks.minimax_h3.streamrender.model import StreamRenderH3Model
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.h3.autoregressive_runner import H3AutoregressiveRunnerMixin
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS, VIDEO_OUTPUT_FIELDS
from lightx2v.models.runners.streamrender_h3.output import StreamRenderVideoOutput
from lightx2v.models.runners.streamrender_h3.request import StreamRenderH3InputInfo
from lightx2v.models.schedulers.streamrender_h3 import StreamRenderH3Scheduler
from lightx2v.models.video_encoders.hf.streamrender_h3 import (
    StreamRenderH3CausalDecoder,
    StreamRenderH3TAEEncoder,
    prepare_picture,
)
from lightx2v.utils.async_vae import AsyncVAEChunkDecoder
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import seed_all
from lightx2v_platform.base.global_var import AI_DEVICE

_QWEN_SDPA_PRIORITY = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


@RUNNER_REGISTER("streamrender_h3")
class StreamRenderH3Runner(H3AutoregressiveRunnerMixin, BaseRunner):
    """Run the released two-step H3 student with native LightX2V modules."""

    input_info_cls_by_task = {"refv2v": StreamRenderH3InputInfo}
    supported_request_fields_by_task = {
        "refv2v": COMMON_REQUEST_FIELDS | VIDEO_OUTPUT_FIELDS | {"prompt", "image_path", "video_path"},
    }
    _WARMUP_SIZE = (256, 256)

    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.scheduler = None
        self.text_encoder = None

    def close(self):
        super().close()
        if self.text_encoder is not None:
            self.text_encoder.unload()

    def get_supported_tasks(self):
        return ("refv2v",)

    def init_modules(self):
        root = Path(self.config["model_path"])
        self.assets = json.loads((root / "assets.json").read_text())["assets"]
        self.scheduler = StreamRenderH3Scheduler(self.config)
        self.model = StreamRenderH3Model(str(root), self.config, torch.device("cpu"))
        self.model.set_scheduler(self.scheduler)
        vae_use_compile = bool(self.config.get("vae_use_compile", False))
        self.video_encoder = StreamRenderH3TAEEncoder(root / self.assets["tae"], device=AI_DEVICE, use_compile=vae_use_compile)
        output_rank = not dist.is_initialized() or dist.get_rank() == 0
        self.video_decoder = StreamRenderH3CausalDecoder.from_model_path(root, device=AI_DEVICE, linear_dtype=torch.bfloat16, use_compile=vae_use_compile) if output_rank else None
        text_config = dict(self.config)
        text_config.update(
            {
                "text_encoder_path": str(root / self.assets["qwen_model"]),
                "processor_path": str(root / self.assets["qwen_processor"]),
                "tokenizer_path": str(root / self.assets["qwen_processor"]),
                "text_encoder_load_on_init": False,
                "text_encoder_cpu_offload": True,
                "text_encoder_offload_granularity": "block",
                "text_encoder_release_block_offload_buffers": True,
                "text_encoder_host_pinned": False,
            }
        )
        self.text_encoder = MiniMaxH3Qwen3VLTextEncoder(text_config)

    @torch.no_grad()
    def warmup(self):
        if not self.config.get("warmup", False):
            return
        import av
        import numpy as np

        sizes = dict.fromkeys((self._WARMUP_SIZE, tuple(self.config["size"])))
        for height, width in sizes:
            self.input_info = self.create_input_info({"task": "refv2v", "seed": 0, "size": [height, width], "num_frames": self.config["num_frames"]})
            logger.info(f"Warmup refv2v: {height}x{width}x{self.input_info.num_frames}")
            try:
                seed_all(self.input_info.seed)
                picture, text, reference_size, generator = self._prepare_inference(self.input_info)
                encoder_stream = self.video_encoder.create_stream()
                output_rank = not dist.is_initialized() or dist.get_rank() == 0
                decoder_stream = self.video_decoder.create_stream() if output_rank else None
                reference_history = torch.empty((24, 0, reference_size[0] // 16, reference_size[1] // 16), device=AI_DEVICE)
                video_history = torch.empty((24, 0, height // 16, width // 16), device=AI_DEVICE)
                audio_history = torch.empty((2, 32, 0), device=AI_DEVICE)
                frame = av.VideoFrame.from_ndarray(np.zeros((height, width, 3), dtype=np.uint8), format="rgb24")
                for index, plan in enumerate(self.scheduler.window_plans):
                    encoded = self._encode_reference_window([frame] * plan.frame_count, encoder_stream, reference_size)
                    reference_history = torch.cat((reference_history, encoded), dim=1)
                    video, audio = self._infer_window(index, reference_history, video_history, audio_history, generator, picture, text)
                    video_history = torch.cat((video_history, video), dim=1)
                    audio_history = torch.cat((audio_history, audio), dim=2)
                    if decoder_stream is not None:
                        logger.info(
                            "StreamRender warmup window {}: allocated {:.3f} GiB, reserved {:.3f} GiB before decode", index, torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30
                        )
                        decoder_stream.push(video.permute(1, 0, 2, 3).unsqueeze(0))
                if decoder_stream is not None:
                    decoder_stream.flush()
                getattr(torch, AI_DEVICE).synchronize()
                del picture, text, generator, encoder_stream, decoder_stream, reference_history, video_history, audio_history, encoded, video, audio, frame
            finally:
                self.text_encoder.to_cpu()
                self.text_encoder.unload_vision_encoder()
                self.model.reset_autoregressive_state()
                if self.config.get("cpu_offload", False):
                    if self.model.block_offload:
                        if not self.model.prepost_resident:
                            self.model.pre_weight.to_cpu()
                            self.model.post_weight.to_cpu()
                    else:
                        self.model.to_cpu()
                self.scheduler.clear()
                self.input_info = None
        if dist.is_initialized():
            dist.barrier()
        self._maybe_freeze_gc()

    @staticmethod
    def _noise(shape, generator, device):
        return torch.empty(shape, device=device, dtype=torch.float32).normal_(generator=generator)

    def _prepare_window(self, index, reference_history, video_history, audio_history, generator, picture):
        plan = self.scheduler.window_plans[index]
        device = reference_history.device
        target_h, target_w = plan.target_latent_size
        video_new = self._noise((24, int(plan.video_noisy_mask.sum()), target_h, target_w), generator, device)
        audio_new = self._noise((2, 32, int(plan.audio_noisy_mask.sum())), generator, device)
        selected_video = torch.zeros((24, plan.video_indices.numel(), target_h, target_w), device=device)
        retained_video = ~plan.video_noisy_mask
        if retained_video.any():
            selected_video[:, retained_video] = video_history.index_select(1, plan.video_indices[retained_video].to(device))
        selected_video = 0.999 * selected_video + 0.001 * self._noise(selected_video.shape, generator, device)
        selected_video[:, plan.video_noisy_mask] = video_new
        selected_audio = torch.zeros((2, 32, plan.audio_indices.numel()), device=device)
        retained_audio = ~plan.audio_noisy_mask
        if retained_audio.any():
            selected_audio[:, :, retained_audio] = audio_history.index_select(2, plan.audio_indices[retained_audio].to(device))
        selected_audio[:, :, plan.audio_noisy_mask] = audio_new
        selected_reference = reference_history.index_select(1, plan.reference_indices.to(device))
        selected_reference = 0.999 * selected_reference + 0.001 * self._noise(selected_reference.shape, generator, device)
        self.scheduler.prepare_window(
            index,
            picture_latents=picture.latents,
            reference_latents=selected_reference,
            selected_video_latents=selected_video,
            selected_audio_latents=selected_audio,
        )

    def _infer_window(self, index, reference_history, video_history, audio_history, generator, picture, text):
        self._prepare_window(index, reference_history, video_history, audio_history, generator, picture)
        for step_index in range(self.scheduler.infer_steps):
            self.scheduler.step_pre(step_index)
            self.model.infer({"text_encoder_output": text})
            self.scheduler.step_post()
        return self.scheduler.finish_window()

    def _encode_reference_window(self, frames, stream, reference_size):
        import numpy as np

        arrays = np.stack([frame.to_ndarray(format="rgb24") for frame in frames])
        pixels = torch.from_numpy(arrays).to(AI_DEVICE).permute(3, 0, 1, 2).float().div_(255)
        pixels = F.interpolate(pixels.permute(1, 0, 2, 3), size=reference_size, mode="nearest").unsqueeze(0)
        pieces = []
        with torch.autocast("cuda", dtype=torch.float16):
            for frame in pixels.unbind(1):
                latent = stream.encode(frame.unsqueeze(1))
                if latent is not None:
                    pieces.append(latent)
        if not pieces:
            raise RuntimeError("StreamRender TAE did not publish reference latents")
        return torch.cat(pieces, dim=1)[0].permute(1, 0, 2, 3).float().contiguous()

    def _publish_decoded(self, output, video, _index):
        if video is not None:
            pixels = video.transpose(1, 2)
            self.record_video_chunk(pixels, layout="BCTHW", value_range=(0, 1))
            output.append(pixels)

    def _prepare_inference(self, input_info):
        root = Path(self.config["model_path"])
        height, width = map(int, input_info.size or self.config["size"])
        reference_size = tuple(self.config.get("reference_size", (480, 832)))
        picture_path = input_info.image_path or root / self.assets["reference_image"]
        prompt = input_info.prompt or self.assets["prompt"]
        picture = prepare_picture(
            picture_path,
            height=height,
            width=width,
            seed=int(input_info.seed),
            encoder=self.video_encoder,
            device=AI_DEVICE,
        )
        # The released FSDP Qwen path selects cuDNN SDPA on H200. Its BF16
        # accumulation determines the exact multimodal conditioning tensor.
        try:
            with sdpa_kernel(_QWEN_SDPA_PRIORITY, set_priority=True):
                text = self.text_encoder.infer(prompt, image_list=[picture.image])
        finally:
            self.text_encoder.to_cpu()
            self.text_encoder.unload_vision_encoder()
        frames_total = int(input_info.num_frames or self.config["num_frames"])
        self.scheduler.prepare_request(
            num_frames=frames_total,
            height=height,
            width=width,
            reference_height=reference_size[0],
            reference_width=reference_size[1],
            text_token_tags=text["text_token_tags"],
            picture_latent_size=tuple(picture.latents.shape[-2:]),
        )
        if self.model.block_offload:
            if self.model.prepost_resident:
                self.model.pre_weight.to_cuda()
                self.model.post_weight.to_cuda()
        else:
            self.model.to_cuda()
        self.model.device = torch.device(AI_DEVICE)
        generator = torch.Generator(device=AI_DEVICE).manual_seed(int(input_info.seed))
        return picture, text, reference_size, generator

    @torch.no_grad()
    def run_pipeline(self, input_info):
        import av

        source_path = Path(input_info.video_path)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        height, width = map(int, input_info.size or self.config["size"])
        frames_total = int(input_info.num_frames or self.config["num_frames"])
        picture, text, reference_size, generator = self._prepare_inference(input_info)
        encoder_stream = self.video_encoder.create_stream()
        output_rank = not dist.is_initialized() or dist.get_rank() == 0
        decoder_stream = self.video_decoder.create_stream() if output_rank else None
        reference_history = torch.empty((24, 0, reference_size[0] // 16, reference_size[1] // 16), device=AI_DEVICE)
        video_history = torch.empty((24, 0, height // 16, width // 16), device=AI_DEVICE)
        audio_history = torch.empty((2, 32, 0), device=AI_DEVICE)
        output = StreamRenderVideoOutput(input_info.return_result_tensor if output_rank else False)
        async_decoder = (
            AsyncVAEChunkDecoder.from_config(
                self.config,
                device=AI_DEVICE,
                vae_decoder=self.video_decoder,
                on_chunk=partial(self._publish_decoded, output),
                dispatch_in_worker=True,
                collect_outputs=False,
            )
            if output_rank
            else None
        )
        self.model.transformer_infer.before_graph_capture = async_decoder.synchronize if async_decoder is not None and async_decoder.is_async else None
        timings = []
        started = perf_counter()
        decode_finished = False
        try:
            with av.open(str(source_path)) as source:
                video_stream = source.streams.video[0]
                if video_stream.average_rate != 24 or (video_stream.height, video_stream.width) != (height, width):
                    raise ValueError(f"StreamRender expects 24 fps RGB at {height}x{width}")
                frames = iter(source.decode(video=0))
                for index, plan in enumerate(self.scheduler.window_plans):
                    consumed = list(islice(frames, plan.frame_count))
                    if len(consumed) != plan.frame_count:
                        raise ValueError(f"Semantic video ended before window {index + 1}")
                    encoded = self._encode_reference_window(consumed, encoder_stream, reference_size)
                    reference_history = torch.cat((reference_history, encoded), dim=1)
                    step_started = perf_counter()
                    video, audio = self._infer_window(index, reference_history, video_history, audio_history, generator, picture, text)
                    timings.append({"round": index, "dit_s": perf_counter() - step_started})
                    video_history = torch.cat((video_history, video), dim=1)
                    audio_history = torch.cat((audio_history, audio), dim=2)
                    if async_decoder is not None:
                        async_decoder.submit(decoder_stream.push, video.permute(1, 0, 2, 3).unsqueeze(0))
                if async_decoder is not None:
                    async_decoder.finish()
                    tail = decoder_stream.flush()
                    if tail is not None:
                        self._publish_decoded(output, tail, None)
                    if output.frames != frames_total:
                        raise RuntimeError(f"StreamRender decoded {output.frames} frames, expected {frames_total}")
                decode_finished = True
        finally:
            try:
                if async_decoder is not None and not decode_finished:
                    async_decoder.finish()
            finally:
                self.model.transformer_infer.before_graph_capture = None
                self.scheduler.clear()
        return {
            "video": output.tensor(),
            "audio": None,
            "output_path": str(self._stream_output_path) if output_rank and self._stream_output_path else None,
            "input_frames": frames_total,
            "output_frames": output.frames,
            "metrics": timings,
            "wall_s": perf_counter() - started,
        }


__all__ = ["StreamRenderH3Runner"]
