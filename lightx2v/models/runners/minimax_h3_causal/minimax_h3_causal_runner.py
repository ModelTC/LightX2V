from contextlib import ExitStack
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.audio_encoders.hf.minimax_h3_causal.driving import MiniMaxH3DrivingAudio
from lightx2v.models.audio_encoders.hf.minimax_h3_causal.streaming import MiniMaxH3StreamingAudioDecoder
from lightx2v.models.audio_encoders.hf.minimax_h3_causal.vae_graph import MiniMaxH3AudioDecoderGraph
from lightx2v.models.networks.minimax_h3.packing import unpatchify_video_tokens, video_latent_num_frames
from lightx2v.models.networks.minimax_h3_causal.clean.model import MiniMaxH3CleanModel
from lightx2v.models.networks.minimax_h3_causal.clean.streaming import GLOBAL_KEY, REFERENCE_KEY, prompt_key
from lightx2v.models.runners.h3.autoregressive_runner import H3AutoregressiveRunnerMixin
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v.models.runners.minimax_h3_causal.motion import prepare_motions
from lightx2v.models.runners.minimax_h3_causal.reference_cache import MiniMaxH3ReferenceCache
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS, VIDEO_OUTPUT_FIELDS
from lightx2v.models.schedulers.minimax_h3_causal.clean_scheduler import MiniMaxH3CleanScheduler
from lightx2v.models.video_encoders.hf.ltx2.audio_vae.ops import Audio
from lightx2v.models.video_encoders.hf.minimax_h3_causal.streaming import MiniMaxH3StreamingVideoDecoder
from lightx2v.models.video_encoders.hf.minimax_h3_causal.vae_graph import MiniMaxH3VideoDecoderGraph
from lightx2v.models.video_encoders.hf.minimax_h3_causal.video_writer import MiniMaxH3StreamingVideoWriter
from lightx2v.utils.profiler import ProfilingContext4DebugL2, no_sync_profiling
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import is_main_process
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("minimax_h3_causal")
class MiniMaxH3CausalRunner(H3AutoregressiveRunnerMixin, MiniMaxH3Runner):
    """Shared v0.1 CleanAR loop; each instance serves one configured task.

    RefA2V supplies driving audio latents; RefA2AV denoises audio with video.
    Both use the same motion prefill, clean KV history and video output path.
    """

    supported_lora_modes = ("merge",)

    supported_request_fields_by_task = {task: COMMON_REQUEST_FIELDS | VIDEO_OUTPUT_FIELDS | {"image_path", "audio_path", "prompt", "motions"} for task in ("refa2v", "refa2av")}

    def __init__(self, config):
        if config["task"] not in ("refa2v", "refa2av"):
            raise ValueError("MiniMax-H3 causal requires task='refa2v' or 'refa2av'")
        self.generate_audio = config["task"] == "refa2av"
        self.reuse_input_audio = config.get("reuse_input_audio", False)
        unsupported = [
            key
            for key in (
                "cpu_offload",
                "video_vae_shared_cpu_weights",
                "warmup",
                "async_vae_decode",
            )
            if config.get(key, False)
        ]
        if unsupported:
            raise ValueError(f"Causal H3 does not support {unsupported}")
        if self.generate_audio and self.reuse_input_audio:
            raise ValueError("RefA2AV generates audio; reuse_input_audio is only supported by RefA2V")
        if config.get("dit_quantized", False) and config.get("dit_quant_scheme") != "fp8-f32-accum":
            raise ValueError("Causal H3 DiT quantization requires dit_quant_scheme='fp8-f32-accum'")
        if config.get("text_encoder_quantized", False) and config.get("text_encoder_quant_scheme") != "fp8-sgl":
            raise ValueError("Causal H3 text encoder quantization requires text_encoder_quant_scheme='fp8-sgl'")
        if config.get("video_vae_quantized", False) and config.get("video_vae_quant_scheme") != "fp8-f32-accum":
            raise ValueError("Causal H3 video VAE quantization requires video_vae_quant_scheme='fp8-f32-accum'")
        if config.get("fps", 24) != 24:
            raise ValueError("Causal H3 requires fps=24")
        if config.get("vae_sensitive_layer_dtype", "fp32") != "fp32":
            raise ValueError("Causal H3 VAE sensitive layers require vae_sensitive_layer_dtype='fp32'")
        if config.get("lora_configs"):
            raise ValueError("Causal H3 loads the distilled checkpoint directly; released H3 LoRA adapters are not supported")

        self.reference_cache = MiniMaxH3ReferenceCache() if config.get("reference_condition_cache", False) else None
        self.driving_audio = None
        self.video_writer = None
        self.audio_decode_graph = None
        self.video_decode_graph = None
        super().__init__(config)
        # Called on rank 0 with (RGB [1,3,F,H,W], start_frame, is_final).
        # With streaming encoding, RGB is borrowed until the callback returns.
        self.stream_callback = None

    def get_supported_tasks(self):
        return (self.config["task"],)

    def load_model(self):
        if self.reference_cache is not None:
            self.reference_cache.clear()
        super().load_model()

    def load_transformer(self):
        return MiniMaxH3CleanModel(self.config["model_path"], self.config, self.init_device)

    def load_vae(self):
        video_vae, audio_vae = super().load_vae()
        video_vae.stitch_preallocate = self.config.get("vae_stitch_preallocate", False)
        if self.config.get("vae_decode_overlap", False):
            self.video_decode_graph = MiniMaxH3VideoDecoderGraph(video_vae)
            logger.info("MiniMax-H3 causal VAE: tile CUDA Graph + DiT overlap enabled")
        if self.config.get("vae_encoder_cpu_offload", False) and not video_vae.cpu_offload:
            video_vae.encoder.cpu()
            video_vae.quant_conv.cpu()
        # The decoder is unused when the output reuses the driving waveform.
        if self.reuse_input_audio and not audio_vae.cpu_offload:
            audio_vae.decoder.cpu()
            audio_vae.dec_in_proj.cpu()
        if not self.reuse_input_audio and self.config.get("audio_vae_decode_overlap", False) and is_main_process():
            self.audio_decode_graph = MiniMaxH3AudioDecoderGraph(audio_vae)
            logger.info("MiniMax-H3 causal: FP32 audio CUDA Graph + clean commit overlap enabled")
        return video_vae, audio_vae

    def init_scheduler(self):
        self.scheduler = MiniMaxH3CleanScheduler(self.config)

    def _resolve_request_geometry(self, geometry_image=None):
        self.request_height, self.request_width = map(int, self.input_info.size or self.config["size"])
        num_frames = self.input_info.num_frames
        self.request_num_frames = int(self.config["num_frames"] if num_frames is None else num_frames)
        if self.request_num_frames < 22 or min(self.request_height, self.request_width) <= 0 or self.request_height % 32 or self.request_width % 32:
            raise ValueError("Causal H3 requires at least 22 frames and positive canvas dimensions divisible by 32")
        self.input_info.size = [self.request_height, self.request_width]
        self.input_info.num_frames = self.request_num_frames

        video_latent_num_frames(self.request_num_frames)

    def _encode_references(self, references):
        if self.video_vae.cpu_offload or not self.config.get("vae_encoder_cpu_offload", False):
            return super()._encode_references(references)
        try:
            self.video_vae.encoder.to(self.video_vae.execution_device)
            self.video_vae.quant_conv.to(self.video_vae.execution_device)
            return super()._encode_references(references)
        finally:
            self.video_vae.encoder.cpu()
            self.video_vae.quant_conv.cpu()

    def _get_cached_conditioning(self):
        if self.reference_cache is None:
            return None, None
        cached = self.reference_cache.get(self.prepared_references, prompt=self.input_info.prompt)
        text, latents = cached if cached is not None else (None, None)
        hits = [latents is not None, text is not None]
        if dist.is_initialized():
            # All ranks must take the same encoding branches, including
            # after a cancelled/failed request left only some caches filled.
            flags = torch.tensor(hits, dtype=torch.int32, device=AI_DEVICE)
            dist.all_reduce(flags, op=dist.ReduceOp.MIN)
            hits = flags.tolist()
        if not hits[0]:
            latents = None
        if not hits[1]:
            text = None
        if is_main_process():
            logger.info("H3 reference cache: references={}, global={}", "hit" if hits[0] else "miss", "hit" if hits[1] else "miss")
        return text, latents

    def _run_input_encoder_local_h3(self):
        self.clear_conditioning_state()
        self._resolve_request_geometry()
        info = self.input_info
        if not info.image_path or not info.audio_path or not info.prompt.strip():
            raise ValueError("Causal H3 requires image_path, audio_path, and a global prompt")
        self.motion_by_chunk = prepare_motions(info.motions, prompt=info.prompt, num_frames=self.request_num_frames)
        reference_sources = {"image": info.image_path}
        if self.generate_audio:
            reference_sources["audio"] = info.audio_path
        self.prepared_references = self._prepare_references(reference_sources)
        if self.generate_audio and sum(reference.kind == "audio" for reference in self.prepared_references) != 1:
            raise ValueError("RefA2AV requires exactly one voice reference")
        encoder = self.text_encoders[0]
        text, latents = self._get_cached_conditioning()
        update_cache = text is None or latents is None
        if text is None:
            text = encoder.infer(info.prompt, references=self.prepared_references)
        # Encode later motions only when their chunk begins.
        motion = encoder.infer(self.motion_by_chunk[0])
        if latents is None:
            latents = self._encode_references(self.prepared_references)
        if self.reference_cache is not None and update_cache:
            self.reference_cache.store(self.prepared_references, text, latents, prompt=info.prompt)
        self.condition_video_latents, self.condition_audio_latents = latents
        if not self.generate_audio:
            self.driving_audio = MiniMaxH3DrivingAudio(
                self.audio_vae,
                info.audio_path,
                num_frames=self.request_num_frames,
                mono=self.config.get("audio_mono", False),
                loop=self.config.get("loop_audio", False),
            )
        return {"text_encoder_output": text, "motion_text_output": motion}

    def clear_conditioning_state(self):
        super().clear_conditioning_state()
        self.driving_audio = None
        self.motion_by_chunk = {}

    def get_stream_output_path(self, input_info):
        # This runner owns joint AV output, including generated audio.
        return None

    def init_video_writer(self):
        output_path = super().get_stream_output_path(self.input_info)
        if not output_path or not self.config.get("video_streaming_encode", False) or self.input_info.return_result_tensor:
            return
        if Path(output_path).suffix.lower() != ".mp4":
            raise ValueError("MiniMax-H3 AV output uses H.264/AAC; save_result_path must end in .mp4")
        if is_main_process():
            self.video_writer = MiniMaxH3StreamingVideoWriter(
                output_path,
                self.request_num_frames,
                int(self.config.get("fps", 24)),
                audio_sampling_rate=self.audio_vae.sampling_rate,
                pinned_memory=self.config.get("vae_rgb_pinned_memory", False),
                video_codec_options=self.config.get("video_codec_options"),
            )

    def init_run(self):
        text_blocks = [self.inputs["text_encoder_output"], self.inputs["motion_text_output"]]
        self.scheduler.prepare(
            seed=self.input_info.seed,
            num_frames=self.request_num_frames,
            height=self.request_height,
            width=self.request_width,
            text_token_tags=[block["text_token_tags"] for block in text_blocks],
            condition_video_latents=self.condition_video_latents,
            condition_audio_latents=self.condition_audio_latents,
            references=self.prepared_references,
            segment_starts=[0],
        )
        logger.info(f"{self.config['task']}: {self.request_num_frames} frames, {self.scheduler.plan.num_chunks} DiT chunks, first motion encoded; later motions encode at chunk boundaries")
        self.init_video_writer()

    def prefill_clean_condition(self, entry_key, text):
        self.scheduler.prepare_condition(entry_key)
        self.model.infer({"text_encoder_output": text})

    def _run_dit_chunk(self, index, media_inputs, audio_decoder):
        scheduler = self.scheduler
        audio_latents = None if self.generate_audio else self.driving_audio.encode(scheduler.plan.chunk_slices(index)[1])
        scheduler.prepare_chunk(index, audio_latents=audio_latents)
        for step in range(scheduler.infer_steps):
            scheduler.step_pre(step)
            self.model.infer(media_inputs)
            scheduler.step_post()
        # Submit audio before the clean forward so decoding can overlap it.
        if audio_decoder is not None and self.audio_decode_graph is not None:
            audio_decoder.submit(scheduler.audio_latents)
        # The final chunk has no successor that could read its clean KV.
        if index + 1 < scheduler.plan.num_chunks:
            scheduler.prepare_chunk(index, clean=True)
            self.model.infer(media_inputs)
        scheduler.cache.finish_chunk(index)

    def run_segment(self, segment_idx=0):
        with no_sync_profiling(self.video_decode_graph is not None), ExitStack() as decoders:
            try:
                scheduler = self.scheduler
                global_text = self.inputs["text_encoder_output"]
                empty_text = {"prompt_embeds": global_text["prompt_embeds"][:0]}
                media_inputs = {"text_encoder_output": empty_text}
                self.prefill_clean_condition(GLOBAL_KEY, global_text)
                self.prefill_clean_condition(REFERENCE_KEY, empty_text)
                scheduler.cache.finalize_condition()
                video_decoder = MiniMaxH3StreamingVideoDecoder(
                    self.video_vae,
                    self.request_num_frames,
                    output_uint8=not self.input_info.return_result_tensor,
                    decode_graph=self.video_decode_graph,
                    output_writer=self.video_writer,
                )
                decoders.callback(video_decoder.close)
                main_rank = is_main_process()
                audio_decoder = None
                if main_rank and not self.reuse_input_audio:
                    audio_decoder = MiniMaxH3StreamingAudioDecoder(self.audio_vae, num_audio_latents=scheduler.num_audio_latents, decode_graph=self.audio_decode_graph)
                    decoders.callback(audio_decoder.close)
                video_outputs, audio_outputs = [], []
                emitted_frames = 0

                def emit_video(frames, *, final=False):
                    nonlocal emitted_frames
                    if not main_rank or frames is None:
                        return
                    if self.video_writer is None:
                        video_outputs.append(frames)
                    if self.stream_callback is not None:
                        self.stream_callback(frames, emitted_frames, final)
                    emitted_frames += frames.shape[2]

                if self.video_writer is not None:
                    self.video_writer.on_rgb_ready = lambda frames: emit_video(frames, final=emitted_frames + frames.shape[2] == self.request_num_frames)

                for index in range(scheduler.plan.num_chunks):
                    self.check_stop()
                    if index == 0:
                        self.prefill_clean_condition(prompt_key(0), self.inputs["motion_text_output"])
                    elif index in self.motion_by_chunk:
                        text = self.text_encoders[0].infer(self.motion_by_chunk[index])
                        scheduler.append_motion(index, text["text_token_tags"])
                        self.prefill_clean_condition(prompt_key(scheduler.plan.segment_index(index)), text)
                    self._run_dit_chunk(index, media_inputs, audio_decoder)
                    num_latent_frames = int(scheduler.plan.chunk_specs[index, 1] - scheduler.plan.chunk_specs[index, 0])
                    latents = unpatchify_video_tokens(scheduler.video_latents, num_latent_frames, scheduler.latent_height, scheduler.latent_width)
                    for frames in video_decoder.push(latents):
                        emit_video(frames)
                    if audio_decoder is not None:
                        ready_audio = audio_decoder.push(scheduler.audio_latents) if self.audio_decode_graph is None else audio_decoder.flush()
                        audio_outputs.extend(audio.cpu() for _, audio in ready_audio)
                    if main_rank:
                        logger.info(f"{self.config['task']} chunk {index + 1}/{scheduler.plan.num_chunks}")
                    if self.progress_callback:
                        self.progress_callback(100 * (index + 1) / scheduler.plan.num_chunks, 100)
                # Emit the pending 17 frames before the final 5, retaining CPU-ready
                # batch boundaries for the stream callback.
                emit_video(video_decoder.flush())
                emit_video(video_decoder.finish(), final=True)
                if main_rank:
                    if audio_decoder is not None:
                        _, audio_tail = audio_decoder.finish()
                        audio_outputs.append(audio_tail.cpu())
                    video = torch.cat(video_outputs, dim=2) if self.video_writer is None else None
                    audio = self.driving_audio.waveform.unsqueeze(0) if self.reuse_input_audio else torch.cat(audio_outputs, dim=-1)
                    return video, audio
                return None, None
            finally:
                # Drain all request streams before ExitStack releases decoder tensors.
                torch.cuda.synchronize()

    def run_vae_decoder(self, video, audio_rows):
        # Local work was drained by run_segment; rendezvous before saving.
        if dist.is_initialized():
            dist.barrier()
        return video, audio_rows

    def process_images_after_vae_decoder(self):
        if not is_main_process():
            return {"video": None, "audio": None}
        if self.video_writer is None:
            return super().process_images_after_vae_decoder()
        audio = Audio(waveform=self.gen_audio[0].float().cpu(), sampling_rate=self.audio_vae.sampling_rate)
        # Only residual encoding remains here; earlier chunks overlapped DiT.
        with ProfilingContext4DebugL2("Finish Audio-Video Output"):
            self.video_writer.finish(audio=audio)
        logger.info("Video saved to {}", self.video_writer.output_path)
        return {"video": None, "audio": None}

    def run_request(self, input_info):
        try:
            return super().run_request(input_info)
        finally:
            # run_main clears the request after inference; input encoding can
            # fail before entering that scope.
            if self.input_info is not None:
                self.end_run()

    def end_run(self):
        try:
            if self.video_writer is not None:
                self.video_writer.close()
        finally:
            self.video_writer = None
            super().end_run()

    def close(self):
        if self.audio_decode_graph is not None:
            self.audio_decode_graph.clear()
        if self.reference_cache is not None:
            self.reference_cache.clear()
        if self.video_decode_graph is not None:
            self.video_decode_graph.clear()
        super().close()
