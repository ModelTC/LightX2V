"""Common request, chunk decoding and local output lifecycle for AR models."""

import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.runners.lifecycle import RunnerLifecycleMixin
from lightx2v.utils.async_vae import AsyncVAEChunkDecoder
from lightx2v.utils.profiler import ProfilingContext4DebugL1, no_sync_profiling
from lightx2v.utils.streaming_video import StreamingVideoWriter, video_chunk_to_uint8
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class AutoregressiveRunnerMixin(RunnerLifecycleMixin):
    """Common lifecycle for autoregressive runners."""

    @property
    def streaming_video_only(self):
        return getattr(self, "_stream_output_file_only", False)

    def get_output_fps(self):
        return getattr(self.input_info, "output_fps", None) or self.config.get("fps", 16)

    def get_stream_output_path(self, input_info):
        path = input_info.save_result_path
        path = path.get("data") if isinstance(path, dict) else path
        return path if path and not str(path).startswith(("http://", "https://", "rtmp://")) else None

    def run_request(self, input_info):
        original_path = input_info.save_result_path
        output_path = self.get_stream_output_path(input_info)
        local_output = output_path is not None
        self._stream_output_file_only = local_output and not input_info.return_result_tensor
        self._stream_output_path = output_path if local_output else None
        self._streaming_video_output = None
        if local_output:
            fps = getattr(input_info, "output_fps", None) or self.config.get("fps", 16)
            if not dist.is_initialized() or dist.get_rank() == 0:
                self._streaming_video_output = StreamingVideoWriter(output_path, fps)
            # The shared sink owns local publication. Model-specific final
            # save functions must not overwrite the incrementally encoded file.
            input_info.save_result_path = None
        try:
            result = super().run_request(input_info)
            writer = self._streaming_video_output
            if writer is not None:
                if writer.frames == 0:
                    raise RuntimeError(f"{type(self).__name__} produced no chunks for streaming video output")
                writer.close()
                logger.info("Streamed {} frames to {}", writer.frames, writer.path)
            return result
        except BaseException:
            if self._streaming_video_output is not None:
                self._streaming_video_output.abort()
            raise
        finally:
            input_info.save_result_path = original_path
            self._streaming_video_output = None
            self._stream_output_file_only = False
            self._stream_output_path = None

    def record_video_chunk(self, pixels, *, layout="BCTHW", value_range=(-1, 1)):
        writer = getattr(self, "_streaming_video_output", None)
        if writer is not None:
            if writer.frames == 0:
                # Input encoding can resolve the source frame rate after the
                # request starts, as in Matrix-Game-4 video conditioning.
                writer.fps = self.get_output_fps()
            writer.submit(video_chunk_to_uint8(pixels, layout=layout, value_range=value_range))

    def set_stream_audio(self, waveform, sample_rate):
        writer = getattr(self, "_streaming_video_output", None)
        if writer is not None:
            writer.set_audio(waveform, sample_rate)

    def process_images_after_vae_decoder(self):
        if self.streaming_video_only:
            return {"video": None}
        return super().process_images_after_vae_decoder()

    @contextmanager
    def streaming_decode(self, *, on_chunk=None, dispatch_in_worker=True, collect_outputs=None):
        """Decode, publish and drain chunks through one shared AR lifecycle."""
        if collect_outputs is None:
            collect_outputs = not self.streaming_video_only
        record = getattr(self, "_stream_output_path", None) is not None

        def publish(output, index):
            if record:
                for pixels in output if isinstance(output, (tuple, list)) else (output,):
                    self.record_video_chunk(pixels)
            if on_chunk is not None:
                on_chunk(output, index)

        decoder = AsyncVAEChunkDecoder.from_config(
            self.config,
            device=torch.device(AI_DEVICE),
            vae_decoder=self.vae_decoder,
            on_chunk=publish if record or on_chunk is not None else None,
            dispatch_in_worker=dispatch_in_worker,
            collect_outputs=collect_outputs,
        )
        first_error = None
        try:
            with no_sync_profiling(enabled=decoder.is_async):
                yield decoder
        except BaseException as error:
            first_error = error
            raise
        finally:
            try:
                decoder.finish()
            except BaseException:
                if first_error is None:
                    raise

    def close(self):
        # Captured collectives must release their graph bindings before the
        # model weights and distributed process groups are destroyed.
        model = getattr(self, "model", None)
        reset = getattr(model, "reset_cuda_graph", None)
        if reset is None:
            reset = getattr(getattr(model, "transformer_infer", None), "reset_cuda_graph", None)
        if reset is not None:
            if AI_DEVICE == "cuda" and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            reset()
        super().close()

    def validate_warmup(self):
        if not self.supports_generic_warmup() or self.config["task"] not in self._WARMUP_TASKS:
            raise NotImplementedError(f"Autoregressive warmup is not supported for {type(self).__name__} {self.config['task']}")

    @ProfilingContext4DebugL1("Warmup")
    def run_warmup(self):
        self.validate_warmup()
        started = time.perf_counter()
        try:
            with torch.random.fork_rng(devices=[torch_device_module.current_device()], device_type=AI_DEVICE):
                self._run_warmup()
        finally:
            self.clear_warmup_state()
        self.warmup_time_seconds = time.perf_counter() - started
        self._maybe_freeze_gc()
        logger.info("[Warmup] {} completed in {:.3f}s", type(self).__name__, self.warmup_time_seconds)
