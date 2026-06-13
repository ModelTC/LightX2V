import os
import time
from abc import ABC
from functools import wraps

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class BaseRunner(ABC):
    """Abstract base class for all Runners

    Defines interface methods that all subclasses must implement
    """

    def __init__(self, config):
        self.config = config
        self.vae_encoder_need_img_original = False
        self.input_info = None

    def _sync_for_step_speed(self):
        if not hasattr(torch_device_module, "synchronize"):
            return
        try:
            torch_device_module.synchronize()
        except Exception:
            pass

    def apply_disagg_request_overrides(self, config_modify):
        """Mirror flat disagg request fields into ``disagg_config`` in disagg mode only."""
        if not isinstance(config_modify, dict):
            return
        if not self.config.get("disagg_mode"):
            return
        disagg_config = self.config.get("disagg_config")
        if not isinstance(disagg_config, dict):
            return

        def _safe_int(key):
            value = config_modify.get(key)
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        with self.config.temporarily_unlocked():
            data_bootstrap_room = _safe_int("data_bootstrap_room")
            if data_bootstrap_room is not None:
                self.config["data_bootstrap_room"] = data_bootstrap_room

            disagg_bootstrap_room = _safe_int("disagg_bootstrap_room")
            if disagg_bootstrap_room is not None:
                disagg_config["bootstrap_room"] = disagg_bootstrap_room
                self.config["data_bootstrap_room"] = disagg_bootstrap_room

            decoder_bootstrap_room = _safe_int("disagg_decoder_bootstrap_room")
            if decoder_bootstrap_room is not None:
                disagg_config["decoder_bootstrap_room"] = decoder_bootstrap_room

            phase1_receiver_engine_rank = _safe_int("disagg_phase1_receiver_engine_rank")
            if phase1_receiver_engine_rank is not None:
                self.config["disagg_phase1_receiver_engine_rank"] = phase1_receiver_engine_rank

            for flat_key, disagg_key in (
                ("disagg_phase1_receiver_engine_rank", "receiver_engine_rank"),
                ("disagg_phase2_sender_engine_rank", "receiver_engine_rank"),
            ):
                value = _safe_int(flat_key)
                if value is not None:
                    disagg_config[disagg_key] = value

    def load_transformer(self):
        """Load transformer model

        Returns:
            Loaded transformer model instance
        """
        pass

    def load_text_encoder(self):
        """Load text encoder

        Returns:
            Text encoder instance or list of text encoder instances
        """
        pass

    def load_image_encoder(self):
        """Load image encoder

        Returns:
            Image encoder instance or None if not needed
        """
        pass

    def load_vae(self):
        """Load VAE encoder and decoder

        Returns:
            Tuple[vae_encoder, vae_decoder]: VAE encoder and decoder instances
        """
        return None, None

    def run_image_encoder(self, img):
        """Run image encoder

        Args:
            img: Input image

        Returns:
            Image encoding result
        """
        pass

    def run_vae_encoder(self, img):
        """Run VAE encoder

        Args:
            img: Input image

        Returns:
            Tuple of VAE encoding result and additional parameters
        """
        pass

    def run_text_encoder(self, prompt, img):
        """Run text encoder

        Args:
            prompt: Input text prompt
            img: Optional input image (for some models)

        Returns:
            Text encoding result
        """
        pass

    def get_encoder_output_i2v(self, clip_encoder_out, vae_encoder_out, text_encoder_output, img):
        """Combine encoder outputs for i2v task

        Args:
            clip_encoder_out: CLIP encoder output
            vae_encoder_out: VAE encoder output
            text_encoder_output: Text encoder output
            img: Original image

        Returns:
            Combined encoder output dictionary
        """
        pass

    def init_scheduler(self):
        """Initialize scheduler."""
        if self.config.get("disagg_mode") == "decode":
            from lightx2v.models.schedulers.scheduler import NullScheduler

            self.scheduler = NullScheduler()

    def install_step_speed_timer(self, scheduler):
        """Wrap scheduler steps and report 10% bucket averages plus the full average."""
        if scheduler is None or getattr(scheduler, "_lightx2v_step_speed_timer_installed", False):
            return
        if not hasattr(scheduler, "step_pre") or not hasattr(scheduler, "step_post"):
            return

        scheduler._lightx2v_step_speed_timer_installed = True
        scheduler._lightx2v_step_speed_records = []
        scheduler._lightx2v_step_speed_start = None
        scheduler._lightx2v_step_speed_current_step = None
        scheduler._lightx2v_step_speed_emitted = False

        original_step_pre = scheduler.step_pre
        original_step_post = scheduler.step_post
        original_clear = getattr(scheduler, "clear", None)

        def _resolve_step_index(args, kwargs):
            step_index = kwargs.get("step_index")
            if step_index is None and len(args) > 0:
                # Most schedulers use step_pre(step_index); segmented schedulers
                # commonly use step_pre(segment_idx, step_index, ...).
                step_index = args[1] if len(args) > 1 else args[0]
            if step_index is None:
                step_index = getattr(scheduler, "step_index", None)
            try:
                return int(step_index)
            except (TypeError, ValueError):
                return None

        @wraps(original_step_pre)
        def timed_step_pre(*args, **kwargs):
            if scheduler._lightx2v_step_speed_emitted:
                scheduler._lightx2v_step_speed_records = []
                scheduler._lightx2v_step_speed_emitted = False
            self._sync_for_step_speed()
            scheduler._lightx2v_step_speed_start = time.perf_counter()
            result = original_step_pre(*args, **kwargs)
            scheduler._lightx2v_step_speed_current_step = _resolve_step_index(args, kwargs)
            return result

        @wraps(original_step_post)
        def timed_step_post(*args, **kwargs):
            try:
                return original_step_post(*args, **kwargs)
            finally:
                start = scheduler._lightx2v_step_speed_start
                if start is not None:
                    self._sync_for_step_speed()
                    elapsed = time.perf_counter() - start
                    step_index = scheduler._lightx2v_step_speed_current_step
                    if step_index is None:
                        step_index = getattr(scheduler, "step_index", None)
                    try:
                        total_steps = int(getattr(scheduler, "infer_steps", self.config.get("infer_steps", 0)) or 0)
                    except (TypeError, ValueError):
                        total_steps = 0
                    scheduler._lightx2v_step_speed_records.append(
                        {
                            "step_index": step_index,
                            "elapsed": elapsed,
                            "total_steps": total_steps,
                        }
                    )
                    scheduler._lightx2v_step_speed_start = None

        scheduler.step_pre = timed_step_pre
        scheduler.step_post = timed_step_post

        if original_clear is not None:

            @wraps(original_clear)
            def timed_clear(*args, **kwargs):
                self.report_step_speed(scheduler)
                return original_clear(*args, **kwargs)

            scheduler.clear = timed_clear

    def report_step_speed(self, scheduler=None):
        scheduler = scheduler or getattr(self, "scheduler", None)
        if scheduler is None or getattr(scheduler, "_lightx2v_step_speed_emitted", False):
            return

        records = getattr(scheduler, "_lightx2v_step_speed_records", [])
        valid_records = [record for record in records if isinstance(record.get("step_index"), int)]
        if not valid_records:
            return

        total_steps = max((record.get("total_steps", 0) for record in valid_records), default=0)
        if total_steps <= 0:
            total_steps = max(record["step_index"] for record in valid_records) + 1

        if dist.is_initialized() and dist.get_rank() != 0:
            scheduler._lightx2v_step_speed_emitted = True
            return

        rank_label = f"rank {dist.get_rank()}" if dist.is_initialized() else "single rank"
        elapsed_by_step = {record["step_index"]: record["elapsed"] for record in valid_records}

        for bucket_idx in range(10):
            start_ratio = bucket_idx / 10.0
            end_ratio = (bucket_idx + 1) / 10.0
            start_index = min(int(total_steps * start_ratio), total_steps - 1)
            end_index = min(int(total_steps * end_ratio), total_steps)
            if end_index <= start_index:
                end_index = min(start_index + 1, total_steps)

            selected = [elapsed_by_step[i] for i in range(start_index, end_index) if i in elapsed_by_step]
            if not selected:
                continue

            avg_seconds = sum(selected) / len(selected)
            logger.info(
                "[Speed] Step {:02d}%-{:02d}% average: {:.6f} s/step ({:.3f} ms/step), steps {}-{}/{}, samples={}, {}",
                bucket_idx * 10,
                (bucket_idx + 1) * 10,
                avg_seconds,
                avg_seconds * 1000.0,
                start_index + 1,
                end_index,
                total_steps,
                len(selected),
                rank_label,
            )

        all_elapsed = [record["elapsed"] for record in valid_records]
        avg_seconds = sum(all_elapsed) / len(all_elapsed)
        logger.info(
            "[Speed] Step full average: {:.6f} s/step ({:.3f} ms/step), steps 1-{}/{}, samples={}, {}",
            avg_seconds,
            avg_seconds * 1000.0,
            total_steps,
            total_steps,
            len(all_elapsed),
            rank_label,
        )
        scheduler._lightx2v_step_speed_emitted = True

    def load_vae_decoder(self):
        """Load VAE decoder

        Default implementation: get decoder from load_vae method
        Subclasses can override this method to provide different loading logic

        Returns:
            VAE decoder instance
        """
        if not hasattr(self, "vae_decoder") or self.vae_decoder is None:
            _, self.vae_decoder = self.load_vae()
        return self.vae_decoder

    def get_video_segment_num(self):
        self.video_segment_num = 1

    def init_run(self):
        pass

    def init_run_segment(self, segment_idx):
        self.segment_idx = segment_idx

    def run_segment(self, segment_idx=0):
        pass

    def end_run_segment(self, segment_idx=None):
        self.gen_video_final = self.gen_video

    def end_run(self):
        pass

    def check_stop(self):
        """Check if the stop signal is received"""

        rank, world_size = 0, 1
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        stop_rank = int(os.getenv("WORKER_RANK", "0")) % world_size  # same as worker hub target_rank
        pause_rank = int(os.getenv("READER_RANK", "0")) % world_size  # same as va_reader target_rank

        stopped, paused = 0, 0
        if rank == stop_rank and hasattr(self, "stop_signal") and self.stop_signal:
            stopped = 1
        if rank == pause_rank and hasattr(self, "pause_signal") and self.pause_signal:
            paused = 1

        if world_size > 1:
            if rank == stop_rank:
                t1 = torch.tensor([stopped], dtype=torch.int32).to(device=AI_DEVICE)
            else:
                t1 = torch.zeros(1, dtype=torch.int32, device=AI_DEVICE)
            if rank == pause_rank:
                t2 = torch.tensor([paused], dtype=torch.int32).to(device=AI_DEVICE)
            else:
                t2 = torch.zeros(1, dtype=torch.int32, device=AI_DEVICE)
            dist.broadcast(t1, src=stop_rank)
            dist.broadcast(t2, src=pause_rank)
            stopped = t1.item()
            paused = t2.item()

        if stopped == 1:
            try:
                self.end_run()
            except Exception as e:
                print(f"end_run failed: {e}")
            raise Exception(f"find rank: {rank} stop_signal, stop running, it's an expected behavior")
        if paused == 1:
            try:
                self.end_run()
            except Exception as e:
                print(f"end_run failed: {e}")
            raise Exception(f"find rank: {rank} pause_signal, pause running, it's an expected behavior")
