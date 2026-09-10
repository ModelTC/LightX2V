import functools
import gc
import os
from abc import ABC

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.input_info import INPUT_INFO_TYPES, UNSET, InputInfo
from lightx2v.utils.utils import seed_all
from lightx2v_platform.base.global_var import AI_DEVICE


class BaseRunner(ABC):
    """Abstract base class for all Runners

    Defines interface methods that all subclasses must implement
    """

    input_info_cls_by_task: dict[str, type[InputInfo]] = {}
    supported_request_fields_by_task: dict[str, frozenset[str]] = {}

    def __init__(self, config):
        self.config = config
        task = config.get("task")
        if task and task not in self.supported_request_fields_by_task:
            raise ValueError(f"{type(self).__name__} does not support task {task!r}")
        self.supported_tasks = self.get_supported_tasks()
        self.vae_encoder_need_img_original = False
        self.input_info = None
        self.enable_reuse = config.get("enable_reuse", False)
        self.reuse = False
        self.reuse_prefix_segments = 0
        self._gc_frozen = False  # one-shot guard for _maybe_freeze_gc()
        self._init_modules_depth = 0
        self._warmup_done = False

    def __init_subclass__(cls, **kwargs):
        """Install common lifecycle hooks around runner entry points."""
        super().__init_subclass__(**kwargs)

        run_pipeline_fn = cls.__dict__.get("run_pipeline")
        if run_pipeline_fn is not None and not getattr(run_pipeline_fn, "_gc_freeze_wrapped", False):

            @functools.wraps(run_pipeline_fn)
            def run_pipeline(self, *args, **kwargs):
                result = run_pipeline_fn(self, *args, **kwargs)
                self._maybe_freeze_gc()
                return result

            run_pipeline._gc_freeze_wrapped = True
            cls.run_pipeline = run_pipeline

        init_modules_fn = cls.__dict__.get("init_modules")
        if init_modules_fn is not None and not getattr(init_modules_fn, "_warmup_wrapped", False):

            @functools.wraps(init_modules_fn)
            def init_modules(self, *args, **kwargs):
                depth = getattr(self, "_init_modules_depth", 0)
                self._init_modules_depth = depth + 1
                try:
                    result = init_modules_fn(self, *args, **kwargs)
                finally:
                    self._init_modules_depth = depth

                if depth == 0 and not self._warmup_done:
                    self.warmup()
                    self._warmup_done = True
                return result

            init_modules._warmup_wrapped = True
            cls.init_modules = init_modules

    def warmup(self):
        """Reject explicit warmup when a runner has no implementation."""
        if self.config.get("warmup", False):
            raise NotImplementedError(f"Warmup is not supported for {type(self).__name__}")

    def get_supported_tasks(self):
        """Return tasks accepted by this initialized runner."""
        task = self.config.get("task")
        if not task:
            raise ValueError("task must be set when the runner is created")
        return (task,)

    def create_input_info(self, request_data):
        """Create the runtime context for one inference request."""
        task = request_data["task"]
        input_info_cls = self.input_info_cls_by_task.get(task) or INPUT_INFO_TYPES[task]
        input_info = input_info_cls()
        input_info.update(self.config)
        input_info.update(request_data)

        if "aspect_ratio" in request_data and "target_shape" not in request_data:
            input_info.target_shape = []
        elif "target_shape" not in request_data and "target_shape" not in self.config and "target_height" in self.config and "target_width" in self.config:
            input_info.update({"target_shape": [self.config["target_height"], self.config["target_width"]]})

        input_info.seed = self.resolve_request_seed(request_data)
        return input_info

    def resolve_request_seed(self, request_data):
        seed = request_data.get("seed")
        return 42 if seed is None else seed

    def get_supported_request_fields(self, task):
        """Return supported request fields for the given task."""
        supported_request_fields = self.supported_request_fields_by_task[task]
        if not self.config.get("enable_cfg", False):
            supported_request_fields = supported_request_fields - {"negative_prompt"}
        return supported_request_fields

    def prepare_request(self, request_data):
        """Build and validate the runtime context for one request."""
        request_data = {key: value for key, value in request_data.items() if value is not UNSET and value is not None}
        task = request_data.get("task")
        if task is None:
            if len(self.supported_tasks) > 1:
                raise ValueError("task is required when the runner supports multiple tasks")
            task = self.supported_tasks[0]
        request_data["task"] = task
        if task not in self.supported_tasks:
            task_names = ", ".join(self.supported_tasks)
            raise ValueError(f"Task {task!r} is not supported by this runner; expected one of: {task_names}")
        unsupported_fields = set(request_data) - self.get_supported_request_fields(task)
        if unsupported_fields:
            raise ValueError(f"{type(self).__name__} ({task}) does not support request fields: {', '.join(sorted(unsupported_fields))}")
        return self.create_input_info(request_data)

    def run_request(self, input_info):
        """Run a request that has already passed request preparation."""
        if input_info.seed is not None:
            seed_all(input_info.seed)
        return self.run_pipeline(input_info)

    def set_reuse(self, reuse, reuse_prefix_segments=0):
        if reuse and not self.enable_reuse:
            raise ValueError(f"This {type(self).__name__} service does not enable reuse")
        if reuse and self.config.get("disagg_mode"):
            raise NotImplementedError(f"{type(self).__name__} reuse does not support disaggregated inference")
        if reuse_prefix_segments and not reuse:
            raise ValueError("reuse_prefix_segments requires reuse=true")
        if reuse:
            self.check_reuse_support()
        if reuse_prefix_segments:
            self.check_segment_reuse_support()
        self.reuse = reuse
        self.reuse_prefix_segments = reuse_prefix_segments

    def check_reuse_support(self):
        raise NotImplementedError(f"{type(self).__name__} does not support reuse")

    def check_segment_reuse_support(self):
        raise NotImplementedError(f"{type(self).__name__} does not support segment reuse")

    def _maybe_freeze_gc(self):
        """Move the steady-state object graph into the GC's permanent generation once."""
        if getattr(self, "_gc_frozen", False):
            return
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self._gc_frozen = True
            logger.info("[GC] skip gc.freeze(): lazy_load/unload_modules rebuilds the model each request")
            return
        # Collect before freezing. gc.freeze() merges everything currently tracked into the
        # permanent generation *without* collecting first, so any uncollected cyclic garbage
        # sitting in the generations right now would be pinned there forever (a leak) and would
        # bloat the permanent set, weakening the win. Sweeping it first means only the genuinely
        # live steady-state graph gets frozen.
        collected = gc.collect()
        n = len(gc.get_objects())
        gc.freeze()
        self._gc_frozen = True
        logger.info(f"[GC] gc.collect() reclaimed {collected} objects; gc.freeze() moved ~{n} live tracked objects out of future GC walks")

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

    def compute_usage(self, prompt: str, target_shape: list[int], has_input_image: bool = False) -> dict | None:
        """Compute token usage for the current generation.

        Returns a dict with fields matching the OpenAI Usage schema, or None if
        the runner cannot compute usage.
        """
        try:
            stride_h, stride_w = self._get_spatial_stride()
            patch_h, patch_w = self._get_spatial_patch()

            text_tokens = self._get_text_token_count(prompt)

            output_image_tokens = 0
            if target_shape and len(target_shape) >= 2:
                h, w = target_shape[0], target_shape[1]
                patched_h = max(1, h // stride_h // patch_h)
                patched_w = max(1, w // stride_w // patch_w)
                output_image_tokens = patched_h * patched_w

            input_image_tokens = output_image_tokens if has_input_image else 0
            output_tokens = output_image_tokens

            return {
                "input_tokens": text_tokens + input_image_tokens,
                "input_tokens_details": {"image_tokens": input_image_tokens, "text_tokens": text_tokens},
                "output_tokens": output_tokens,
                "total_tokens": text_tokens + input_image_tokens + output_tokens,
                "output_tokens_details": {"image_tokens": output_image_tokens, "text_tokens": 0},
            }
        except Exception:
            return None

    def _get_spatial_stride(self) -> tuple[int, int]:
        vae_stride = self.config.get("vae_stride")
        if vae_stride and len(vae_stride) >= 3:
            return vae_stride[1], vae_stride[2]
        vae_scale_factor = self.config.get("vae_scale_factor")
        if vae_scale_factor:
            sf = int(vae_scale_factor)
            return sf, sf
        return 8, 8

    def _get_spatial_patch(self) -> tuple[int, int]:
        patch_size = self.config.get("patch_size")
        if patch_size:
            if isinstance(patch_size, (list, tuple)):
                if len(patch_size) >= 3:
                    return patch_size[1], patch_size[2]
                return int(patch_size[0]), int(patch_size[0])
            return int(patch_size), int(patch_size)
        return 2, 2

    def _get_text_token_count(self, prompt: str) -> int:
        try:
            text_encoders = getattr(self, "text_encoders", None)
            if text_encoders and len(text_encoders) > 0:
                tokenizer = getattr(text_encoders[0], "tokenizer", None)
                if tokenizer:
                    return len(tokenizer.encode(prompt))
        except Exception:
            pass

        try:
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer:
                return len(tokenizer.encode(prompt))
        except Exception:
            pass

        try:
            model = getattr(self, "model", None)
            if model:
                tokenizer = getattr(model, "tokenizer", None)
                if tokenizer:
                    return len(tokenizer.encode(prompt))
        except Exception:
            pass

        return 0

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
            raise Exception(f"find rank: {rank} pause_signal, pause running, it's an expected behavior")
