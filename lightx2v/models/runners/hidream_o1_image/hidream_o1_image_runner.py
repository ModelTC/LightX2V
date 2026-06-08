import gc
import os

import torch
import tqdm
from loguru import logger
from transformers import AutoProcessor

from lightx2v.models.networks.hidream_o1_image.model import HidreamO1ImageModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.hidream_o1_image.scheduler import HidreamO1ImageScheduler
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER


def _get_tokenizer(processor):
    from transformers import PreTrainedTokenizerBase

    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


def _add_special_tokens(tokenizer):
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


@RUNNER_REGISTER("hidream_o1_image")
class HidreamO1ImageRunner(DefaultRunner):
    """LightX2V runner for HiDream-O1-Image t2i / i2i."""

    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.default_timesteps = None
        self.dtype = torch.bfloat16

    def init_scheduler(self):
        self.scheduler = HidreamO1ImageScheduler(self.config, torch.bfloat16)

    def init_modules(self):
        task = self.config["task"]
        if task not in ("t2i", "i2i"):
            raise NotImplementedError(f"HidreamO1ImageRunner supports t2i and i2i, got: {task}")

        logger.info(f"Initializing HiDream-O1-Image {task} runner...")
        self.load_model()
        self.run_input_encoder = self._run_input_encoder_local_t2i if task == "t2i" else self._run_input_encoder_local_i2i
        self.config.lock()

    @ProfilingContext4DebugL2("Load HiDream-O1-Image model")
    def load_model(self):
        from lightx2v.models.networks.hidream_o1_image.utils import DEFAULT_TIMESTEPS

        logger.info(f"Loading HiDream processor from {self.config['model_path']}")
        self.processor = AutoProcessor.from_pretrained(self.config["model_path"])
        self.tokenizer = _get_tokenizer(self.processor)
        _add_special_tokens(self.tokenizer)
        self.dtype = self._resolve_torch_dtype()
        self.default_timesteps = DEFAULT_TIMESTEPS

        self.scheduler.set_dtype(self.dtype)
        self.model = HidreamO1ImageModel(self.config["model_path"], self.config, self.dtype)
        self.model.set_scheduler(self.scheduler)

    def _resolve_torch_dtype(self):
        dtype = self.config.get("torch_dtype", "bfloat16")
        if dtype in ["bf16", "bfloat16", "BF16"]:
            return torch.bfloat16
        if dtype in ["fp16", "float16", "FP16"]:
            return torch.float16
        if dtype in ["fp32", "float32", "FP32"]:
            return torch.float32
        raise ValueError(f"Unsupported HiDream torch_dtype: {dtype}")

    def _resolve_generation_config(self):
        model_type = self.config.get("hidream_model_type", "full")
        if model_type == "full":
            return {
                "num_inference_steps": int(self.config.get("infer_steps", self.config.get("num_inference_steps", 50))),
                "guidance_scale": float(self.config.get("guidance_scale", 5.0)),
                "shift": float(self.config.get("shift", 3.0)),
                "timesteps_list": None,
                "scheduler_name": self.config.get("scheduler_name", "default"),
                "noise_scale_start": float(self.config.get("noise_scale_start", 8.0)),
                "noise_scale_end": float(self.config.get("noise_scale_end", 8.0)),
                "noise_clip_std": float(self.config.get("noise_clip_std", 0.0)),
            }
        if model_type == "dev":
            return {
                "num_inference_steps": int(self.config.get("infer_steps", self.config.get("num_inference_steps", 28))),
                "guidance_scale": float(self.config.get("guidance_scale", 0.0)),
                "shift": float(self.config.get("shift", 1.0)),
                "timesteps_list": self.config.get("timesteps_list", self.default_timesteps),
                "scheduler_name": self.config.get("scheduler_name", "flash"),
                "noise_scale_start": float(self.config.get("noise_scale_start", 7.5)),
                "noise_scale_end": float(self.config.get("noise_scale_end", 7.5)),
                "noise_clip_std": float(self.config.get("noise_clip_std", 2.5)),
            }
        raise ValueError(f"Unsupported HiDream hidream_model_type: {model_type}")

    @ProfilingContext4DebugL2("Run HiDream input encoder")
    def _run_input_encoder_local_t2i(self):
        from lightx2v.models.networks.hidream_o1_image.utils import build_t2i_text_sample, find_closest_resolution

        generation_config = self._resolve_generation_config()
        height = self._resolve_size("height", "target_height", 2048)
        width = self._resolve_size("width", "target_width", 2048)
        width, height = find_closest_resolution(width, height)

        device = self.model.device
        samples = [
            self._sample_to_device(
                build_t2i_text_sample(
                    self.input_info.prompt,
                    height,
                    width,
                    self.tokenizer,
                    self.processor,
                    self.model.model_config,
                ),
                device,
            )
        ]
        if generation_config["guidance_scale"] > 1.0:
            samples.append(
                self._sample_to_device(
                    build_t2i_text_sample(
                        " ",
                        height,
                        width,
                        self.tokenizer,
                        self.processor,
                        self.model.model_config,
                    ),
                    device,
                )
            )

        return {
            "samples": samples,
            "height": height,
            "width": width,
            "h_patches": height // 32,
            "w_patches": width // 32,
            "seed": self.input_info.seed,
            "save_result_path": self.input_info.save_result_path,
            "generation_config": generation_config,
        }

    @ProfilingContext4DebugL2("Run HiDream i2i input encoder")
    def _run_input_encoder_local_i2i(self):
        ref_image_paths = [path.strip() for path in self.input_info.image_path.split(",") if path.strip()]
        if not ref_image_paths:
            raise ValueError("HiDream i2i requires --image_path with one or more reference image paths.")

        layout_bboxes = getattr(self.input_info, "layout_bboxes", "") or self.config.get("layout_bboxes")
        if layout_bboxes in ("", None):
            layout_bboxes = None

        from lightx2v.models.networks.hidream_o1_image.i2i_utils import build_i2i_samples

        generation_config = self._resolve_generation_config()
        inputs = build_i2i_samples(
            prompt=self.input_info.prompt,
            ref_image_paths=ref_image_paths,
            height=self._resolve_size("height", "target_height", 2048),
            width=self._resolve_size("width", "target_width", 2048),
            keep_original_aspect=getattr(self.input_info, "keep_original_aspect", False) or self.config.get("keep_original_aspect", False),
            layout_bboxes=layout_bboxes,
            tokenizer=self.tokenizer,
            processor=self.processor,
            model_config=self.model.model_config,
            device=self.model.device,
            dtype=self.dtype,
            guidance_scale=generation_config["guidance_scale"],
        )
        for sample in inputs["samples"]:
            sample["tgt_image_len"] = inputs["tgt_image_len"]
        inputs.update(
            {
                "seed": self.input_info.seed,
                "save_result_path": self.input_info.save_result_path,
                "generation_config": generation_config,
            }
        )
        return inputs

    def _sample_to_device(self, sample, device):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}

    def _resolve_size(self, config_key, target_key, default):
        if self.input_info.target_shape:
            if len(self.input_info.target_shape) == 1:
                return int(self.input_info.target_shape[0])
            if config_key == "height":
                return int(self.input_info.target_shape[0])
            return int(self.input_info.target_shape[1])
        return int(self.config.get(target_key, self.config.get(config_key, default)))

    def init_run(self):
        self.scheduler.prepare(self.inputs)

    def run_segment(self, segment_idx=0):
        for step_idx in tqdm.trange(self.scheduler.infer_steps, desc="Generating"):
            self.scheduler.step_pre(step_idx)
            self.model.infer(self.inputs)
            self.scheduler.step_post()
        return self.scheduler.latents

    def run_vae_decoder(self, latents):
        return self.scheduler.decode()

    @ProfilingContext4DebugL2("Run HiDream main")
    def run_main(self):
        self.init_run()
        latents = self.run_segment(0)
        self.gen_image = self.run_vae_decoder(latents)
        self.end_run_segment(0)
        return self.gen_image

    def end_run_segment(self, segment_idx=None):
        self.gen_video_final = self.gen_image

    @ProfilingContext4DebugL1(
        "RUN HiDream pipeline",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_worker_request_duration,
        metrics_labels=["HidreamO1ImageRunner"],
    )
    def run_pipeline(self, input_info):
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_count.inc()
        self.input_info = input_info
        self.inputs = self.run_input_encoder()
        image = self.run_main()

        save_result_path = self.inputs.get("save_result_path")
        if self.input_info.return_result_tensor:
            self.end_run()
            return {"image": image}
        if save_result_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_result_path)), exist_ok=True)
            image.save(save_result_path)
            logger.info(f"HiDream image saved successfully to: {save_result_path}")

        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        self.end_run()
        return {"image": None}

    def end_run(self):
        if hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.clear()
        elif hasattr(self, "model") and self.model is not None:
            self.model.clear()
        if hasattr(self, "inputs"):
            del self.inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
