import gc

import numpy as np
import requests
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from loguru import logger
from requests.exceptions import RequestException

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.generate_task_id import generate_task_id
from lightx2v.utils.global_paras import CALIB
from lightx2v.utils.memory_profiler import peak_memory_decorator
from lightx2v.utils.profiler import *
from lightx2v.utils.utils import get_optimal_patched_size_with_sp, isotropic_crop_resize, save_to_video, vae_to_comfyui_image
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def resize_image(img, resolution, bucket_shape=None):
    assert resolution in ["480p", "540p", "720p"]
    if bucket_shape is None:
        bucket_config = {
            0.667: np.array([[480, 832], [544, 960], [720, 1280]], dtype=np.int64),
            1.500: np.array([[832, 480], [960, 544], [1280, 720]], dtype=np.int64),
            1.000: np.array([[480, 480], [576, 576], [960, 960]], dtype=np.int64),
        }
    else:
        bucket_config = {}
        for ratio, resolutions in bucket_shape.items():
            bucket_config[float(ratio)] = np.array(resolutions, dtype=np.int64)
    ori_height = img.shape[-2]
    ori_weight = img.shape[-1]
    ori_ratio = ori_height / ori_weight

    aspect_ratios = np.array(np.array(list(bucket_config.keys())))
    closet_aspect_idx = np.argmin(np.abs(aspect_ratios - ori_ratio))
    closet_ratio = aspect_ratios[closet_aspect_idx]
    if resolution == "480p":
        target_h, target_w = bucket_config[closet_ratio][0]
    elif resolution == "540p":
        target_h, target_w = bucket_config[closet_ratio][1]
    elif resolution == "720p":
        target_h, target_w = bucket_config[closet_ratio][2]

    cropped_img = isotropic_crop_resize(img, (target_h, target_w))
    logger.info(f"resize_image: {img.shape} -> {cropped_img.shape}, target_h: {target_h}, target_w: {target_w}")
    return cropped_img, target_h, target_w


class DefaultRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.has_prompt_enhancer = False
        self.progress_callback = None
        if self.config["task"] == "t2v" and self.config.get("sub_servers", {}).get("prompt_enhancer") is not None:
            self.has_prompt_enhancer = True
            if not self.check_sub_servers("prompt_enhancer"):
                self.has_prompt_enhancer = False
                logger.warning("No prompt enhancer server available, disable prompt enhancer.")
        if not self.has_prompt_enhancer:
            self.config["use_prompt_enhancer"] = False
        self.set_init_device()
        self.init_scheduler()

    def init_modules(self):
        logger.info("Initializing runner modules...")
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            self.load_model()
        elif self.config.get("lazy_load", False):
            assert self.config.get("cpu_offload", False)
        if hasattr(self, "model"):
            self.model.set_scheduler(self.scheduler)  # set scheduler to model
        if self.config["task"] == "i2v":
            self.run_input_encoder = self._run_input_encoder_local_i2v
        elif self.config["task"] == "flf2v":
            self.run_input_encoder = self._run_input_encoder_local_flf2v
        elif self.config["task"] == "t2v":
            self.run_input_encoder = self._run_input_encoder_local_t2v
        elif self.config["task"] == "vace":
            self.run_input_encoder = self._run_input_encoder_local_vace
        elif self.config["task"] == "animate":
            self.run_input_encoder = self._run_input_encoder_local_animate
        elif self.config["task"] in ["s2v", "rs2v"]:
            self.run_input_encoder = self._run_input_encoder_local_s2v
        elif self.config["task"] == "t2av":
            self.run_input_encoder = self._run_input_encoder_local_t2av
        elif self.config["task"] == "i2av":
            self.run_input_encoder = self._run_input_encoder_local_i2av
        self.config.lock()  # lock config to avoid modification
        if self.config.get("compile", False) and hasattr(self.model, "compile"):
            logger.info(f"[Compile] Compile all shapes: {self.config.get('compile_shapes', [])}")
            self.model.compile(self.config.get("compile_shapes", []))

    def set_init_device(self):
        if self.config["cpu_offload"]:
            self.init_device = torch.device("cpu")
        else:
            self.init_device = torch.device(AI_DEVICE)

    def load_vfi_model(self):
        if self.config["video_frame_interpolation"].get("algo", None) == "rife":
            from lightx2v.models.vfi.rife.rife_comfyui_wrapper import RIFEWrapper

            logger.info("Loading RIFE model...")
            return RIFEWrapper(self.config["video_frame_interpolation"]["model_path"])
        else:
            raise ValueError(f"Unsupported VFI model: {self.config['video_frame_interpolation']['algo']}")

    def load_vsr_model(self):
        if "video_super_resolution" in self.config:
            from lightx2v.models.runners.vsr.vsr_wrapper import VSRWrapper

            logger.info("Loading VSR model...")
            return VSRWrapper(self.config["video_super_resolution"]["model_path"])
        else:
            return None

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.image_encoder = self.load_image_encoder()
        self.vae_encoder, self.vae_decoder = self.load_vae()
        self.vfi_model = self.load_vfi_model() if "video_frame_interpolation" in self.config else None
        self.vsr_model = self.load_vsr_model() if "video_super_resolution" in self.config else None

    def check_sub_servers(self, task_type):
        urls = self.config.get("sub_servers", {}).get(task_type, [])
        available_servers = []
        for url in urls:
            try:
                status_url = f"{url}/v1/local/{task_type}/generate/service_status"
                response = requests.get(status_url, timeout=2)
                if response.status_code == 200:
                    available_servers.append(url)
                else:
                    logger.warning(f"Service {url} returned status code {response.status_code}")

            except RequestException as e:
                logger.warning(f"Failed to connect to {url}: {str(e)}")
                continue
        logger.info(f"{task_type} available servers: {available_servers}")
        self.config["sub_servers"][task_type] = available_servers
        return len(available_servers) > 0

    def set_inputs(self, inputs):
        self.input_info.seed = inputs.get("seed", 42)
        self.input_info.prompt = inputs.get("prompt", "")
        if self.config["use_prompt_enhancer"]:
            self.input_info.prompt_enhanced = inputs.get("prompt_enhanced", "")
        self.input_info.negative_prompt = inputs.get("negative_prompt", "")
        if "image_path" in self.input_info.__dataclass_fields__:
            self.input_info.image_path = inputs.get("image_path", "")
        if "audio_path" in self.input_info.__dataclass_fields__:
            self.input_info.audio_path = inputs.get("audio_path", "")
        if "video_path" in self.input_info.__dataclass_fields__:
            self.input_info.video_path = inputs.get("video_path", "")
        self.input_info.save_result_path = inputs.get("save_result_path", "")

    def set_config(self, config_modify):
        logger.info(f"modify config: {config_modify}")
        with self.config.temporarily_unlocked():
            self.config.update(config_modify)

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    @peak_memory_decorator
    def run_segment(self, segment_idx=0):
        infer_steps = self.model.scheduler.infer_steps

        for step_index in range(infer_steps):
            # only for single segment, check stop signal every step
            with ProfilingContext4DebugL1(
                f"Run Dit every step",
                recorder_mode=GET_RECORDER_MODE(),
                metrics_func=monitor_cli.lightx2v_run_per_step_dit_duration,
                metrics_labels=[step_index + 1, infer_steps],
            ):
                if self.video_segment_num == 1:
                    self.check_stop()
                logger.info(f"==> step_index: {step_index + 1} / {infer_steps}")

                with ProfilingContext4DebugL1("step_pre"):
                    self.model.scheduler.step_pre(step_index=step_index)

                with ProfilingContext4DebugL1("🚀 infer_main"):
                    self.model.infer(self.inputs)

                with ProfilingContext4DebugL1("step_post"):
                    self.model.scheduler.step_post()

                if self.progress_callback:
                    current_step = segment_idx * infer_steps + step_index + 1
                    total_all_steps = self.video_segment_num * infer_steps
                    self.progress_callback((current_step / total_all_steps) * 100, 100)

        if segment_idx is not None and segment_idx == self.video_segment_num - 1:
            del self.inputs
            torch_device_module.empty_cache()

        return self.model.scheduler.latents

    def run_step(self):
        self.inputs = self.run_input_encoder()
        if hasattr(self, "sr_version") and self.sr_version is not None is not None:
            self.config_sr["is_sr_running"] = True
            self.inputs_sr = self.run_input_encoder()
            self.config_sr["is_sr_running"] = False

        self.run_main(total_steps=1)

    def end_run(self):
        self.model.scheduler.clear()
        if hasattr(self, "inputs"):
            del self.inputs
        self.input_info = None
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            if hasattr(self.model, "model") and len(self.model.model) == 2:  # MultiModelStruct
                for model in self.model.model:
                    if hasattr(model.transformer_infer, "offload_manager"):
                        del model.transformer_infer.offload_manager
                        torch_device_module.empty_cache()
                        gc.collect()
                    del model
            else:
                if hasattr(self.model.transformer_infer, "offload_manager"):
                    del self.model.transformer_infer.offload_manager
                    torch_device_module.empty_cache()
                    gc.collect()
                del self.model
        if self.config.get("do_mm_calib", False):
            calib_path = os.path.join(os.getcwd(), "calib.pt")
            torch.save(CALIB, calib_path)
            logger.info(f"[CALIB] Saved calibration data successfully to: {calib_path}")
        torch_device_module.empty_cache()
        gc.collect()

    def read_image_input(self, img_path):
        if isinstance(img_path, Image.Image):
            img_ori = img_path
        else:
            img_ori = Image.open(img_path).convert("RGB")

        if GET_RECORDER_MODE():
            width, height = img_ori.size
            monitor_cli.lightx2v_input_image_len.observe(width * height)
        img = TF.to_tensor(img_ori).sub_(0.5).div_(0.5).unsqueeze(0).to(self.init_device)
        self.input_info.original_size = img_ori.size

        if self.config.get("resize_mode", None) == "adaptive":
            img, h, w = resize_image(img, self.config.get("resolution", "480p"), self.config.get("bucket_shape", None))
            logger.info(f"resize_image target_h: {h}, target_w: {w}")
            patched_h = h // self.config["vae_stride"][1] // self.config["patch_size"][1]
            patched_w = w // self.config["vae_stride"][2] // self.config["patch_size"][2]

            patched_h, patched_w = get_optimal_patched_size_with_sp(patched_h, patched_w, 1)

            latent_h = patched_h * self.config["patch_size"][1]
            latent_w = patched_w * self.config["patch_size"][2]

            latent_shape = self.get_latent_shape_with_lat_hw(latent_h, latent_w)
            target_shape = [latent_h * self.config["vae_stride"][1], latent_w * self.config["vae_stride"][2]]

            logger.info(f"target_h: {target_shape[0]}, target_w: {target_shape[1]}, latent_h: {latent_h}, latent_w: {latent_w}")

            img = torch.nn.functional.interpolate(img, size=(target_shape[0], target_shape[1]), mode="bicubic")
            self.input_info.latent_shape = latent_shape  # Important: set latent_shape in input_info
            self.input_info.target_shape = target_shape  # Important: set target_shape in input_info

        return img, img_ori

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2v(self):
        img, img_ori = self.read_image_input(self.input_info.image_path)
        clip_encoder_out = self.run_image_encoder(img) if self.config.get("use_image_encoder", True) else None
        vae_encode_out, latent_shape = self.run_vae_encoder(img_ori if self.vae_encoder_need_img_original else img)
        self.input_info.latent_shape = latent_shape  # Important: set latent_shape in input_info
        text_encoder_output = self.run_text_encoder(self.input_info)
        torch_device_module.empty_cache()
        gc.collect()
        return self.get_encoder_output_i2v(clip_encoder_out, vae_encode_out, text_encoder_output, img)

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_t2v(self):
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw()  # Important: set latent_shape in input_info
        text_encoder_output = self.run_text_encoder(self.input_info)
        torch_device_module.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_flf2v(self):
        first_frame, _ = self.read_image_input(self.input_info.image_path)
        last_frame, _ = self.read_image_input(self.input_info.last_frame_path)
        clip_encoder_out = self.run_image_encoder(first_frame, last_frame) if self.config.get("use_image_encoder", True) else None
        vae_encode_out, latent_shape = self.run_vae_encoder(first_frame, last_frame)
        self.input_info.latent_shape = latent_shape  # Important: set latent_shape in input_info
        text_encoder_output = self.run_text_encoder(self.input_info)
        torch_device_module.empty_cache()
        gc.collect()
        return self.get_encoder_output_i2v(clip_encoder_out, vae_encode_out, text_encoder_output)

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_vace(self):
        src_video = self.input_info.src_video
        src_mask = self.input_info.src_mask
        src_ref_images = self.input_info.src_ref_images
        src_video, src_mask, src_ref_images = self.prepare_source(
            [src_video],
            [src_mask],
            [None if src_ref_images is None else src_ref_images.split(",")],
            (self.config["target_width"], self.config["target_height"]),
        )
        self.src_ref_images = src_ref_images

        vae_encoder_out, latent_shape = self.run_vae_encoder(src_video, src_ref_images, src_mask)
        self.input_info.latent_shape = latent_shape  # Important: set latent_shape in input_info
        text_encoder_output = self.run_text_encoder(self.input_info)
        torch_device_module.empty_cache()
        gc.collect()
        return self.get_encoder_output_i2v(None, vae_encoder_out, text_encoder_output)

    @ProfilingContext4DebugL2("Run Text Encoder")
    def _run_input_encoder_local_animate(self):
        text_encoder_output = self.run_text_encoder(self.input_info)
        torch_device_module.empty_cache()
        gc.collect()
        return self.get_encoder_output_i2v(None, None, text_encoder_output, None)

    def _run_input_encoder_local_s2v(self):
        pass

    def init_run(self):
        self.gen_video_final = None
        self.get_video_segment_num()

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)

        self.model.scheduler.prepare(seed=self.input_info.seed, latent_shape=self.input_info.latent_shape, image_encoder_output=self.inputs["image_encoder_output"])
        if self.config.get("model_cls") == "wan2.2" and self.config["task"] in ["i2v", "s2v", "rs2v"]:
            self.inputs["image_encoder_output"]["vae_encoder_out"] = None

        if hasattr(self, "sr_version") and self.sr_version is not None:
            self.lq_latents_shape = self.model.scheduler.latents.shape
            self.model_sr.set_scheduler(self.scheduler_sr)
            self.config_sr["is_sr_running"] = True
            self.inputs_sr = self.run_input_encoder()
            self.config_sr["is_sr_running"] = False

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        self.init_run()
        if self.config.get("compile", False) and hasattr(self.model, "comple"):
            self.model.select_graph_for_compile(self.input_info)
        for segment_idx in range(self.video_segment_num):
            logger.info(f"🔄 start segment {segment_idx + 1}/{self.video_segment_num}")
            with ProfilingContext4DebugL1(
                f"segment end2end {segment_idx + 1}/{self.video_segment_num}",
                recorder_mode=GET_RECORDER_MODE(),
                metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                metrics_labels=["DefaultRunner"],
            ):
                self.check_stop()
                # 1. default do nothing
                self.init_run_segment(segment_idx)
                # 2. main inference loop
                latents = self.run_segment(segment_idx)
                # 3. vae decoder
                if self.config.get("use_stream_vae", False):
                    frames = []
                    for frame_segment in self.run_vae_decoder_stream(latents):
                        frames.append(frame_segment)
                        logger.info(f"frame sagment: {len(frames)} done")
                    self.gen_video = torch.cat(frames, dim=2)
                else:
                    self.gen_video = self.run_vae_decoder(latents)
                # 4. default do nothing
                self.end_run_segment(segment_idx)
        gen_video_final = self.process_images_after_vae_decoder()
        self.end_run()
        return gen_video_final

    @ProfilingContext4DebugL1("Run VAE Decoder", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_run_vae_decode_duration, metrics_labels=["DefaultRunner"])
    def run_vae_decoder(self, latents):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_decoder = self.load_vae_decoder()
        images = self.vae_decoder.decode(latents.to(GET_DTYPE()))
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_decoder
            torch_device_module.empty_cache()
            gc.collect()
        return images

    @ProfilingContext4DebugL1("Run VAE Decoder Stream", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_run_vae_decode_duration, metrics_labels=["DefaultRunner"])
    def run_vae_decoder_stream(self, latents):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_decoder = self.load_vae_decoder()

        for frame_segment in self.vae_decoder.decode_stream(latents.to(GET_DTYPE())):
            yield frame_segment

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_decoder
            torch_device_module.empty_cache()
            gc.collect()

    def post_prompt_enhancer(self):
        while True:
            for url in self.config["sub_servers"]["prompt_enhancer"]:
                response = requests.get(f"{url}/v1/local/prompt_enhancer/generate/service_status").json()
                if response["service_status"] == "idle":
                    response = requests.post(
                        f"{url}/v1/local/prompt_enhancer/generate",
                        json={
                            "task_id": generate_task_id(),
                            "prompt": self.config["prompt"],
                        },
                    )
                    enhanced_prompt = response.json()["output"]
                    logger.info(f"Enhanced prompt: {enhanced_prompt}")
                    return enhanced_prompt

    def process_images_after_vae_decoder(self):
        self.gen_video_final = vae_to_comfyui_image(self.gen_video_final)

        if "video_frame_interpolation" in self.config:
            assert self.vfi_model is not None and self.config["video_frame_interpolation"].get("target_fps", None) is not None
            target_fps = self.config["video_frame_interpolation"]["target_fps"]
            logger.info(f"Interpolating frames from {self.config.get('fps', 16)} to {target_fps}")
            self.gen_video_final = self.vfi_model.interpolate_frames(
                self.gen_video_final,
                source_fps=self.config.get("fps", 16),
                target_fps=target_fps,
            )

        if self.input_info.return_result_tensor:
            return {"video": self.gen_video_final}
        elif self.input_info.save_result_path is not None:
            if "video_frame_interpolation" in self.config and self.config["video_frame_interpolation"].get("target_fps"):
                fps = self.config["video_frame_interpolation"]["target_fps"]
            else:
                fps = self.config.get("fps", 16)

            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info(f"🎬 Start to save video 🎬")

                save_to_video(self.gen_video_final, self.input_info.save_result_path, fps=fps, method="ffmpeg")
                logger.info(f"✅ Video saved successfully to: {self.input_info.save_result_path} ✅")
            return {"video": None}

    @ProfilingContext4DebugL1("RUN pipeline", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_worker_request_duration, metrics_labels=["DefaultRunner"])
    def run_pipeline(self, input_info):
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_count.inc()
        self.input_info = input_info

        if self.config["use_prompt_enhancer"]:
            self.input_info.prompt_enhanced = self.post_prompt_enhancer()

        self.inputs = self.run_input_encoder()

        gen_video_final = self.run_main()

        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        return gen_video_final

    def __del__(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "text_encoders"):
            del self.text_encoders
        if hasattr(self, "image_encoder"):
            del self.image_encoder
        if hasattr(self, "vae_encoder"):
            del self.vae_encoder
        if hasattr(self, "vae_decoder"):
            del self.vae_decoder
        torch_device_module.empty_cache()
        gc.collect()
