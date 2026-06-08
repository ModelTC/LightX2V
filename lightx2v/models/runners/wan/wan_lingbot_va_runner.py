import gc
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger

from lightx2v.common.kvcache import KVCacheManager
from lightx2v.models.networks.wan.lingbot_va_model import WanLingbotVAModel
from lightx2v.models.runners.wan.wan_runner import Wan22DenseRunner
from lightx2v.models.schedulers.wan.lingbot_va.scheduler import LingbotVAFlowMatchScheduler
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import GET_DTYPE, GET_RECORDER_MODE
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import save_to_video
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


@RUNNER_REGISTER("lingbot_va")
class LingbotVARunner(Wan22DenseRunner):
    def __init__(self, config):
        config["enable_cfg"] = config.get("enable_cfg", config.get("sample_guide_scale", 1.0) > 1)
        config["enable_action_cfg"] = config.get("enable_action_cfg", config.get("action_sample_guide_scale", 1.0) > 1)
        config["cfg_parallel"] = False
        super().__init__(config)
        self.cache_name = "pos"

    def init_scheduler(self):
        self.scheduler = LingbotVAFlowMatchScheduler(self.config, shift_key="sample_shift", infer_steps_key="infer_steps")
        self.action_scheduler = LingbotVAFlowMatchScheduler(self.config, shift_key="action_sample_shift", infer_steps_key="action_infer_steps")

    def load_transformer(self):
        return WanLingbotVAModel(
            model_path=self.config["model_path"],
            config=self.config,
            device=self.init_device,
            model_type="lingbot_va",
        )

    @ProfilingContext4DebugL1("init kv cache manager")
    def init_kv_cache_manager(self):
        kv_mgr = getattr(self.model, "kv_cache_manager", None)
        if kv_mgr is None:
            kv_mgr = KVCacheManager(config=self.config, device=torch.device(AI_DEVICE))
            self.model.kv_cache_manager = kv_mgr
        kv_mgr.ar_config = dict(self.config.get("ar_config", {}))
        self.model.transformer_infer.kv_cache_manager = kv_mgr
        return kv_mgr

    def _get_ar_config(self):
        ar_config = self.config.get("ar_config", {})
        if "num_frame_per_chunk" not in ar_config:
            raise ValueError("LingBot-VA requires ar_config.num_frame_per_chunk.")
        if "num_action_per_frame" not in ar_config:
            raise ValueError("LingBot-VA requires ar_config.num_action_per_frame.")
        if "num_chunks" not in ar_config:
            raise ValueError("LingBot-VA requires ar_config.num_chunks.")
        return ar_config

    def _cache_name_for_cfg_mode(self, enable_cfg):
        if enable_cfg or not (self.use_cfg or self.use_action_cfg):
            return self.cache_name
        return self.model.cfg_cache_name(self.cache_name, True)

    def init_modules(self):
        super().init_modules()
        if self.config["task"] == "i2va":
            self.run_input_encoder = self._run_input_encoder_local_i2va

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2va(self):
        text_encoder_output = self.run_text_encoder(self.input_info)
        torch_device_module.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }

    def _encode_obs(self, obs):
        images = obs["obs"]
        if not isinstance(images, list):
            images = [images]
        videos = []
        for cam_idx, key in enumerate(self.config["obs_cam_keys"]):
            if self.config["env_type"] == "robotwin_tshape":
                height_i, width_i = (self.height, self.width) if cam_idx == 0 else (self.height // 2, self.width // 2)
            else:
                height_i, width_i = self.height, self.width
            history_video = torch.from_numpy(np.stack([item[key] for item in images])).float().permute(3, 0, 1, 2)
            history_video = F.interpolate(history_video, size=(height_i, width_i), mode="bilinear", align_corners=False).unsqueeze(0)
            videos.append(history_video)

        if self.config["env_type"] == "robotwin_tshape":
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:], dim=0) / 255.0 * 2.0 - 1.0
            enc_high = self.vae_encoder.encode(videos_high.to(AI_DEVICE).to(self.vae_encoder.dtype))
            enc_left_right = self.vae_encoder.encode(videos_left_and_right.to(AI_DEVICE).to(self.vae_encoder.dtype))
            if enc_high.dim() == 4:
                enc_high = enc_high.unsqueeze(0)
            if enc_left_right.dim() == 4:
                enc_left_right = enc_left_right.unsqueeze(0)
            enc_out = torch.cat([torch.cat(enc_left_right.split(1, dim=0), dim=-1), enc_high], dim=-2)
        else:
            videos_all = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            enc_out = self.vae_encoder.encode(videos_all.to(AI_DEVICE).to(self.vae_encoder.dtype))
            if enc_out.dim() == 4:
                enc_out = enc_out.unsqueeze(0)
            enc_out = torch.cat(enc_out.split(1, dim=0), dim=-1)
        return enc_out.to(AI_DEVICE).to(GET_DTYPE())

    def init_run(self):
        self.gen_video_final = None
        self.gen_video = None
        self.pred_action = None
        self.frame_st_id = 0
        self.init_latent = None
        self.pred_latent_lst = []
        self.pred_action_lst = []
        self.init_obs = self._load_init_obs()
        self.use_cfg = self.config.get("enable_cfg", False)
        self.use_action_cfg = self.config.get("enable_action_cfg", False)

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)

        kv_mgr = self.init_kv_cache_manager()
        self.model.clear_cache(self.cache_name)

        ar_config = self._get_ar_config()
        self.num_frame_per_chunk = ar_config["num_frame_per_chunk"]
        self.num_action_per_frame = ar_config["num_action_per_frame"]
        self.num_chunks = ar_config["num_chunks"]
        self.height, self.width = self.config["target_height"], self.config["target_width"]
        if self.config["env_type"] == "robotwin_tshape":
            self.latent_height = ((self.height // 16) * 3) // 2
            self.latent_width = self.width // 16
        else:
            self.latent_height = self.height // 16
            self.latent_width = self.width // 16 * len(self.config["obs_cam_keys"])

        patch_size = tuple(self.config["patch_size"])
        latent_token_per_chunk = (self.num_frame_per_chunk * self.latent_height * self.latent_width) // (patch_size[0] * patch_size[1] * patch_size[2])
        action_token_per_chunk = self.num_frame_per_chunk * self.num_action_per_frame
        local_attn_size = ar_config.get("local_attn_size")
        if local_attn_size is None:
            raise ValueError("LingBot-VA requires ar_config.local_attn_size for FIFO KV cache sizing.")
        cache_names = [self.model.cfg_cache_name(self.cache_name, True), self.model.cfg_cache_name(self.cache_name, False)] if self.use_cfg or self.use_action_cfg else [self.cache_name]
        kv_size = (local_attn_size // 2) * latent_token_per_chunk + (local_attn_size // 2) * action_token_per_chunk
        for cache_name in cache_names:
            kv_mgr.create_self_attn_kv_cache(
                cache_name=cache_name,
                kv_size=kv_size,
                kv_cache_scheme=ar_config.get("kv_cache_scheme", "fifo"),
                step_kv_cache=ar_config.get("step_kv_cache", False),
                dtype=GET_DTYPE(),
            )

        self.action_mask = torch.zeros([self.config["action_dim"]], dtype=torch.bool, device=AI_DEVICE)
        self.action_mask[self.config["used_action_channel_ids"]] = True
        self.actions_q01 = torch.tensor(self.config["norm_stat"]["q01"], dtype=torch.float32, device=AI_DEVICE).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.config["norm_stat"]["q99"], dtype=torch.float32, device=AI_DEVICE).reshape(-1, 1, 1)
        text_encoder_output = self.inputs["text_encoder_output"]
        self.prompt_embeds = text_encoder_output["context"].to(AI_DEVICE)
        self.negative_prompt_embeds = text_encoder_output.get("context_null")
        if self.negative_prompt_embeds is not None:
            self.negative_prompt_embeds = self.negative_prompt_embeds.to(AI_DEVICE)
        elif self.use_cfg or self.use_action_cfg:
            raise ValueError("LingBot-VA CFG is enabled but text_encoder_output does not include context_null.")

    def init_run_segment(self, segment_idx):
        super().init_run_segment(segment_idx)
        self.frame_st_id = segment_idx * self.num_frame_per_chunk

    def _prepare_model_input(
        self,
        latent_model_input,
        action_model_input,
        latent_t=0,
        action_t=0,
        latent_cond=None,
        action_cond=None,
        frame_st_id=0,
    ):
        patch_size = tuple(self.config["patch_size"])
        input_dict = {}
        if latent_model_input is not None:
            latent_res = {
                "noisy_latents": latent_model_input,
                "timesteps": torch.ones([1, latent_model_input.shape[2]], dtype=torch.float32, device=AI_DEVICE) * latent_t,
                "grid_id": self.model.build_grid_id(
                    latent_model_input.shape[-3] // patch_size[0],
                    latent_model_input.shape[-2] // patch_size[1],
                    latent_model_input.shape[-1] // patch_size[2],
                    0,
                    1,
                    frame_st_id,
                )[None].to(AI_DEVICE),
                "text_emb": self.prompt_embeds.to(GET_DTYPE()).clone(),
            }
            if self.use_cfg:
                latent_res["negative_text_emb"] = self.negative_prompt_embeds.to(GET_DTYPE()).clone()
            if latent_cond is not None:
                latent_res["noisy_latents"][:, :, 0:1] = latent_cond[:, :, 0:1]
                latent_res["timesteps"][:, 0:1] *= 0
            input_dict["latent_res_lst"] = latent_res

        if action_model_input is not None:
            action_res = {
                "noisy_latents": action_model_input,
                "timesteps": torch.ones([1, action_model_input.shape[2]], dtype=torch.float32, device=AI_DEVICE) * action_t,
                "grid_id": self.model.build_grid_id(
                    action_model_input.shape[-3],
                    action_model_input.shape[-2],
                    action_model_input.shape[-1],
                    1,
                    1,
                    frame_st_id,
                    action=True,
                )[None].to(AI_DEVICE),
                "text_emb": self.prompt_embeds.to(GET_DTYPE()).clone(),
            }
            if self.use_action_cfg:
                action_res["negative_text_emb"] = self.negative_prompt_embeds.to(GET_DTYPE()).clone()
            if action_cond is not None:
                action_res["noisy_latents"][:, :, 0:1] = action_cond[:, :, 0:1]
                action_res["timesteps"][:, 0:1] *= 0
            action_res["noisy_latents"][:, ~self.action_mask] *= 0
            input_dict["action_res_lst"] = action_res
        return input_dict

    def _build_video_step_inputs(self, scheduler):
        latent_cond = scheduler.cond_latent if scheduler.cond_latent is not None else None
        input_dict = self._prepare_model_input(
            scheduler.latents,
            None,
            latent_t=scheduler.current_timestep,
            action_t=scheduler.current_timestep,
            latent_cond=latent_cond,
            action_cond=None,
            frame_st_id=self.frame_st_id,
        )
        model_inputs = input_dict["latent_res_lst"]
        model_inputs.update(
            {
                "action_mode": False,
                "update_cache": 1 if scheduler.last_step else 0,
                "cache_name": self._cache_name_for_cfg_mode(self.use_cfg),
                "enable_cfg": self.use_cfg,
                "guide_scale": self.config["sample_guide_scale"],
            }
        )
        scheduler.step_latents = (not scheduler.last_step) or self.config["video_exec_step"] != -1
        return model_inputs

    def _build_action_step_inputs(self, scheduler):
        action_cond = scheduler.cond_latent if scheduler.cond_latent is not None else None
        input_dict = self._prepare_model_input(
            None,
            scheduler.latents,
            latent_t=scheduler.current_timestep,
            action_t=scheduler.current_timestep,
            latent_cond=None,
            action_cond=action_cond,
            frame_st_id=self.frame_st_id,
        )
        model_inputs = input_dict["action_res_lst"]
        model_inputs.update(
            {
                "action_mode": True,
                "update_cache": 1 if scheduler.last_step else 0,
                "cache_name": self._cache_name_for_cfg_mode(self.use_action_cfg),
                "enable_cfg": self.use_action_cfg,
                "guide_scale": self.config["action_sample_guide_scale"],
            }
        )
        scheduler.step_latents = not scheduler.last_step
        return model_inputs

    def _postprocess_video_noise_pred(self, noise_pred):
        return self.scheduler.seq_to_patch(
            tuple(self.config["patch_size"]),
            noise_pred,
            self.num_frame_per_chunk,
            self.latent_height,
            self.latent_width,
            batch_size=1,
        )

    def _postprocess_action_noise_pred(self, noise_pred):
        noise_pred = noise_pred.reshape(
            noise_pred.shape[0],
            self.num_frame_per_chunk,
            self.num_action_per_frame,
            -1,
        )
        return noise_pred.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()

    def _postprocess_action(self, action):
        action = action.detach().float().cpu()[0, ..., 0]
        if self.config["action_norm_method"] != "quantiles":
            raise NotImplementedError(f"Unsupported action_norm_method: {self.config['action_norm_method']}")
        q01 = self.actions_q01.cpu()
        q99 = self.actions_q99.cpu()
        action = (action + 1) / 2 * (q99 - q01 + 1e-6) + q01
        action_np = action.squeeze(0).numpy()
        return action_np[self.config["used_action_channel_ids"]]

    def _run_scheduler_loop(self, scheduler):
        for step_index in range(scheduler.infer_steps):
            with ProfilingContext4DebugL1("step_pre"):
                scheduler.step_pre(step_index=step_index)
            with ProfilingContext4DebugL1("🚀 infer_main"):
                self.model.infer(self.inputs)
            with ProfilingContext4DebugL1("step_post"):
                scheduler.step_post()

    def run_segment(self, segment_idx=0):
        num_frame_per_chunk = self.num_frame_per_chunk
        init_latent = None
        if self.frame_st_id == 0:
            init_latent = self._encode_obs(self.init_obs)
            self.init_latent = init_latent

        latent_shape = (
            1,
            48,
            num_frame_per_chunk,
            self.latent_height,
            self.latent_width,
        )
        action_shape = (
            1,
            self.config["action_dim"],
            num_frame_per_chunk,
            self.num_action_per_frame,
            1,
        )

        latent_cond = init_latent[:, :, 0:1].to(GET_DTYPE()) if self.frame_st_id == 0 else None
        self.scheduler.prepare_loop(
            infer_steps=self.config["infer_steps"],
            device=AI_DEVICE,
            latent_shape=latent_shape,
            seed=self.input_info.seed,
            dtype=GET_DTYPE(),
            cond_latent=latent_cond,
            video_exec_step=self.config["video_exec_step"],
        )
        self.scheduler.bind_step_inputs(self.inputs, self._build_video_step_inputs)
        self.scheduler.bind_noise_pred_processor(self._postprocess_video_noise_pred)
        self.model.set_scheduler(self.scheduler)
        self._run_scheduler_loop(self.scheduler)
        latents = self.scheduler.latents

        action_cond = torch.zeros([1, self.config["action_dim"], 1, self.num_action_per_frame, 1], device=AI_DEVICE, dtype=GET_DTYPE()) if self.frame_st_id == 0 else None
        self.action_scheduler.generator = self.scheduler.generator
        self.action_scheduler.prepare_loop(
            infer_steps=self.config["action_infer_steps"],
            device=AI_DEVICE,
            latent_shape=action_shape,
            seed=self.input_info.seed,
            dtype=GET_DTYPE(),
            cond_latent=action_cond,
        )
        self.action_scheduler.bind_step_inputs(self.inputs, self._build_action_step_inputs)
        self.action_scheduler.bind_noise_pred_processor(self._postprocess_action_noise_pred)
        self.model.set_scheduler(self.action_scheduler)
        self._run_scheduler_loop(self.action_scheduler)
        actions = self.action_scheduler.latents

        actions[:, ~self.action_mask] *= 0
        self.model.set_scheduler(self.scheduler)
        return self._postprocess_action(actions), latents

    def _load_init_obs(self):
        image_path = getattr(self.input_info, "image_path", "")
        if not image_path:
            raise ValueError("LingBot-VA requires image_path from input_info.")
        image_path = os.path.expanduser(str(image_path))
        cam_keys = self.config["obs_cam_keys"]
        if os.path.isdir(image_path):
            image_files = [os.path.join(image_path, f"{key}.png") for key in cam_keys]
        else:
            image_files = [item.strip() for item in image_path.split(",") if item.strip()]
            if len(image_files) != len(cam_keys):
                raise ValueError(f"Expected {len(cam_keys)} camera images, got {len(image_files)} from image_path={image_path}")
        obs = {}
        for key, file_path in zip(cam_keys, image_files):
            obs[key] = np.array(Image.open(file_path).convert("RGB"))
        return {"obs": [obs]}

    def run_vae_decoder(self, pred_latent):
        decoded = self.vae_decoder.decode(pred_latent.squeeze(0).to(GET_DTYPE()))
        return decoded[0].permute(1, 2, 3, 0).add(1.0).mul(0.5).clamp(0, 1)

    def end_run_segment(self, segment_idx=None):
        pass

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        self.init_run()
        for chunk_id in range(self.num_chunks):
            logger.info(f"LingBot-VA chunk {chunk_id + 1}/{self.num_chunks}")
            with ProfilingContext4DebugL1(f"chunk end2end {chunk_id + 1}/{self.num_chunks}"):
                self.check_stop()
                self.init_run_segment(chunk_id)
                actions, latents = self.run_segment(chunk_id)
                self.pred_latent_lst.append(latents)
                self.pred_action_lst.append(torch.from_numpy(actions))
                self.end_run_segment(chunk_id)
            if self.progress_callback:
                self.progress_callback((chunk_id + 1) / self.num_chunks * 100, 100)

        pred_latent = torch.cat(self.pred_latent_lst, dim=2)
        self.pred_action = torch.cat(self.pred_action_lst, dim=1)
        self.gen_video = self.run_vae_decoder(pred_latent)
        gen_video_final = self.process_images_after_vae_decoder()
        self.end_run()
        return gen_video_final

    def process_images_after_vae_decoder(self):
        self.gen_video_final = self.gen_video
        video_path = getattr(self.input_info, "save_result_path", None)
        if not video_path:
            raise ValueError("LingBot-VA requires save_result_path from input_info.")
        video_path = str(video_path)
        action_path = str(Path(video_path).with_suffix(".actions.npy"))
        save_to_video(self.gen_video_final, video_path, fps=self.config.get("target_fps", 10), method=self.config.get("save_video_method", "imageio"))
        os.makedirs(os.path.dirname(action_path) or ".", exist_ok=True)
        np.save(action_path, self.pred_action.flatten(1).numpy())
        logger.info("Saved LingBot-VA video to {}", video_path)
        logger.info("Saved LingBot-VA actions to {}", action_path)
        if self.input_info.return_result_tensor:
            return {"video": self.gen_video_final, "actions": self.pred_action}
        return {"video": None}

    def end_run(self):
        self.model.clear_cache(self.cache_name)
        if self.scheduler is not None:
            self.scheduler.clear()
        if self.action_scheduler is not None:
            self.action_scheduler.clear()
        if hasattr(self, "inputs"):
            del self.inputs
        self.input_info = None
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            if hasattr(self.model.transformer_infer, "offload_manager"):
                del self.model.transformer_infer.offload_manager
            del self.model
        torch_device_module.empty_cache()
        gc.collect()

    def _run_pipeline_local(self):
        if self.config["use_prompt_enhancer"]:
            self.input_info.prompt_enhanced = self.post_prompt_enhancer()
        self.inputs = self.run_input_encoder()
        return self.run_main()

    @ProfilingContext4DebugL1("RUN pipeline", recorder_mode=GET_RECORDER_MODE(), metrics_func=monitor_cli.lightx2v_worker_request_duration, metrics_labels=["LingbotVARunner"])
    @torch.no_grad()
    def run_pipeline(self, input_info):
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_count.inc()
        self.input_info = input_info
        gen_video_final = self._run_pipeline_local()
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        return gen_video_final
