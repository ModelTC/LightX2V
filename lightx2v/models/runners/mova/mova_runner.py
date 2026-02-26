import os
import json
import torch
import torch.distributed as dist
import tempfile
import numpy as np
from PIL import Image
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE
import torchvision.transforms as transforms

# 导入所需组件
from diffusers.models.autoencoders import AutoencoderKLWan
from lightx2v.models.audio_encoders.dac_vae import DAC  # 修正：导入 DAC 而非 DACVAE
from lightx2v.models.networks.mova.wan_video_dit import WanModel, sinusoidal_embedding_1d
from lightx2v.models.networks.mova.wan_audio_dit import WanAudioModel
from lightx2v.models.networks.mova.interactionv2 import DualTowerConditionalBridge
from lightx2v.models.schedulers.mova.flow_match_pair import FlowMatchPairScheduler
from safetensors.torch import load_file
from transformers import T5EncoderModel, AutoTokenizer


# 辅助函数：合并分片权重
def load_safetensors_sharded(folder):
    state_dict = {}
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.safetensors'):
            part = load_file(os.path.join(folder, fname), device='cpu')
            state_dict.update(part)
    return state_dict


@RUNNER_REGISTER("mova")
class MOVARunner(DefaultRunner):
    def _run_input_encoder_local_i2av(self):
        return {}

    def __init__(self, config):
        super().__init__(config)
        self.config["task"] = "i2av"
        self.num_frames = self.config.get("num_frames", 193)
        self.height = self.config.get("height", 352)
        self.width = self.config.get("width", 640)
        self.video_fps = self.config.get("video_fps", 24.0)
        self.num_inference_steps = self.config.get("num_inference_steps", 10)
        self.cfg_scale = self.config.get("cfg_scale", 5.0)
        self.sigma_shift = self.config.get("sigma_shift", 5.0)
        self._use_manual = True
        self._dist_initialized = False

    def _init_distributed(self):
        if self._dist_initialized:
            return True
        if dist.is_initialized():
            print("Distributed already initialized externally.")
            self._dist_initialized = True
            return True
        try:
            if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
                dist.init_process_group(backend="nccl", init_method="env://")
                print("Initialized distributed from environment (torchrun).")
                self._dist_initialized = True
                return True
            else:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(local_rank)
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                init_method = f"file://{temp_file.name}"
                temp_file.close()
                dist.init_process_group(
                    backend="nccl",
                    init_method=init_method,
                    rank=0,
                    world_size=1
                )
                print("Initialized single-process distributed group with file init_method.")
                self._dist_initialized = True
                return True
        except Exception as e:
            print(f"Failed to initialize distributed group: {e}")
            return False

    def load_model(self):
        """加载所有 MOVA 组件，不依赖外部 pipeline"""
        ckpt_path = self.config["model_path"]
        torch_dtype = GET_DTYPE()
        device = AI_DEVICE

        # 1. 视频 VAE
        print("Loading video VAE...")
        self.video_vae = AutoencoderKLWan.from_pretrained(
            os.path.join(ckpt_path, "video_vae"),
            torch_dtype=torch_dtype
        ).to(device)
        # 获取标准化参数
        self.vae_mean = torch.tensor(self.video_vae.config.latents_mean, device=device, dtype=torch_dtype).view(1,16,1,1,1)
        self.vae_std = torch.tensor(self.video_vae.config.latents_std, device=device, dtype=torch_dtype).view(1,16,1,1,1)

        # 2. 音频 VAE
        print("Loading audio VAE...")
        self.audio_vae = DAC.from_pretrained(
            os.path.join(ckpt_path, "audio_vae"),
            torch_dtype=torch_dtype
        ).to(device)

        # 3. 视频 DiT
        print("Loading video DiT...")
        video_dit_path = os.path.join(ckpt_path, "video_dit")
        with open(os.path.join(video_dit_path, "config.json")) as f:
            video_config = json.load(f)
        self.video_dit = WanModel(**video_config).to(device=device, dtype=torch_dtype)
        # 加载权重（可能分片）
        video_state_dict = load_safetensors_sharded(video_dit_path)
        self.video_dit.load_state_dict(video_state_dict, strict=False)
        print("Video DiT loaded")

        # 4. 音频 DiT
        print("Loading audio DiT...")
        audio_dit_path = os.path.join(ckpt_path, "audio_dit")
        with open(os.path.join(audio_dit_path, "config.json")) as f:
            audio_config = json.load(f)
        self.audio_dit = WanAudioModel(**audio_config).to(device=device, dtype=torch_dtype)
        audio_state_dict = load_safetensors_sharded(audio_dit_path)
        self.audio_dit.load_state_dict(audio_state_dict, strict=False)
        print("Audio DiT loaded")

        # 5. 桥接模块
        print("Loading bridge...")
        bridge_path = os.path.join(ckpt_path, "dual_tower_bridge")
        with open(os.path.join(bridge_path, "config.json")) as f:
            bridge_config = json.load(f)
        self.bridge = DualTowerConditionalBridge(**bridge_config).to(device=device, dtype=torch_dtype)
        bridge_state_dict = load_safetensors_sharded(bridge_path)
        self.bridge.load_state_dict(bridge_state_dict, strict=False)
        print("Bridge loaded")

        # 6. 文本编码器和 tokenizer
        print("Loading text encoder and tokenizer...")
        self.text_encoder = T5EncoderModel.from_pretrained(
            os.path.join(ckpt_path, "text_encoder"),
            torch_dtype=torch_dtype
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(ckpt_path, "tokenizer"),
            use_fast=True
        )

        # 7. 调度器
        self.scheduler = FlowMatchPairScheduler(
            num_inference_steps=self.num_inference_steps,
            shift=self.sigma_shift,
        )

        # 初始化分布式
        self._init_distributed()
        print("All components loaded successfully")

    # 辅助函数：视频 latent 标准化/反标准化
    def normalize_video_latents(self, latents):
        return (latents - self.vae_mean) * (1.0 / self.vae_std)

    def denormalize_video_latents(self, latents):
        return latents * self.vae_std + self.vae_mean

    # 复制的 forward_dual_tower_dit 方法（来自官方 pipeline，稍作调整）
    def _forward_dual_tower_dit(
        self,
        visual_dit,
        visual_x,
        audio_x,
        visual_context,
        audio_context,
        visual_t_mod,
        audio_t_mod,
        visual_freqs,
        audio_freqs,
        grid_size,
        video_fps,
        condition_scale=1.0,
        a2v_condition_scale=None,
        v2a_condition_scale=None,
        cp_mesh=None,
    ):
        min_layers = min(len(visual_dit.blocks), len(self.audio_dit.blocks))
        visual_layers = len(visual_dit.blocks)

        sp_enabled = False
        sp_group = None
        sp_rank = 0
        sp_size = 1
        visual_pad_len = 0
        audio_pad_len = 0

        if self.bridge.apply_cross_rope:
            (visual_rope_cos_sin, audio_rope_cos_sin) = self.bridge.build_aligned_freqs(
                video_fps=video_fps,
                grid_size=grid_size,
                audio_steps=audio_x.shape[1],
                device=visual_x.device,
                dtype=visual_x.dtype,
            )
        else:
            visual_rope_cos_sin = None
            audio_rope_cos_sin = None

        if cp_mesh is not None:
            # 此处简化，省略分布式处理，若需要可自行添加
            pass

        for layer_idx in range(min_layers):
            visual_block = visual_dit.blocks[layer_idx]
            audio_block = self.audio_dit.blocks[layer_idx]

            if self.bridge.should_interact(layer_idx, 'a2v'):
                visual_x, audio_x = self.bridge(
                    layer_idx,
                    visual_x,
                    audio_x,
                    x_freqs=visual_rope_cos_sin,
                    y_freqs=audio_rope_cos_sin,
                    a2v_condition_scale=a2v_condition_scale,
                    v2a_condition_scale=v2a_condition_scale,
                    condition_scale=condition_scale,
                    video_grid_size=grid_size,
                )

            visual_x = visual_block(visual_x, visual_context, visual_t_mod, visual_freqs)
            audio_x = audio_block(audio_x, audio_context, audio_t_mod, audio_freqs)

        for layer_idx in range(min_layers, visual_layers):
            visual_block = visual_dit.blocks[layer_idx]
            visual_x = visual_block(visual_x, visual_context, visual_t_mod, visual_freqs)

        if sp_enabled:
            # 省略分布式合并
            pass

        return visual_x, audio_x

    # 手动推理主函数
    def _run_pipeline_manual(self, prompt, negative_prompt, image, seed,
                              height, width, num_frames, video_fps,
                              num_inference_steps, cfg_scale, sigma_shift):
        device = AI_DEVICE
        dtype = GET_DTYPE()

        # ========== 1. 图像预处理（确保 img_tensor_5d 被定义） ==========
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(image).unsqueeze(0).to(device, dtype=dtype)
        img_tensor_5d = img_tensor.unsqueeze(2)  # [1, 3, 1, height, width]

        # ========== 2. 构造完整的 video_condition 并编码 ==========
        video_condition = torch.cat([
            img_tensor_5d,  # 第一帧
            torch.zeros(1, 3, num_frames - 1, height, width, device=device, dtype=dtype)
        ], dim=2)  # [1, 3, num_frames, height, width]

        with torch.no_grad():
            latent_dist = self.video_vae.encode(video_condition).latent_dist
            latent_condition = latent_dist.mode()  # [1, 16, num_latent_frames, latent_H, latent_W]
            print(f"[Manual] Encoded video_condition latent shape: {latent_condition.shape}")

        # 归一化
        latent_condition = self.normalize_video_latents(latent_condition)
        print(f"[Manual] Normalized condition latent: mean={latent_condition.mean():.3f}, std={latent_condition.std():.3f}")

        # ========== 3. 构造 mask（与之前相同） ==========
        vae_scale_factor_temporal = 4
        vae_scale_factor_spatial = 8
        num_latent_frames = latent_condition.shape[2]
        latent_height = latent_condition.shape[3]
        latent_width = latent_condition.shape[4]

        mask_lat_size = torch.ones(1, 1, num_frames, latent_height, latent_width, device=device)
        mask_lat_size[:, :, 1:, :, :] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1, :, :]
        first_frame_mask = first_frame_mask.repeat(1, 1, vae_scale_factor_temporal, 1, 1)
        mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:, :, :]], dim=2)
        mask_lat_size = mask_lat_size.view(1, -1, vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.permute(0, 2, 1, 3, 4).contiguous()  # [1, 4, num_latent_frames, latent_H, latent_W]

        # ========== 4. 拼接 condition ==========
        condition = torch.cat([mask_lat_size, latent_condition], dim=1)  # [1, 20, num_latent_frames, latent_H, latent_W]
        print(f"[Manual] Condition shape: {condition.shape}, mean={condition.mean():.3f}, std={condition.std():.3f}")

        # ========== 3. 调度器初始化 ==========
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas
        print(f"[Manual] Timesteps shape: {timesteps.shape}")
        print(f"[Manual] First 5 timesteps: {timesteps[:5].cpu().numpy()}")

        # ========== 4. 文本编码 ==========
        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=512,
                                      truncation=True, return_tensors="pt").to(device)
        prompt_embeds = self.text_encoder(**text_inputs).last_hidden_state
        print(f"[Manual] Prompt embeds shape: {prompt_embeds.shape}")

        neg_text_inputs = self.tokenizer(negative_prompt, padding="max_length", max_length=512,
                                         truncation=True, return_tensors="pt").to(device)
        negative_prompt_embeds = self.text_encoder(**neg_text_inputs).last_hidden_state

        # ========== 5. 初始化 latents ==========
        latents = torch.randn(1, 16, num_latent_frames, latent_height, latent_width,
                              device=device, dtype=dtype)
        print(f"[Manual] Initial latents mean={latents.mean():.3f}, std={latents.std():.3f}")

        # ========== 6. 音频 latent ==========
        audio_latent_dim = self.audio_vae.latent_dim
        # 计算音频 latent 长度（根据视频时长、采样率和 VAE 下采样倍数）
        audio_len = int((num_frames - 1) / video_fps * self.audio_vae.sample_rate / self.audio_vae.hop_length)
        audio_latents = torch.randn(1, audio_latent_dim, audio_len, device=device, dtype=dtype) 
        print(f"[Manual] Audio latents shape: {audio_latents.shape}")

        # ========== 7. 去噪循环 ==========
        for i, t in enumerate(timesteps):
            t_tensor = t.unsqueeze(0).to(device=device, dtype=dtype)

            # 时间嵌入
            t_emb_v = sinusoidal_embedding_1d(self.video_dit.freq_dim, t_tensor)
            visual_t = self.video_dit.time_embedding(t_emb_v)
            visual_t_mod = self.video_dit.time_projection(visual_t).unflatten(1, (6, self.video_dit.dim))

            t_emb_a = sinusoidal_embedding_1d(self.audio_dit.freq_dim, t_tensor)
            audio_t = self.audio_dit.time_embedding(t_emb_a)
            audio_t_mod = self.audio_dit.time_projection(audio_t).unflatten(1, (6, self.audio_dit.dim))

            # 文本嵌入
            visual_context_emb = self.video_dit.text_embedding(prompt_embeds)
            audio_context_emb = self.audio_dit.text_embedding(prompt_embeds)
            neg_visual_context_emb = self.video_dit.text_embedding(negative_prompt_embeds)
            neg_audio_context_emb = self.audio_dit.text_embedding(negative_prompt_embeds)

            # 拼接视频输入
            video_input = torch.cat([latents, condition], dim=1)
            video_input = video_input.to(dtype=dtype)

            # patchify
            visual_x, (t_len, h_len, w_len) = self.video_dit.patchify(video_input)
            audio_x, (a_len,) = self.audio_dit.patchify(audio_latents)

            # 构建视频频率
            visual_freqs = torch.cat([
                self.video_dit.freqs[0][:t_len].view(t_len, 1, 1, -1).expand(t_len, h_len, w_len, -1),
                self.video_dit.freqs[1][:h_len].view(1, h_len, 1, -1).expand(t_len, h_len, w_len, -1),
                self.video_dit.freqs[2][:w_len].view(1, 1, w_len, -1).expand(t_len, h_len, w_len, -1)
            ], dim=-1).reshape(t_len * h_len * w_len, 1, -1).to(device=device, dtype=dtype)

            # 构建音频频率
            audio_freqs = torch.cat([
                self.audio_dit.freqs[0][:a_len].view(a_len, -1).expand(a_len, -1),
                self.audio_dit.freqs[1][:a_len].view(a_len, -1).expand(a_len, -1),
                self.audio_dit.freqs[2][:a_len].view(a_len, -1).expand(a_len, -1)
            ], dim=-1).reshape(a_len, 1, -1).to(device=device, dtype=dtype)

            # 计算对齐频率（用于桥接）
            if self.bridge.apply_cross_rope:
                (visual_rope_cos_sin, audio_rope_cos_sin) = self.bridge.build_aligned_freqs(
                    video_fps=video_fps,
                    grid_size=(t_len, h_len, w_len),
                    audio_steps=audio_x.shape[1],
                    device=visual_x.device,
                    dtype=visual_x.dtype,
                )
            else:
                visual_rope_cos_sin = None
                audio_rope_cos_sin = None

            # ---------- 正分支 ----------
            with torch.no_grad():
                v_pos, a_pos = self._forward_dual_tower_dit(
                    visual_dit=self.video_dit,
                    visual_x=visual_x,
                    audio_x=audio_x,
                    visual_context=visual_context_emb,
                    audio_context=audio_context_emb,
                    visual_t_mod=visual_t_mod,
                    audio_t_mod=audio_t_mod,
                    visual_freqs=visual_freqs,
                    audio_freqs=audio_freqs,
                    grid_size=(t_len, h_len, w_len),
                    video_fps=video_fps,
                    condition_scale=1.0,
                    a2v_condition_scale=None,
                    v2a_condition_scale=None,
                    cp_mesh=None,
                )
                v_pos_head = self.video_dit.head(v_pos, visual_t)
                a_pos_head = self.audio_dit.head(a_pos, audio_t)
                v_pos_unpatch = self.video_dit.unpatchify(v_pos_head, (t_len, h_len, w_len))
                a_pos_unpatch = self.audio_dit.unpatchify(a_pos_head, (a_len,))
                del v_pos, a_pos, v_pos_head, a_pos_head

            # ---------- 负分支 ----------
            with torch.no_grad():
                v_neg, a_neg = self._forward_dual_tower_dit(
                    visual_dit=self.video_dit,
                    visual_x=visual_x,
                    audio_x=audio_x,
                    visual_context=neg_visual_context_emb,
                    audio_context=neg_audio_context_emb,
                    visual_t_mod=visual_t_mod,
                    audio_t_mod=audio_t_mod,
                    visual_freqs=visual_freqs,
                    audio_freqs=audio_freqs,
                    grid_size=(t_len, h_len, w_len),
                    video_fps=video_fps,
                    condition_scale=1.0,
                    a2v_condition_scale=None,
                    v2a_condition_scale=None,
                    cp_mesh=None,
                )
                v_neg_head = self.video_dit.head(v_neg, visual_t)
                a_neg_head = self.audio_dit.head(a_neg, audio_t)
                v_neg_unpatch = self.video_dit.unpatchify(v_neg_head, (t_len, h_len, w_len))
                a_neg_unpatch = self.audio_dit.unpatchify(a_neg_head, (a_len,))
                del v_neg, a_neg, v_neg_head, a_neg_head

            # CFG组合
            v_pred = v_neg_unpatch + cfg_scale * (v_pos_unpatch - v_neg_unpatch)
            a_pred = a_neg_unpatch + cfg_scale * (a_pos_unpatch - a_neg_unpatch)

            # 更新
            latents = self.scheduler.step(v_pred, t, latents)
            audio_latents = self.scheduler.step(a_pred, t, audio_latents)

            if i % 10 == 0:
                print(f"Step {i}: latents mean={latents.mean():.3f}, std={latents.std():.3f}")

            if i % 5 == 0:
                torch.cuda.empty_cache()

        # ========== 8. 解码视频 ==========
        with torch.no_grad():
            video_latents = self.denormalize_video_latents(latents)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video = self.video_vae.decode(video_latents).sample
            video = video.float().cpu()
            print(f"Video shape: {video.shape}, min={video.min():.3f}, max={video.max():.3f}")

            video = (video + 1) / 2
            video = video.clamp(0, 1)
            video_frames = []
            for i in range(video.shape[2]):
                frame = video[0, :, i, :, :].permute(1, 2, 0).numpy()
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
                video_frames.append(Image.fromarray(frame))
            print(f"Converted {len(video_frames)} frames manually")

        # ========== 9. 解码音频 ==========
        with torch.no_grad():
            audio = self.audio_vae.decode(audio_latents)  # [1, 1, T]
            audio = audio.cpu().squeeze(0)  # [1, T]
            audio = audio.float().unsqueeze(0)  # [1, 1, T]

        return video_frames, audio

    # run_pipeline 入口，只保留手动分支
    def run_pipeline(self, input_info):
        if not self._dist_initialized:
            if not self._init_distributed():
                os.environ.setdefault("MASTER_ADDR", "localhost")
                os.environ.setdefault("MASTER_PORT", "29500")
                try:
                    dist.init_process_group(
                        backend="nccl",
                        init_method="env://",
                        rank=0,
                        world_size=1
                    )
                    print("Initialized distributed with env:// fallback.")
                    self._dist_initialized = True
                except Exception as e:
                    raise RuntimeError("Failed to initialize distributed.") from e

        prompt = input_info.prompt
        negative_prompt = input_info.negative_prompt or ""
        image_path = input_info.image_path
        save_path = input_info.save_result_path
        seed = input_info.seed

        if not image_path or not save_path:
            raise ValueError("image_path and save_result_path must be provided.")

        # 图像预处理
        from mova.datasets.transforms.custom import crop_and_resize
        img = Image.open(image_path).convert("RGB")
        img = crop_and_resize(img, self.height, self.width)

        # 调用手动推理
        video_frames, audio = self._run_pipeline_manual(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img,
            seed=seed,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            video_fps=self.video_fps,
            num_inference_steps=self.num_inference_steps,
            cfg_scale=self.cfg_scale,
            sigma_shift=self.sigma_shift,
        )

        # 保存视频
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        from mova.utils.data import save_video_with_audio
        save_video_with_audio(
            video_frames,
            audio.cpu().squeeze(),
            save_path,
            fps=self.video_fps,
            sample_rate=self.audio_vae.sample_rate,
            quality=9,
        )
        print(f"Video saved to {save_path}")

        self.gen_video_final = torch.zeros(1)
        return self.gen_video_final

    # 以下为占位方法，满足父类要求
    def run_text_encoder(self, input_info):
        pass

    def run_vae_encoder(self, img=None):
        pass

    def process_images_after_vae_decoder(self):
        pass

    def init_run(self):
        pass