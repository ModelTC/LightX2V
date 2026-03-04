import os
import json
import tempfile
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE
import torchvision.transforms as transforms
from lightx2v.models.networks.mova.video_model import MOVACompatVideoDiT
from diffusers.models.autoencoders import AutoencoderKLWan
from lightx2v.models.audio_encoders.mova_dac import MOVAAudioVAE
from lightx2v.models.networks.wan.model import WanModel as LightX2VWanModel
from lightx2v.models.networks.mova.audio_model import MOVAAudioDiT
from lightx2v.models.networks.mova.bridge import MOVADualTowerBridge
from lightx2v.models.schedulers.mova.flow_match_pair import FlowMatchPairScheduler
from safetensors.torch import load_file
from transformers import T5EncoderModel, AutoTokenizer
import torch.nn.functional as F
def load_safetensors_sharded(folder):
    state_dict = {}
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.safetensors'):
            part = load_file(os.path.join(folder, fname), device='cpu')
            state_dict.update(part)
    return state_dict

# ========== 视频 DiT 子类：完全手动加载 ==========
class LightX2VWanModelForMOVA(LightX2VWanModel):
    def __init__(self, model_path, config, device, mova_state_dict, **kwargs):
        self.mova_state_dict = mova_state_dict
        self.target_device = device
        self.lazy_load_path = None
        super().__init__(model_path, config, device, **kwargs)

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        # 返回空字典，完全跳过自动加载
        return {}

    def _init_weights(self, weight_dict=None):
        # 只创建权重容器，不加载任何权重
        self.pre_weight = self.pre_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config)
        if hasattr(self, "post_weight_class") and self.post_weight_class is not None:
            self.post_weight = self.post_weight_class(self.config)

    def load_mova_weights(self):
        """手动将 MOVA 权重加载到 LightX2V 的内部容器中（保持张量在 CPU）"""
        # 键名映射：移除可能的前缀
        mapped_dict = {}
        for key, tensor in self.mova_state_dict.items():
            if key.startswith("model.diffusion_model."):
                new_key = key[len("model.diffusion_model."):]
            else:
                new_key = key
            # 确保张量在 CPU 上（它们应该已经在 CPU 上）
            mapped_dict[new_key] = tensor.cpu() if tensor.device.type != 'cpu' else tensor

        # 加载到各权重容器（此时张量仍在 CPU，会创建 pin_tensor）
        self.pre_weight.load(mapped_dict)
        self.transformer_weights.load(mapped_dict)
        if hasattr(self, "post_weight"):
            self.post_weight.load(mapped_dict)
        print("Video DiT: MOVA weights manually loaded into LightX2V WanModel (CPU).")

# ========== 音频 DiT 子类同理 ==========
class LightX2VAudioDiTForMOVA(MOVAAudioDiT):
    def __init__(self, model_path, config, device, mova_state_dict, **kwargs):
        self.mova_state_dict = mova_state_dict
        self.target_device = device
        self.lazy_load_path = None
        super().__init__(model_path, config, device, **kwargs)

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        return {}

    def _init_weights(self, weight_dict=None):
        self.pre_weight = self.pre_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config)
        if hasattr(self, "post_weight_class") and self.post_weight_class is not None:
            self.post_weight = self.post_weight_class(self.config)

    def load_mova_weights(self):
        mapped_dict = {}
        for key, tensor in self.mova_state_dict.items():
            if key.startswith("model.diffusion_model."):
                new_key = key[len("model.diffusion_model."):]
            else:
                new_key = key
            mapped_dict[new_key] = tensor.cpu() if tensor.device.type != 'cpu' else tensor
        self.pre_weight.load(mapped_dict)
        self.transformer_weights.load(mapped_dict)
        if hasattr(self, "post_weight"):
            self.post_weight.load(mapped_dict)
        print("Audio DiT: MOVA weights manually loaded into LightX2V WanModel (CPU).")
def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)
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
        ckpt_path = self.config["model_path"]
        torch_dtype = GET_DTYPE()
        device = torch.device(AI_DEVICE)

        # 1. 视频 VAE
        print("Loading video VAE...")
        self.video_vae = AutoencoderKLWan.from_pretrained(
            os.path.join(ckpt_path, "video_vae"),
            torch_dtype=torch_dtype
        ).to(device)
        self.vae_mean = torch.tensor(self.video_vae.config.latents_mean, device=device, dtype=torch_dtype).view(1,16,1,1,1)
        self.vae_std = torch.tensor(self.video_vae.config.latents_std, device=device, dtype=torch_dtype).view(1,16,1,1,1)
        print(f"[Load] vae_mean shape: {self.vae_mean.shape}, mean: {self.vae_mean.mean().item():.3f}")
        print(f"[Load] vae_std shape: {self.vae_std.shape}, mean: {self.vae_std.mean().item():.3f}")

        # 2. 音频 VAE
        print("Loading audio VAE...")
        audio_vae_path = os.path.join(ckpt_path, "audio_vae")
        with open(os.path.join(audio_vae_path, "config.json")) as f:
            audio_vae_config = json.load(f)
        self.audio_vae = MOVAAudioVAE(**audio_vae_config)
        weight_file = os.path.join(audio_vae_path, "diffusion_pytorch_model.safetensors")
        if os.path.exists(weight_file):
            state_dict = load_file(weight_file, device='cpu')
            self.audio_vae.load_state_dict(state_dict, strict=False)
        else:
            state_dict = load_safetensors_sharded(audio_vae_path)
            self.audio_vae.load_state_dict(state_dict, strict=False)
        self.audio_vae = self.audio_vae.to(device, dtype=torch_dtype)
        print(f"[Load] audio_vae latent_dim: {self.audio_vae.latent_dim}, hop_length: {self.audio_vae.hop_length}, sample_rate: {self.audio_vae.sample_rate}")

        # ========== 视频 DiT ==========
        print("Loading video DiT...")
        video_dit_path = os.path.join(ckpt_path, "video_dit")
        with open(os.path.join(video_dit_path, "config.json")) as f:
            video_config = json.load(f)

        default_video_config = {
            "seq_parallel": False,
            "feature_caching": "NoCaching",
            "cpu_offload": False,
            "offload_granularity": "model",
            "model_cls": "wan2.1",
            "task": "t2v",
            "self_attn_1_type": "flash_attn2",
            "self_attn_2_type": "flash_attn2",
            "cross_attn_1_type": "flash_attn2",
            "cross_attn_2_type": "flash_attn2",
            "attn_mode": "flash_attn2",
            "rope_type": "torch",   # 新增
            "lazy_load": False,
        }
        merged_video_config = {**default_video_config, **video_config}

        video_sd = load_safetensors_sharded(video_dit_path)
        self.video_dit = MOVACompatVideoDiT(
            model_path=video_dit_path,
            config=merged_video_config,
            device=device,
            mova_state_dict=video_sd,   # 添加这一行
            model_type="wan2.1"
        )

        # 从 video_dit_path 加载权重
        video_sd = load_safetensors_sharded(video_dit_path)
        self.video_dit.load_mova_weights(video_sd)

        # 移动到 GPU
        if device.type == "cuda":
            self.video_dit.to_cuda()
        else:
            self.video_dit.to_cpu()
        print("Video DiT loaded.")

        # ========== 音频 DiT ==========
        print("Loading audio DiT...")
        audio_dit_path = os.path.join(ckpt_path, "audio_dit")
        with open(os.path.join(audio_dit_path, "config.json")) as f:
            audio_config = json.load(f)

        default_audio_config = {
            "seq_parallel": False,
            "feature_caching": "NoCaching",
            "cpu_offload": False,
            "offload_granularity": "model",
            "model_cls": "wan2.1",
            "task": "t2v",
            "self_attn_1_type": "flash_attn2",
            "self_attn_2_type": "flash_attn2",
            "cross_attn_1_type": "flash_attn2",
            "cross_attn_2_type": "flash_attn2",
            "attn_mode": "flash_attn2",
            "rope_type": "torch",
        }
        merged_audio_config = {**default_audio_config, **audio_config}

        audio_sd = load_safetensors_sharded(audio_dit_path)

        self.audio_dit = MOVAAudioDiT(
            model_path=audio_dit_path,
            config=merged_audio_config,
            device=device,
            mova_state_dict=audio_sd,
        )

        if device.type == "cuda":
            self.audio_dit.to_cuda()
        else:
            self.audio_dit.to_cpu()
        print("Audio DiT loaded.")

        # ----- 新增：手动替换音频推理引擎为 MOVATransformerInfer -----
        from lightx2v.models.networks.mova.transformer_infer import MOVATransformerInfer
        if not hasattr(self.audio_dit.transformer_infer, 'forward_block'):
            print("[MOVA] Replacing audio transformer_infer with MOVATransformerInfer")
            self.audio_dit.transformer_infer = MOVATransformerInfer(self.audio_dit.config)
        
        # -------------------------------------------------------------

        # ========== 桥接模块 ==========
        print("Loading bridge...")
        bridge_path = os.path.join(ckpt_path, "dual_tower_bridge")
        with open(os.path.join(bridge_path, "config.json")) as f:
            bridge_config = json.load(f)
        self.bridge = MOVADualTowerBridge(**bridge_config).to(device=device, dtype=torch_dtype)
        bridge_sd = load_safetensors_sharded(bridge_path)
        self.bridge.load_state_dict(bridge_sd, strict=False)
        print(f"Bridge loaded, apply_cross_rope: {self.bridge.apply_cross_rope}")

        # ========== 文本编码器和 tokenizer ==========
        print("Loading text encoder and tokenizer...")
        self.text_encoder = T5EncoderModel.from_pretrained(
            os.path.join(ckpt_path, "text_encoder"),
            torch_dtype=torch_dtype
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(ckpt_path, "tokenizer"),
            use_fast=True
        )

        # ========== 调度器 ==========
        self.scheduler = FlowMatchPairScheduler(
            num_inference_steps=self.num_inference_steps,
            shift=self.sigma_shift,
        )
        self.video_dit.set_scheduler(self.scheduler)
        self.audio_dit.set_scheduler(self.scheduler)
        self._init_distributed()
        print("All components loaded successfully")


    # 以下辅助函数和推理循环（_run_pipeline_manual, run_pipeline 等）与之前相同，此处省略以节省篇幅
    # 请将之前提供的 _run_pipeline_manual, run_pipeline 等代码复制到这里

    def normalize_video_latents(self, latents):
        return (latents - self.vae_mean) * (1.0 / self.vae_std)

    def denormalize_video_latents(self, latents):
        return latents * self.vae_std + self.vae_mean

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

        # 构建跨模态 RoPE（如果有）
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

        # 循环执行各层
        for layer_idx in range(min_layers):
            # 桥接交互（如果该层需要交互）
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
            print(f"[DEBUG] visual_freqs: {visual_freqs.shape if visual_freqs is not None else None}")
            print(f"[DEBUG] audio_freqs: {audio_freqs.shape if audio_freqs is not None else None}")
            # 视频层前向
            v_context_flat = visual_context.view(-1, visual_context.shape[-1])  # [B*L_v, D_v]
            v_pre = type('', (), {})()
            v_pre.context = v_context_flat
            v_pre.embed0 = visual_t_mod          # 保持 3D [B, 6, D]
            v_pre.cos_sin = visual_freqs          # 2D [L_v, head_dim] 或类似
            v_pre.adapter_args = {}

            visual_x = visual_dit.infer_engine.forward_block(
            block_weights=visual_dit.blocks[layer_idx],
            x=visual_x,
            pre_infer_out=v_pre,
            block_idx=layer_idx
        )

            # 音频层前向
            a_context_flat = audio_context.view(-1, audio_context.shape[-1])
            a_pre = type('', (), {})()
            a_pre.context = a_context_flat
            a_pre.embed0 = audio_t_mod
            a_pre.cos_sin = audio_freqs
            a_pre.adapter_args = {}

            audio_x = self.audio_dit.infer_engine.forward_block(
            block_weights=self.audio_dit.blocks[layer_idx],
            x=audio_x,
            pre_infer_out=a_pre,
            block_idx=layer_idx
        )

        # 处理视频剩余的层（如果视频层数多于音频）
        for layer_idx in range(min_layers, visual_layers):
            v_context_flat = visual_context.view(-1, visual_context.shape[-1])
            v_pre = type('', (), {})()
            v_pre.context = v_context_flat
            v_pre.embed0 = visual_t_mod
            v_pre.cos_sin = visual_freqs
            v_pre.adapter_args = {}

            visual_x = visual_dit.infer_engine.forward_block(
                block_weights=visual_dit.blocks[layer_idx],
                x=visual_x,
                pre_infer_out=v_pre,
                block_idx=layer_idx
            )

        return visual_x, audio_x

    # ---------- 手动推理主函数（保留大量调试日志） ----------
    def _run_pipeline_manual(self, prompt, negative_prompt, image, seed,
                              height, width, num_frames, video_fps,
                              num_inference_steps, cfg_scale, sigma_shift):
        device = AI_DEVICE
        dtype = GET_DTYPE()
        # 从视频和音频 DiT 的配置中获取必要参数
        video_freq_dim = self.video_dit.config.get('freq_dim', 256)
        audio_freq_dim = self.audio_dit.config.get('freq_dim', 256)
        video_dim = self.video_dit.config.get('dim', 5120)
        audio_dim = self.audio_dit.config.get('dim', 1536)
        # ========== 1. 图像预处理 ==========
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(image).unsqueeze(0).to(device, dtype=dtype)
        img_tensor_5d = img_tensor.unsqueeze(2)  # [1, 3, 1, height, width]
        print(f"\n[Debug] img_tensor_5d: shape={img_tensor_5d.shape}, "
              f"min={img_tensor_5d.min().item():.3f}, max={img_tensor_5d.max().item():.3f}, "
              f"mean={img_tensor_5d.mean().item():.3f}")

        # ========== 2. 构造 video_condition ==========
        video_condition = torch.cat([
            img_tensor_5d,
            torch.zeros(1, 3, num_frames - 1, height, width, device=device, dtype=dtype)
        ], dim=2)
        print(f"[Debug] video_condition: shape={video_condition.shape}, "
              f"min={video_condition.min().item():.3f}, max={video_condition.max().item():.3f}, "
              f"mean={video_condition.mean().item():.3f}")

        # ========== 3. VAE 编码 ==========
        with torch.no_grad():
            latent_dist = self.video_vae.encode(video_condition).latent_dist
            print(f"[Debug] latent_dist: mean stats: {latent_dist.mean.mean().item():.3f} ± {latent_dist.std.mean().item():.3f}")
            latent_condition = latent_dist.mode()
            print(f"[Debug] latent_condition (before norm): shape={latent_condition.shape}, "
                  f"min={latent_condition.min().item():.3f}, max={latent_condition.max().item():.3f}, "
                  f"mean={latent_condition.mean().item():.3f}, std={latent_condition.std().item():.3f}")

        latent_condition = self.normalize_video_latents(latent_condition)
        print(f"[Debug] latent_condition (after norm): mean={latent_condition.mean().item():.3f}, std={latent_condition.std().item():.3f}")

        # ========== 4. 构造 mask ==========
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
        mask_lat_size = mask_lat_size.permute(0, 2, 1, 3, 4).contiguous()  # [1,4,num_latent_frames,latent_H,latent_W]
        print(f"[Debug] mask_lat_size: shape={mask_lat_size.shape}, "
              f"min={mask_lat_size.min().item():.3f}, max={mask_lat_size.max().item():.3f}, "
              f"mean={mask_lat_size.mean().item():.3f}")

        # ========== 5. 拼接 condition ==========
        condition = torch.cat([mask_lat_size, latent_condition], dim=1)
        print(f"[Debug] condition: shape={condition.shape}, "
              f"min={condition.min().item():.3f}, max={condition.max().item():.3f}, "
              f"mean={condition.mean().item():.3f}, std={condition.std().item():.3f}")
        print(f"[Debug] condition mask part (ch 0-3): mean={condition[:,:4].mean().item():.3f}, std={condition[:,:4].std().item():.3f}")
        print(f"[Debug] condition latent part (ch 4-19): mean={condition[:,4:].mean().item():.3f}, std={condition[:,4:].std().item():.3f}")

        # ========== 6. 调度器初始化 ==========
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas
        print(f"[Debug] timesteps: shape={timesteps.shape}, first 5: {timesteps[:5].cpu().numpy()}")
        print(f"[Debug] sigmas: shape={sigmas.shape}, first 5: {sigmas[:5].cpu().numpy()}")

        # ========== 7. 文本编码 ==========
        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=512,
                                      truncation=True, return_tensors="pt").to(device)
        prompt_embeds = self.text_encoder(**text_inputs).last_hidden_state
        print(f"[Debug] prompt_embeds: shape={prompt_embeds.shape}, "
              f"min={prompt_embeds.min().item():.3f}, max={prompt_embeds.max().item():.3f}, "
              f"mean={prompt_embeds.mean().item():.3f}")

        neg_text_inputs = self.tokenizer(negative_prompt, padding="max_length", max_length=512,
                                         truncation=True, return_tensors="pt").to(device)
        negative_prompt_embeds = self.text_encoder(**neg_text_inputs).last_hidden_state
        print(f"[Debug] negative_prompt_embeds: shape={negative_prompt_embeds.shape}, "
              f"min={negative_prompt_embeds.min().item():.3f}, max={negative_prompt_embeds.max().item():.3f}, "
              f"mean={negative_prompt_embeds.mean().item():.3f}")

        # ========== 8. 初始化 latents ==========
        latents = torch.randn(1, 16, num_latent_frames, latent_height, latent_width,
                              device=device, dtype=dtype)
        print(f"[Debug] Initial latents: mean={latents.mean().item():.3f}, std={latents.std().item():.3f}")

        # ========== 9. 音频 latent ==========
        audio_latent_dim = self.audio_vae.latent_dim
        audio_len = int((num_frames - 1) / video_fps * self.audio_vae.sample_rate / self.audio_vae.hop_length)
        audio_latents = torch.randn(1, audio_latent_dim, audio_len, device=device, dtype=dtype)
        print(f"[Debug] Audio latents: shape={audio_latents.shape}, "
              f"mean={audio_latents.mean().item():.3f}, std={audio_latents.std().item():.3f}")

        # ========== 10. 去噪循环 ==========
        # 在循环前获取维度（可从配置中获取）
        video_dim = self.video_dit.config.get('dim', 5120)
        audio_dim = self.audio_dit.config.get('dim', 1536)

        for i, t in enumerate(timesteps):
            t_tensor = t.unsqueeze(0).to(device=device, dtype=dtype)
            print(f"\n--- Step {i} (t={t.item():.2f}) ---")

            # 视频时间嵌入
            t_emb_v = sinusoidal_embedding_1d(video_freq_dim, t_tensor)
            embed_v = self.video_dit.pre_weight.time_embedding_0.apply(t_emb_v)
            embed_v = F.silu(embed_v)
            embed_v = self.video_dit.pre_weight.time_embedding_2.apply(embed_v)
            visual_t = embed_v
            visual_t_mod = self.video_dit.pre_weight.time_projection_1.apply(visual_t)
            visual_t_mod = visual_t_mod.unflatten(1, (6, video_dim))

            # 音频时间嵌入
            t_emb_a = sinusoidal_embedding_1d(audio_freq_dim, t_tensor)
            embed_a = self.audio_dit.pre_weight.time_embedding_0.apply(t_emb_a)
            embed_a = F.silu(embed_a)
            embed_a = self.audio_dit.pre_weight.time_embedding_2.apply(embed_a)
            audio_t = embed_a
            audio_t_mod = self.audio_dit.pre_weight.time_projection_1.apply(audio_t)
            audio_t_mod = audio_t_mod.unflatten(1, (6, audio_dim))

            print(f"    visual_t: mean={visual_t.mean().item():.3f}, std={visual_t.std().item():.3f}")
            print(f"    visual_t_mod: shape={visual_t_mod.shape}, mean={visual_t_mod.mean().item():.3f}")
            print(f"    audio_t: mean={audio_t.mean().item():.3f}, std={audio_t.std().item():.3f}")

            # 文本嵌入
            # 视频文本嵌入
            out_v = self.video_dit.pre_weight.text_embedding_0.apply(prompt_embeds.squeeze(0))
            out_v = F.gelu(out_v, approximate="tanh")
            visual_context_emb = self.video_dit.pre_weight.text_embedding_2.apply(out_v)
            visual_context_emb = visual_context_emb.unsqueeze(0)  # [1, seq_len, dim]

            # 音频文本嵌入
            out_a = self.audio_dit.pre_weight.text_embedding_0.apply(prompt_embeds.squeeze(0))
            out_a = F.gelu(out_a, approximate="tanh")
            audio_context_emb = self.audio_dit.pre_weight.text_embedding_2.apply(out_a)
            audio_context_emb = audio_context_emb.unsqueeze(0)

            # 负分支同理
            out_neg_v = self.video_dit.pre_weight.text_embedding_0.apply(negative_prompt_embeds.squeeze(0))
            out_neg_v = F.gelu(out_neg_v, approximate="tanh")
            neg_visual_context_emb = self.video_dit.pre_weight.text_embedding_2.apply(out_neg_v)
            neg_visual_context_emb = neg_visual_context_emb.unsqueeze(0)

            out_neg_a = self.audio_dit.pre_weight.text_embedding_0.apply(negative_prompt_embeds.squeeze(0))
            out_neg_a = F.gelu(out_neg_a, approximate="tanh")
            neg_audio_context_emb = self.audio_dit.pre_weight.text_embedding_2.apply(out_neg_a)
            neg_audio_context_emb = neg_audio_context_emb.unsqueeze(0)
            # 拼接视频输入
            video_input = torch.cat([latents, condition], dim=1)
            video_input = video_input.to(dtype=dtype)
            print(f"    video_input: shape={video_input.shape}, mean={video_input.mean().item():.3f}, std={video_input.std().item():.3f}")

            # patchify
            visual_x, (t_len, h_len, w_len) = self.video_dit.patchify(video_input)
            audio_x, (a_len,) = self.audio_dit.patchify(audio_latents)
            print(f"    visual_x: shape={visual_x.shape}, mean={visual_x.mean().item():.3f}, std={visual_x.std().item():.3f}")
            print(f"    audio_x: shape={audio_x.shape}, mean={audio_x.mean().item():.3f}, std={audio_x.std().item():.3f}")

            # 构建频率
            visual_freqs = torch.cat([
                self.video_dit.freqs[0][:t_len].view(t_len, 1, 1, -1).expand(t_len, h_len, w_len, -1),
                self.video_dit.freqs[1][:h_len].view(1, h_len, 1, -1).expand(t_len, h_len, w_len, -1),
                self.video_dit.freqs[2][:w_len].view(1, 1, w_len, -1).expand(t_len, h_len, w_len, -1)
            ], dim=-1).reshape(t_len * h_len * w_len, 1, -1).to(device=device, dtype=dtype)
            audio_freqs = self.audio_dit.freqs[0][:a_len].view(a_len, 1, -1).to(device=device, dtype=dtype)

            # 对齐频率（用于桥接）
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

                # 视频 head
                v_pos_flat = v_pos.view(-1, v_pos.shape[-1])  # [B*L_v, D_v]
                v_pos_norm = self.video_dit.transformer_weights.norm.apply(v_pos_flat)
                mod_v = self.video_dit.transformer_weights.head_modulation.tensor  # [1, 2, D_v]
                e_v = (mod_v + visual_t.unsqueeze(1)).chunk(2, dim=1)
                shift_v, scale_v = e_v[0].squeeze(1), e_v[1].squeeze(1)
                v_pos_mod = v_pos_norm * (1 + scale_v) + shift_v
                v_pos_head_out = self.video_dit.transformer_weights.head.apply(v_pos_mod)  # [B*L_v, out_dim]
                v_pos_head = v_pos_head_out.view(1, -1, v_pos_head_out.shape[-1])  # [1, L_v, out_dim]

                # 音频 head
                a_pos_flat = a_pos.view(-1, a_pos.shape[-1])  # [B*L_a, D_a]
                a_pos_norm = self.audio_dit.transformer_weights.norm.apply(a_pos_flat)
                mod_a = self.audio_dit.transformer_weights.head_modulation.tensor  # [1, 2, D_a]
                e_a = (mod_a + audio_t.unsqueeze(1)).chunk(2, dim=1)
                shift_a, scale_a = e_a[0].squeeze(1), e_a[1].squeeze(1)
                a_pos_mod = a_pos_norm * (1 + scale_a) + shift_a
                a_pos_head_out = self.audio_dit.transformer_weights.head.apply(a_pos_mod)  # [B*L_a, out_dim_a]
                a_pos_head = a_pos_head_out.view(1, -1, a_pos_head_out.shape[-1])

                # unpatchify
                v_pos_unpatch = self.video_dit.unpatchify(v_pos_head, (t_len, h_len, w_len))
                a_pos_unpatch = self.audio_dit.unpatchify(a_pos_head, (a_len,))
                print(f"    v_pos_unpatch: shape={v_pos_unpatch.shape}, mean={v_pos_unpatch.mean().item():.3f}, std={v_pos_unpatch.std().item():.3f}")
                print(f"    a_pos_unpatch: shape={a_pos_unpatch.shape}, mean={a_pos_unpatch.mean().item():.3f}, std={a_pos_unpatch.std().item():.3f}")
                del v_pos, a_pos, v_pos_head, a_pos_head
                a_pos_head = a_pos_head_out.view(1, -1, a_pos_head_out.shape[-1])
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

                # 视频 head
                v_neg_flat = v_neg.view(-1, v_neg.shape[-1])  # [B*L_v, D_v]
                v_neg_norm = self.video_dit.transformer_weights.norm.apply(v_neg_flat)
                e_v_neg = (mod_v + visual_t.unsqueeze(1)).chunk(2, dim=1)
                shift_v_neg, scale_v_neg = e_v_neg[0].squeeze(1), e_v_neg[1].squeeze(1)
                v_neg_mod = v_neg_norm * (1 + scale_v_neg) + shift_v_neg
                v_neg_head_out = self.video_dit.transformer_weights.head.apply(v_neg_mod)
                v_neg_head = v_neg_head_out.view(1, -1, v_neg_head_out.shape[-1])

                # 音频 head
                a_neg_flat = a_neg.view(-1, a_neg.shape[-1])
                a_neg_norm = self.audio_dit.transformer_weights.norm.apply(a_neg_flat)
                e_a_neg = (mod_a + audio_t.unsqueeze(1)).chunk(2, dim=1)
                shift_a_neg, scale_a_neg = e_a_neg[0].squeeze(1), e_a_neg[1].squeeze(1)
                a_neg_mod = a_neg_norm * (1 + scale_a_neg) + shift_a_neg
                a_neg_head_out = self.audio_dit.transformer_weights.head.apply(a_neg_mod)
                a_neg_head = a_neg_head_out.view(1, -1, a_neg_head_out.shape[-1])

                # unpatchify
                v_neg_unpatch = self.video_dit.unpatchify(v_neg_head, (t_len, h_len, w_len))
                a_neg_unpatch = self.audio_dit.unpatchify(a_neg_head, (a_len,))
                print(f"    v_neg_unpatch: shape={v_neg_unpatch.shape}, mean={v_neg_unpatch.mean().item():.3f}, std={v_neg_unpatch.std().item():.3f}")
                print(f"    a_neg_unpatch: shape={a_neg_unpatch.shape}, mean={a_neg_unpatch.mean().item():.3f}, std={a_neg_unpatch.std().item():.3f}")
                del v_neg, a_neg, v_neg_head, a_neg_head
            v_pred = v_neg_unpatch + cfg_scale * (v_pos_unpatch - v_neg_unpatch)
            a_pred = a_neg_unpatch + cfg_scale * (a_pos_unpatch - a_neg_unpatch)
            print(f"    v_pred: mean={v_pred.mean().item():.3f}, std={v_pred.std().item():.3f}")
            print(f"    a_pred: mean={a_pred.mean().item():.3f}, std={a_pred.std().item():.3f}")

            # 更新
            latents = self.scheduler.step(v_pred, t, latents)
            audio_latents = self.scheduler.step(a_pred, t, audio_latents)
            print(f"    latents after step: mean={latents.mean().item():.3f}, std={latents.std().item():.3f}")
            print(f"    audio_latents after step: mean={audio_latents.mean().item():.3f}, std={audio_latents.std().item():.3f}")

            if i % 5 == 0:
                torch.cuda.empty_cache()

        # ========== 11. 解码视频 ==========
        with torch.no_grad():
            video_latents = self.denormalize_video_latents(latents)
            print(f"[Debug] video_latents (after denorm): mean={video_latents.mean().item():.3f}, std={video_latents.std().item():.3f}")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video = self.video_vae.decode(video_latents).sample
            video = video.float().cpu()
            print(f"[Debug] video (raw decode): shape={video.shape}, min={video.min().item():.3f}, max={video.max().item():.3f}")

            video = (video + 1) / 2
            video = video.clamp(0, 1)
            print(f"[Debug] video (after norm to [0,1]): min={video.min().item():.3f}, max={video.max().item():.3f}")

            video_frames = []
            for i in range(video.shape[2]):
                frame = video[0, :, i, :, :].permute(1, 2, 0).numpy()
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
                video_frames.append(Image.fromarray(frame))
            print(f"[Debug] Converted {len(video_frames)} frames")

        # ========== 12. 解码音频 ==========
        with torch.no_grad():
            audio = self.audio_vae.decode(audio_latents)
            audio = audio.cpu().squeeze(0)
            audio = audio.float().unsqueeze(0)
            print(f"[Debug] audio: shape={audio.shape}, min={audio.min().item():.3f}, max={audio.max().item():.3f}")

        return video_frames, audio

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

        from mova.datasets.transforms.custom import crop_and_resize
        img = Image.open(image_path).convert("RGB")
        img = crop_and_resize(img, self.height, self.width)
        print(f"[Pipeline] Input image resized to {self.height}x{self.width}")

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

    # 占位方法
    def run_text_encoder(self, input_info):
        pass

    def run_vae_encoder(self, img=None):
        pass

    def process_images_after_vae_decoder(self):
        pass

    def init_run(self):
        pass