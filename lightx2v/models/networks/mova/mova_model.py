import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from .wan_video_dit import WanModel
from .wan_audio_dit import WanAudioModel
from .interactionv2 import DualTowerConditionalBridge

def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)

class MOVAModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        video_tower_config: dict,
        audio_tower_config: dict,
        bridge_config: dict,
        **kwargs
    ):
        super().__init__()
        self.video_tower = WanModel(**video_tower_config)
        self.audio_tower = WanAudioModel(**audio_tower_config)
        self.bridge = DualTowerConditionalBridge(**bridge_config)
        self.num_layers = video_tower_config.get("num_layers", 30)

    def forward(self, video_latent, audio_latent, timestep, context, video_grid_size=None, bridge_scale=1.0, step_idx=0, branch='unc', video_fps=24.0, **kwargs):
        # 1. Patchify
        v_hidden, (f, h, w) = self.video_tower.patchify(video_latent)
        # 固定缩放因子，使标准差约为 1 (基于当前 std=0.128，1/0.128 ≈ 7.8)
        scale_factor = 7.8
        v_hidden = v_hidden * scale_factor
        print(f"After patchify scaling: mean={v_hidden.mean():.3f}, std={v_hidden.std():.3f}, scale_factor={scale_factor}")
        a_hidden, (t,) = self.audio_tower.patchify(audio_latent)

        # 关键修复：缩放 v_hidden 使其 std 为 1（可选，根据调试需要保留或移除）
        # scale_factor = 1.0 / (v_hidden.std() + 1e-8)
        # v_hidden = v_hidden * scale_factor
        # if step_idx == 0:
        #     print(f"After patchify scaling: mean={v_hidden.mean():.3f}, std={v_hidden.std():.3f}, scale_factor={scale_factor:.3f}")

        # 2. 时间嵌入
        t_emb_v = self.video_tower.time_embedding(
            sinusoidal_embedding_1d(self.video_tower.freq_dim, timestep)
        )
        t_mod_v = self.video_tower.time_projection(t_emb_v).unflatten(1, (6, self.video_tower.dim))

        t_emb_a = self.audio_tower.time_embedding(
            sinusoidal_embedding_1d(self.audio_tower.freq_dim, timestep)
        )
        t_mod_a = self.audio_tower.time_projection(t_emb_a).unflatten(1, (6, self.audio_tower.dim))

        if step_idx == 0:
            print(f"t_mod_v: mean={t_mod_v.mean():.3f}, std={t_mod_v.std():.3f}, min={t_mod_v.min():.3f}, max={t_mod_v.max():.3f}")
            print(f"modulation[0]: mean={self.video_tower.blocks[0].modulation.mean():.3f}, std={self.video_tower.blocks[0].modulation.std():.3f}")

        # 3. 文本嵌入
        if context is None:
            B = video_latent.shape[0]
            text_dim = self.video_tower.text_embedding[0].in_features
            context_zero = torch.zeros(B, 1, text_dim, device=video_latent.device, dtype=video_latent.dtype)
            context_v = self.video_tower.text_embedding(context_zero)
            context_a = self.audio_tower.text_embedding(context_zero)
        else:
            context_v = self.video_tower.text_embedding(context)
            context_a = self.audio_tower.text_embedding(context)

        # 4. 构造视频和音频自注意力的频率（单张量）
        v_freqs_self = torch.cat([
            self.video_tower.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.video_tower.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.video_tower.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(v_hidden.device)

        a_freqs_self = torch.cat([
            self.audio_tower.freqs[0][:t].view(t, -1).expand(t, -1),
            self.audio_tower.freqs[1][:t].view(t, -1).expand(t, -1),
            self.audio_tower.freqs[2][:t].view(t, -1).expand(t, -1),
        ], dim=-1).reshape(t, 1, -1).to(a_hidden.device)

        # 5. 构造对齐的跨模态频率（用于桥接模块）
        if self.bridge.apply_cross_rope:
            (visual_rope_cos_sin, audio_rope_cos_sin) = self.bridge.build_aligned_freqs(
                video_fps=video_fps,
                grid_size=(f, h, w),
                audio_steps=a_hidden.shape[1],
                device=v_hidden.device,
                dtype=v_hidden.dtype,
            )
        else:
            visual_rope_cos_sin = None
            audio_rope_cos_sin = None

        # 6. 逐层处理
        num_v_layers = len(self.video_tower.blocks)
        num_a_layers = len(self.audio_tower.blocks)
        num_layers = min(num_v_layers, num_a_layers, self.num_layers)

        if step_idx == 0:
            print("="*50)
            print(f"Initial v_hidden after patchify: mean={v_hidden.mean():.3f}, std={v_hidden.std():.3f}, min={v_hidden.min():.3f}, max={v_hidden.max():.3f}")

        for layer_idx in range(num_layers):
            # 视频块
            v_hidden = self.video_tower.blocks[layer_idx](
                v_hidden, context_v, t_mod_v, v_freqs_self,
                step_idx=step_idx, layer_idx=layer_idx, branch=branch
            )
            # 音频块
            a_hidden = self.audio_tower.blocks[layer_idx](
                a_hidden, context_a, t_mod_a, a_freqs_self
            )
            # 桥接模块（使用对齐的频率对）
            v_hidden, a_hidden = self.bridge(
                layer_idx=layer_idx,
                visual_hidden_states=v_hidden,
                audio_hidden_states=a_hidden,
                x_freqs=visual_rope_cos_sin,
                y_freqs=audio_rope_cos_sin,
                video_grid_size=(f, h, w),
                condition_scale=bridge_scale,
            )
            if step_idx == 0 and branch == 'unc':
                print(f"Layer {layer_idx}: v_hidden mean={v_hidden.mean():.3f}, std={v_hidden.std():.3f}, min={v_hidden.min():.3f}, max={v_hidden.max():.3f}")

        # 7. Head 输出预测
        v_pred_before_head = v_hidden
        if step_idx == 0 and branch == 'unc':
            print(f"Before head: mean={v_pred_before_head.mean():.3f}, std={v_pred_before_head.std():.3f}")
        v_pred = self.video_tower.head(v_hidden, t_emb_v)
        a_pred = self.audio_tower.head(a_hidden, t_emb_a)
        if step_idx == 0 and branch == 'unc':
            print(f"After head: mean={v_pred.mean():.3f}, std={v_pred.std():.3f}, min={v_pred.min():.3f}, max={v_pred.max():.3f}")

        # 8. Unpatchify
        v_pred = self.video_tower.unpatchify(v_pred, (int(f), int(h), int(w)))
        a_pred = self.audio_tower.unpatchify(a_pred, (int(t),))
        if step_idx == 0 and branch == 'unc':
            print(f"After unpatchify: mean={v_pred.mean():.3f}, std={v_pred.std():.3f}, min={v_pred.min():.3f}, max={v_pred.max():.3f}")
            print("="*50)

        return v_pred, a_pred