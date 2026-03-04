import sys
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
sys.path.append("C:/Users/ASUS/Desktop/LightX2V")  # 你的LightX2V根目录

from lightx2v.models.networks.mova.mova_model import MOVAModel

# 视频塔配置
video_tower_config = {
    "dim": 3072,
    "in_dim": 16,
    "ffn_dim": 8960,
    "out_dim": 16,
    "text_dim": 4096,
    "freq_dim": 256,
    "eps": 1e-6,
    "patch_size": [1, 2, 2],
    "num_heads": 12,
    "num_layers": 30,
    "has_image_input": False,  # 测试T2VA时设为False
    "has_image_pos_emb": False,
    "has_ref_conv": False,
}

# 音频塔配置
audio_tower_config = {
    "dim": 1536,
    "in_dim": 8,
    "ffn_dim": 4096,
    "out_dim": 1,
    "text_dim": 4096,
    "freq_dim": 256,
    "eps": 1e-6,
    "patch_size": 4,          # 注意：音频塔的patch_size是int
    "num_heads": 12,
    "num_layers": 30,
    "vae_type": "dac",
    "has_image_input": False,
}

# 桥接模块配置
bridge_config = {
    "visual_layers": 30,
    "audio_layers": 30,
    "visual_hidden_dim": 3072,
    "audio_hidden_dim": 1536,
    "audio_fps": 44100 / 2048,  # ≈21.533
    "head_dim": 256,
    "interaction_strategy": "shallow_focus",
    "apply_cross_rope": True,
    "pooled_adaln": False,
}

# 实例化模型
model = MOVAModel(
    video_tower_config=video_tower_config,
    audio_tower_config=audio_tower_config,
    bridge_config=bridge_config,
)
print("✅ Model instantiated successfully")

# 创建 dummy 输入
batch_size = 1
video_latent = torch.randn(batch_size, 16, 48, 30, 40)  # (B, C, T, H, W)
audio_latent = torch.randn(batch_size, 8, 1500)          # (B, C, T)
timestep = torch.tensor([0.5])
context = torch.randn(batch_size, 77, 4096)              # (B, seq_len, dim)

# 前向传播测试
with torch.no_grad():
    v_pred, a_pred = model(video_latent, audio_latent, timestep, context)

print(f"✅ v_pred shape: {v_pred.shape}")
print(f"✅ a_pred shape: {a_pred.shape}")