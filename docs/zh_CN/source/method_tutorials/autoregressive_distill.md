# 自回归蒸馏

自回归蒸馏是 LightX2V 中的一个技术探索，通过训练蒸馏模型将推理步数从原始的 40-50 步减少到 **8 步**，在实现推理加速的同时能够通过 KV Cache 技术生成无限长视频。

> ⚠️ 警告：目前自回归蒸馏的效果一般，加速效果也没有达到预期，但是可以作为一个长期的研究项目。目前 LightX2V 仅支持 T2V 的自回归模型。

## 🔍 技术原理

自回归蒸馏通过 [CausVid](https://github.com/tianweiy/CausVid) 技术实现。CausVid 针对 1.3B 的自回归模型进行步数蒸馏、CFG蒸馏。LightX2V 在其基础上，进行了一系列扩展：

1. **更大的模型**：支持 14B 模型的自回归蒸馏训练；
2. **更完整的数据处理流程**：生成 50,000 个提示词-视频对的训练数据集；

具体实现可参考 [CausVid-Plus](https://github.com/GoatWu/CausVid-Plus)。

## 🛠️ 配置文件说明

### 配置文件

在 [configs/causvid/](https://github.com/ModelTC/lightx2v/tree/main/configs/causvid) 目录下提供了配置选项：

| 配置文件 | 模型地址 |
|----------|------------|
| [wan_t2v_causvid.json](https://github.com/ModelTC/lightx2v/blob/main/configs/causvid/wan_t2v_causvid.json) | https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid |

### 关键配置参数

```json
{
  "enable_cfg": false,          // 关闭CFG以提升速度
  "num_fragments": 3,           // 一次生成视频的段数，每段5s
  "num_frames": 21,             // 每段视频的帧数，谨慎修改！
  "num_frame_per_block": 3,     // 每个自回归块的帧数，谨慎修改！
  "num_blocks": 7,              // 每段视频的自回归块数，谨慎修改！
  "frame_seq_length": 1560,     // 每帧的编码长度，谨慎修改！
  "denoising_step_list": [      // 去噪时间步列表
    999, 934, 862, 756, 603, 410, 250, 140, 74
  ]
}
```

## 📜 使用方法

### 模型准备

将下载好的模型（`causal_model.pt` 或者 `causal_model.safetensors`）放到 Wan 模型根目录的 `causvid_models/` 文件夹下即可
- 对于 T2V：`Wan2.1-T2V-14B/causvid_models/`

### 推理脚本

```bash
bash scripts/wan/run_wan_t2v_causvid.sh
```
