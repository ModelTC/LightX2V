# 步数蒸馏

步数蒸馏是 LightX2V 中的一项重要优化技术，通过训练蒸馏模型将推理步数从原始的 40-50 步大幅减少到 **4 步**，在保持视频质量的同时显著提升推理速度。LightX2V 在实现步数蒸馏的同时也加入了 CFG 蒸馏，进一步提升推理速度。

## 🔍 技术原理

### DMD 蒸馏

步数蒸馏的核心技术是 [DMD 蒸馏](https://arxiv.org/abs/2311.18828)。DMD 蒸馏的框架如下图所示：

<div align="center">
<img alt="DMD 蒸馏框架" src="https://raw.githubusercontent.com/ModelTC/LightX2V/main/assets/figs/step_distill/fig_01.png" width="75%">
</div>

DMD蒸馏的核心思想是最小化蒸馏模型与原始模型输出分布的 KL 散度：

$$
\begin{aligned}
D_{KL}\left(p_{\text{fake}} \; \| \; p_{\text{real}} \right) &= \mathbb{E}{x\sim p\text{fake}}\left(\log\left(\frac{p_\text{fake}(x)}{p_\text{real}(x)}\right)\right)\\
&= \mathbb{E}{\substack{
z \sim \mathcal{N}(0; \mathbf{I}) \\
x = G_\theta(z)
}}-\big(\log~p_\text{real}(x) - \log~p_\text{fake}(x)\big).
\end{aligned}
$$

由于直接计算概率密度几乎是不可能的，因此 DMD 蒸馏改为计算这个 KL 散度的梯度：

$$
\begin{aligned}
\nabla_\theta D_{KL}
&= \mathbb{E}{\substack{
z \sim \mathcal{N}(0; \mathbf{I}) \\
x = G_\theta(z)
} } \Big[-
\big(
s_\text{real}(x) - s_\text{fake}(x)\big)
\hspace{.5mm} \frac{dG}{d\theta}
\Big],
\end{aligned}
$$

其中 $s_\text{real}(x) =\nabla_{x} \text{log}~p_\text{real}(x)$ 和 $s_\text{fake}(x) =\nabla_{x} \text{log}~p_\text{fake}(x)$ 为得分函数。得分函数可以由模型进行计算。因此，DMD 蒸馏一共维护三个模型：

- `real_score`，计算真实分布的得分；由于真实分布是固定的，因此 DMD 蒸馏使用固定权重的原始模型作为其得分函数；
- `fake_score`，计算伪分布的得分；由于伪分布是不断更新的，因此 DMD 蒸馏使用原始模型对其初始化，并对其进行微调以学习生成器的输出分布；
- `generator`，学生模型，通过计算 `real_score` 与 `fake_score` KL 散度的梯度指导其优化方向。

> 参考文献：
> 1. [DMD (One-step Diffusion with Distribution Matching Distillation)](https://arxiv.org/abs/2311.18828)
> 2. [DMD2 (Improved Distribution Matching Distillation for Fast Image Synthesis)](https://arxiv.org/abs/2405.14867)

### Self-Forcing

DMD 蒸馏技术是针对图像生成的。Lightx2v 中的步数蒸馏基于 [Self-Forcing](https://github.com/guandeh17/Self-Forcing) 技术实现。Self-Forcing 的整体实现与 DMD 类似，但是仿照 DMD2，去掉了它的回归损失，而是使用了 ODE 初始化。此外，Self-Forcing 针对视频生成任务加入了一个重要优化：

目前基于 DMD 蒸馏的方法难以一步生成视频。Self-Forcing 每次选择一个时间步进行优化，generator 仅仅在这一步计算梯度。这种方法使得 Self-Forcing 的训练速度显著提升，并且提升了中间时间步的去噪质量，其效果亦有所提升。

> 参考文献：
> 1. [Self-Forcing (Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion)](https://arxiv.org/abs/2506.08009)

### Lightx2v

Self-Forcing 针对 1.3B 的自回归模型进行步数蒸馏、CFG蒸馏。LightX2V 在其基础上，进行了一系列扩展：

1. **更大的模型**：支持 14B 模型的步数蒸馏训练；
2. **更多的模型**：支持标准的双向模型，以及 I2V 模型的步数蒸馏训练；
3. **更好的效果**：Lightx2v 使用了约 50,000 条数据的高质量 prompt 进行训练；

具体实现可参考 [Self-Forcing-Plus](https://github.com/GoatWu/Self-Forcing-Plus)。

## 🎯 技术特性

- **推理加速**：推理步数从 40-50 步减少到 4 步且无需 CFG，速度提升约 **20-24x**
- **质量保持**：通过蒸馏技术保持原有的视频生成质量
- **兼容性强**：支持 T2V 和 I2V 任务
- **使用灵活**：支持加载完整步数蒸馏模型，或者在原生模型的基础上加载步数蒸馏LoRA；支持与 int8/fp8 模型量化相兼容

## 🛠️ 配置文件说明

### 基础配置文件

在 [configs/distill/](https://github.com/ModelTC/lightx2v/tree/main/configs/distill) 目录下提供了多种配置选项：

| 配置文件 | 用途 | 模型地址 |
|----------|------|------------|
| [wan_t2v_distill_model_4step_cfg.json](https://github.com/ModelTC/lightx2v/blob/main/configs/distill/wan21/wan_t2v_distill_model_4step_cfg.json) | 加载 T2V 4步蒸馏完整模型 | [hugging-face](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v/blob/main/distill_models/distill_model.safetensors) |
| [wan_i2v_distill_4step_cfg.json](https://github.com/ModelTC/lightx2v/blob/main/configs/distill/wan_i2v_distill_4step_cfg.json) | 加载 I2V 4步蒸馏完整模型 | [hugging-face](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/blob/main/distill_models/distill_model.safetensors) |
| [wan_t2v_distill_lora_4step_cfg.json](https://github.com/ModelTC/lightx2v/blob/main/configs/distill/wan21/wan_t2v_distill_lora_4step_cfg.json) | 加载 Wan-T2V 模型和步数蒸馏 LoRA | [hugging-face](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v/blob/main/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors) |
| [wan_i2v_distill_lora_4step_cfg.json](https://github.com/ModelTC/lightx2v/blob/main/configs/distill/wan21/wan_i2v_distill_lora_4step_cfg.json) | 加载 Wan-I2V 模型和步数蒸馏 LoRA | [hugging-face](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/blob/main/loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors) |

### 关键配置参数

- 由于 DMD 蒸馏仅训练几个固定的时间步，因此我们推荐使用 `LCM Scheduler` 进行推理。[WanStepDistillScheduler](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/models/schedulers/wan/step_distill/scheduler.py) 中，已经固定使用 `LCM Scheduler`，无需用户进行配置。
- `infer_steps`, `denoising_step_list` 和 `sample_shift` 设置为与训练时相匹配的参数，一般不建议用户修改。
- `enable_cfg` 一定设置为 `false`（等价于设置 `sample_guide_scale = 1`），否则可能出现视频完全模糊的现象。
- `lora_configs` 支持融合不同强度的多个 lora。当 `lora_configs` 不为空时，默认加载原始的 `Wan2.1` 模型。因此使用 `lora_config` 并且想要使用步数蒸馏时，请设置步数蒸馏lora的路径与强度。

```json
{
  "infer_steps": 4,                              // 推理步数
  "denoising_step_list": [1000, 750, 500, 250],  // 去噪时间步列表
  "sample_shift": 5,                             // 调度器 timestep shift
  "enable_cfg": false,                           // 关闭CFG以提升速度
  "lora_configs": [                              // LoRA权重路径（可选）
    {
      "path": "path/to/distill_lora.safetensors",
      "strength": 1.0
    }
  ]
}
```

## 📜 使用方法

### 模型准备

**完整模型：**
将下载好的模型（`distill_model.pt` 或者 `distill_model.safetensors`）放到 Wan 模型根目录的 `distill_models/` 文件夹下即可

- 对于 T2V：`Wan2.1-T2V-14B/distill_models/`
- 对于 I2V-480P：`Wan2.1-I2V-14B-480P/distill_models/`

**LoRA：**

1. 将下载好的 LoRA 放到任意位置
2. 修改配置文件中的 `lora_path` 参数为 LoRA 存放路径即可

### 推理脚本

**T2V 完整模型：**

```bash
bash scripts/wan/distill/run_wan_t2v_distill_model_4step_cfg.sh
```

**I2V 完整模型：**

```bash
bash scripts/wan/distill/run_wan_i2v_distill_4step_cfg.sh
```

### 步数蒸馏 LoRA 推理脚本

**T2V LoRA：**

```bash
bash scripts/wan/distill/run_wan_t2v_distill_lora_4step_cfg.sh
```

**I2V LoRA：**

```bash
bash scripts/wan/distill/run_wan_i2v_distill_lora_4step_cfg.sh
```

## 🔧 服务化部署

### 启动蒸馏模型服务

对 [scripts/server/start_server.sh](https://github.com/ModelTC/lightx2v/blob/main/scripts/server/start_server.sh) 中的启动命令进行修改：

```bash
python -m lightx2v.api_server \
  --model_cls wan2.1 \
  --task t2v \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/distill/wan21/wan_t2v_distill_model_4step_cfg.json \
  --port 8000 \
  --nproc_per_node 1
```

运行服务启动脚本：

```bash
scripts/server/start_server.sh
```

更多详细信息见[服务化部署](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/deploy_service.html)。

### 在 Gradio 界面中使用

见 [Gradio 文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/deploy_gradio.html)
