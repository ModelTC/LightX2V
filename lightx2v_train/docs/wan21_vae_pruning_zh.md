# Wan2.1 VAE：编码器与解码器两阶段剪枝蒸馏

这两套实验独立于 H3，不加载 H3 的 VAE、latent 或剪枝导出。只复用现有清单中的原始视频路径。

## 训练对象

| 实验 | 搜索预算 | 搜索阶段 | 恢复阶段 |
| --- | --- | --- | --- |
| Encoder | 10 个残差主分支保留 3 个 | 原权重冻结，训练门控与临时 rank16/alpha32 卷积适配器 | 学生 encoder 全参；原版 teacher decoder 冻结，但保留输入梯度 |
| Decoder | 14 个残差主分支保留 5 个 | 同上 | 学生 decoder 全参；原版 teacher encoder 在线生成 latent |

两端都是全局固定数量选择，不采用连续分组。门控只删除残差主分支，形状变换旁路、上/下采样等必要部分保留。导出按门控 EMA 选结构，复制对应的原始教师权重，不合并搜索适配器。恢复阶段重新建立优化器与调度。两端独立恢复后能否直接组合达到原版质量，还需要验证。

## 数据与数值约定

默认读取现有 T2AV、I2AV、FL2AV、L2AV 四份视频清单，但 `load_cached_latents=false`，不会读取里面的 H3 latent。缺少旧 latent 不会丢弃仍有原视频的样本。无需先生成 Wan latent cache。

- `VideoDataset` 将视频保持宽高比缩放后居中裁剪，不拉伸为正方形；画面左右或上下内容可能被裁掉。
- 训练读取随机起点的 65 帧、384×384；验证固定起点65帧、512×512。按24fps采样，并非强制重建完整5秒视频。
- `geometry_from_metadata=false`，不沿用 H3 清单中的124帧/768p几何。时序满足 `4n+1`，空间为8的倍数；当前公共视频读取器还要求配置尺寸为16的倍数，默认值满足两者。
- 数据读取器输出 `[-1,1]`，本 processor 转为 `[0,1]`；Wan 模型封装负责转换回原版 VAE 的 `[-1,1]`。
- 这四份训练配置设置 `min_source_frames: 65`，训练与验证都要求至少65个真实源帧。短源会在读取时明确报错，不能用末帧填充伪造满足训练长度的样本。默认读取器恰好选取65帧，因此正常样本没有时间填充。
- 每个训练裁剪作为独立短视频，从时间零重新进行学生和教师编码；不能用整段视频的cached latent裁剪替代，因为两者的因果历史不同。

训练损失进一步在视频内抽取连续片段和空间裁剪：搜索及恢复前600步为33帧、256×256；恢复第600步起为65帧、384×384。以上是计算规模设置，不是已经证明足够恢复高分辨率长视频质量。

## 损失与调度

| 阶段 | 更新次数 | 重建 | VGG-LPIPS | Feature MSE | GAN | Aux | Encoder额外后验 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 搜索 | 0–999 | 1 | 0.05 | 0 | 0 | 0 | 1 |
| 恢复前段 | 0–599 | 1 | 0.1 | 0.01 | 0 | 0 | 1 |
| 恢复后段 | 600–2999 | 1 | 0.2 | 0.005 | 0.5 | 渐入至0.1 | 1 |

主重建是原版 Wan `[-1,1]` 空间中的 Charbonnier，目标为原始视频。LPIPS使用反标准化后的RGB；搜索最多32帧，恢复最多64帧，每次计算2帧。GAN采用条件3D加逐帧2D分支，连续最多32帧、256/384空间裁剪。

GAN条件使用停止梯度的教师潜变量：首个时间token对应1帧，后续每个对应4帧，先展开到像素时间轴再同步裁剪；空间按8倍压缩对齐。

Feature MSE对应原始残差索引的同形输出，包括被剪主分支剩下的shortcut输出，不是H3的stage锚点。Aux每次只采一个锚点：encoder `[5,7,8]`，decoder `[8,11,12]`（从零计数）。学生特征经过冻结教师后缀，与教师重建计算 `重建 + 0.1空间差分 + 0.1时空混合差分`。

Encoder后验对齐监督均值及标准差，按原版Wan latent统计量缩放；主重建通过冻结原版decoder回传到学生encoder。Decoder训练不添加后验loss。

Aux从恢复第600步起渐入200步；GAN判别器同期开启，预热100步，再将生成器修正幅度渐入300步。GAN沿用独立判别器优化与停止梯度修正目标，不是DMD假分数网络。

搜索学习率1e-4、gate学习率×10、温度4→0.1、EMA0.999、预热50步；恢复学习率5e-5、预热200步。两阶段余弦衰减，每卡1个样本、梯度累积4步；双卡每次更新累计8个样本/裁剪组。

两阶段均每100步保存checkpoint、每100步生成一个重建预览，保留最近5个checkpoint。预览来自训练清单，不是独立验证集。默认训练4个worker、pin memory开启；验证2个worker、pin memory关闭。

## 启动

默认使用现有 Python 环境，不重新安装软件。解释器路径中的 `MiniMax-H3/local_diffusers/.venv` 只是环境位置，模型权重仍是 Wan。默认权重：

```text
/data/nvme6/gushiqiao/models/Sekotalk-ar-2step/Wan2.1_VAE.pth
```

在6、7号卡上按顺序运行某个组件的搜索与恢复：

```bash
cd /data/nvme6/gushiqiao/codes/latest/vae/LightX2V/lightx2v_train

GPU_LIST=6,7 bash scripts/run_wan21_vae_prune.sh encoder all
```

要训练decoder，将`encoder`改为`decoder`。不要在相同两张卡仍有任务时同时启动两者。也可分开运行：

```bash
GPU_LIST=6,7 bash scripts/run_wan21_vae_prune.sh decoder search
GPU_LIST=6,7 bash scripts/run_wan21_vae_prune.sh decoder recover
```

`GPU_LIST`可改为其他空闲卡列表，DDP进程数按列表长度确定；配置文件不硬编码两卡。默认输出分别为：

```text
output_train/wan21_encoder_prune_search_keep3_ddp
output_train/wan21_encoder_prune_recover_keep3_ddp
output_train/wan21_decoder_prune_search_keep5_ddp
output_train/wan21_decoder_prune_recover_keep5_ddp
```

重复执行会自动续训各自目录。需要新实验时设置独立输出目录；不要把encoder/decoder或search/recover指向同一个目录：

- `WAN_VAE_PATH`：原版 `Wan2.1_VAE.pth`。
- `WAN_VAE_PYTHON`：Python解释器。
- `WAN_VAE_ENCODER_SEARCH_OUTPUT` / `WAN_VAE_ENCODER_RECOVER_OUTPUT`：encoder输出。
- `WAN_VAE_DECODER_SEARCH_OUTPUT` / `WAN_VAE_DECODER_RECOVER_OUTPUT`：decoder输出。
- `WAN_VAE_ENCODER_SELECTION` / `WAN_VAE_DECODER_SELECTION`：该组件搜索导出的 `export/kept_layers.json`。

恢复脚本要求结构文件及同目录的 `wan21_pruned_encoder.safetensors` 或 `wan21_pruned_decoder.safetensors` 同时存在。不能用H3的导出、另一组件的导出或临时search checkpoint代替。

## 文件与验证

| 组件 | 搜索配置 | 恢复配置 |
| --- | --- | --- |
| Encoder | [search keep3](../configs/train/vae/wan21_encoder_prune_search_keep3_ddp.yaml) | [recover keep3](../configs/train/vae/wan21_encoder_prune_recover_keep3_ddp.yaml) |
| Decoder | [search keep5](../configs/train/vae/wan21_decoder_prune_search_keep5_ddp.yaml) | [recover keep5](../configs/train/vae/wan21_decoder_prune_recover_keep5_ddp.yaml) |

[启动脚本](../scripts/run_wan21_vae_prune.sh)、[剪枝模型](../lightx2v_train/model_zoo/native/wan/pruned_vae.py)、[模型封装](../lightx2v_train/model_zoo/wan/wan_pruned_vae.py)、[蒸馏损失](../lightx2v_train/model_zoo/wan/capability_adapters/wan_vae_distillation_capability.py)。

已完成真实Wan权重的键与形状加载检查、缩小宽度模型与原版因果流式实现的数值对齐、教师冻结/学生梯度测试，以及两进程CPU/Gloo的搜索、导出、恢复、预览和断点续训测试。包含H3回归的110项VAE测试通过。

本次没有启动正式GPU训练；默认裁剪下的GPU显存峰值、训练速度和最终重建质量尚未实测。发布的4份配置使用DDP，不是已经验证过的FSDP配置。
