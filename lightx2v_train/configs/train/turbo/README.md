# H3 Turbo8 训练

这套独立的 `training.method: turbo` 实现把 [TurboT2VA](https://arxiv.org/abs/2608.24674) 的 dCM → sCM → sCM+DMD 课程迁移到 MiniMax-H3，支持 Ref2AV 和 T2AV/I2AV/FL2AV 三任务混训。训练从 H3 基础权重开始，不加载 Larry LoRA。

TurboT2VA 仓库公开的 LTX-2 部分目前没有完整训练器。这里的目标函数、TrigFlow 参数化和采样器依据同仓库公开的 Wan rCM 训练代码，固定版本为 [`thu-ml/TurboDiffusion@e3d61360b0eeab60c5cd09ff7d6c1020cccc80b2`](https://github.com/thu-ml/TurboDiffusion/tree/e3d61360b0eeab60c5cd09ff7d6c1020cccc80b2)，主要参照 `turbodiffusion/rcm/models/t2v_model_distill_rcm.py`。这是 H3 适配实现，包含用户指定的数据、8 步目标、LoRA 和 FSDP 配置；不代表复现了尚未公开的 LTX-2 全部训练细节。

## 配置与计数

| 项目 | Ref2AV | FL2V 三任务 |
|---|---|---|
| 配置 | `minimax_h3_ref2av_omni_imageonly_turbo8_shift12_3_r128_a128_adaln16_fsdp32.yaml` | `minimax_h3_fl2av_3task_turbo8_shift6_3_r128_a128_adaln16_fsdp32.yaml` |
| 模型组件 | `transformer_ref/` | `transformer/` |
| CM 数据 | 同一份 Ref2AV clean AV cache | 独立 T2AV teacher clean AV，9424条 |
| DMD/fake 数据 | 同一份 Ref2AV cache | 原 T2AV、I2AV、FL2AV condition view，59684条 |
| video/audio shift | 12/3 | 6/3 |
| 全局 batch | 32 | 32 |
| 并行 | 4 节点 × 8 GPU，FSDP32，SP1 | 相同 |

`max_train_iters: 7000` 统计 **student optimizer 更新**：2000 次 dCM、500 次 sCM、4500 次 sCM+DMD。前两个阶段不更新 fake；联合阶段每次 student 更新后做 **4 次 fake 更新**。公开代码的 `student_update_freq=5` 表示一组中 1 次 student 加 4 次 fake。

因此默认完整训练是 7000 次 student、18000 次 fake，共 **25000 次 optimizer 更新**。Ref 单数据源沿用每角色更新取一批的布局，共25000个全局 microbatch，32卡累积1时约80万样本条次。

FL2V 双数据源下，所有7000次student更新都读取一批 T2AV clean AV 用于 dCM/sCM；联合阶段还分别读取一批三任务条件用于student DMD，以及4批三任务条件用于fake。因此 CM为7000批、online为4500×5=22500批，共 **29500个全局microbatch、94.4万样本条次**。这些是采样次数，不等于不同视频数量；CM与online分别保存数据cursor与fingerprint。

这与原 official PDMD 的计数不同：PDMD 的 `iter` 统计单个角色更新，5 fake 后1 student，且 student 复用 preceding critic 轨迹。Turbo 的 `iter` 统计student更新，联合阶段fake更新另计，数据消耗按具体的单源或双源布局分别统计。

Ref 仍使用 `count_random`：同一全局 batch 的参考图数量相同，按现有的1..6参考图桶平衡采样，横竖构图自然采样，余数轮转，不承诺每个 sampler epoch 完整覆盖数据。FL2V 的online数据仍让三任务数量以11/11/10轮转，按绝对 microbatch cursor 连续路由，并用 route loss weights 平衡任务；独立CM loader只shuffle T2AV clean AV，不套三任务路由。CM使用连续shuffle数据流，`drop_last=false`：跨数据epoch的全局batch延续下一轮数据，不padding补齐、不丢弃尾部。

## 数值与模型设置

- 常规 attention/FFN LoRA 为 **rank128、alpha128**；50层主干 `adaln_proj.linear` 和输出 `norm_out.linear` 为 **rank16、alpha16**。Q/K/V 使用独立 adapter，与 Larry 的融合 QKV adapter 参数化不同。
- student LoRA master 参数 FP32、冻结基础权重 BF16；fake 全参数 FP32 master；teacher 冻结 BF16。FSDP 使用 BF16 compute、FP32 reduction。
- teacher 直接使用条件预测，`guidance_scale=1`。dCM/sCM 不要求 CFG。
- RF clean-minus-noise velocity 显式转换为 TrigFlow；时间、三角系数和归一化目标保持 FP64 数值计算，网络按上述 BF16/FP32精度运行。
- 时间分布使用公开 rCM 类的默认值：generator `mean=-0.8,std=1.6`，score/fake `mean=0,std=1.6`。Wan 的实验配置曾使用不同采样参数；这组默认值是显式的 H3 适配选择，并非声称来自未公开的 LTX-2 配方。两模态共享基础时间，再分别应用 Ref12/3 或 FL6/3 shift，且只应用一次。
- sCM 使用视频、音频及时间的 **真实联合 JVP**；使用 forward AD 和支持 JVP 的 attention。没有有限差分替代；若模型内切断 tangent，会报错。
- student 默认 EMA decay0.999，每次 student optimizer 更新后维护。checkpoint 同时包含可继续训练的原始 student 和 EMA；EMA 推理导出在 `student_ema/`。
- student LR `2e-5`，fake LR `4e-6`，AdamW betas `(0,0.999)`、weight decay0.01。student LR 参考论文，fake/student LR比例0.2沿用公开 rCM，属于 H3 适配默认值。

训练中的生成分支每次固定走完 **8步 x0预测→重新加噪**采样。这是用户8步目标下的设计选择，和公开 Wan rCM 源码在 `1..max_steps` 间循环生成步数的策略不同，也和 PDMD 的确定性 RF Euler rollout 不同。对应实现是 `lightx2v_train.trainers.turbo.sampler.sample_turbo_latents`；导出 adapter 后应配套该 Turbo 采样方式评估，不能把旧 PDMD Euler 推理配置视为相同路线。

## 数据要求和只读预检

Turbo 的 dCM/sCM 需要 **clean video latent 和 clean audio latent**；用于CM的每条metadata必须包含有效的 `condition_path`、`video_latent_path`、`audio_latent_path`。

当前FL2V配置使用两个独立数据源：`data.cm` 指向 `H3_CM_CACHE`，默认选择 `/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_teacher_latents_5s/metadata.jsonl` 中的9424条T2AV；`data.train` 继续使用原三任务condition-only view。I2AV和FL2AV接受online DMD监督，没有对应的offline CM目标。

CM manifest中原来的绝对cache路径通过 `data.cm.path_mapping` 从 `/data/nvme6/gushiqiao` 映射到 `/mnt/lm_data_afs/gushiqiao`。只映射condition和两类latent路径；相对路径仍按manifest目录解析，原始媒体provenance保持不动，不改写旧metadata。同步数据时需要：

- `minimax_h3_teacher_latents_5s/metadata.jsonl` 及其 `video_latents/`、`audio_latents/`。
- manifest引用的 `minimax_h3_dmd_cache_5s_768p/t2av/conditions/`。

新机器上尚未搬齐这些文件时，先完成同步再运行预检。CM路径映射只解决路径位置，不会生成缺失latent。

启动器只用CPU加载真正的 `TurboConfig` schema和共享CM manifest校验，不导入Torch、不反序列化latent。它检查CM全部行的3类payload、T2AV身份、124帧768p几何和可选的shape/normalized声明。FL2V online view只要求condition文件、三任务双方向及每任务至少32条，不要求clean AV。Ref旧单源配置继续检查clean AV与对应 `.complete.json` 的计数、fingerprint和SHA256。

在下面任一命令末尾追加 `--dry-run`，只做预检并打印 torchrun 命令，不创建训练输出目录、不启动 GPU 进程。用 `H3_TURBO_CONFIG` 可选择自定义 Turbo YAML；旧 `H3_PDMD*` 配置或输出变量不会选择这条新方法。

## ACP 32卡启动命令

先同步完整代码和配置到新机器的 LightX2V 根目录。在申请的 **4个ACP worker上执行同一条任务命令**；ACP需向四个worker提供相同的 `MASTER_ADDR`、`MASTER_PORT`。两个训练任务使用独立的输出目录和 rendezvous ID。

Ref2AV：

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

H3_CODE_ROOT="$PWD" \
H3_PYTHON="$VIRTUAL_ENV/bin/python" \
H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3 \
H3_REF2AV_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl \
H3_REF2AV_TURBO_OUTPUT=/mnt/lm_data_afs/gushiqiao/outputs/h3_ref2av_turbo8_shift12_3_r128a128_adaln16_32gpu_20261028 \
H3_RDZV_ID=h3_ref2av_turbo8_shift12_3_r128a128_adaln16_32gpu_20261028 \
KERNELS_CACHE=/mnt/lm_data_afs/gushiqiao/cache/kernels \
bash lightx2v_train/scripts/run_minimax_h3_ref2av_turbo8_adaln16_fsdp32_32gpu_acp.sh
```

FL2V 三任务：

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

H3_CODE_ROOT="$PWD" \
H3_PYTHON="$VIRTUAL_ENV/bin/python" \
H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3 \
H3_FL2AV_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_dmd_cache_5s_768p/task_aware_view_3task_portrait \
H3_CM_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_teacher_latents_5s/metadata.jsonl \
H3_FL2AV_TURBO_OUTPUT=/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_3task_turbo8_shift6_3_r128a128_adaln16_cm_t2av_9424_32gpu_20261028 \
H3_RDZV_ID=h3_fl2av_3task_turbo8_shift6_3_r128a128_adaln16_cm_t2av_9424_32gpu_20261028 \
KERNELS_CACHE=/mnt/lm_data_afs/gushiqiao/cache/kernels \
bash lightx2v_train/scripts/run_minimax_h3_fl2av_turbo8_adaln16_fsdp32_32gpu_acp.sh
```

## 保存与续训

默认每50次student更新保存一次，Ref保留5份、FL2V保留2份。FL双数据源输出名含 `cm_t2av_9424`，使用全新目录；PDMD及旧单数据源Turbo checkpoint缺少这套双游标语义，不能直接续成双数据源训练。

`resume.auto_resume: true` 只选择带 `turbo.complete.json` 的完整 checkpoint，并验证 rank/world size、配置 recipe、student/fake计数和绝对数据 cursor。recipe 还保存按加载顺序排列的全部 metadata 记录及 base_dir 的稳定 SHA；同一路径换序或修改记录也会拒绝续训。Turbo 不允许 `resume.allow_distribution_matching_transition` 绕过一致性检查；关闭 auto-resume 时也要求没有既有 checkpoint，避免覆盖旧实验。

checkpoint 包含 student、fake、各 optimizer/scheduler 和 student EMA，只在一组 student及其全部fake更新完成后标记完整。新完整 marker 写入后才删除超出保留数量的旧完整 Turbo checkpoint，避免保存中断时提前失去上一份可恢复状态。

本机的 CPU 小模型、公式和预检验证不能替代真实32卡、124帧768p任务的显存、吞吐及画质验证；命令准备完成不表示已经在远端启动训练。
