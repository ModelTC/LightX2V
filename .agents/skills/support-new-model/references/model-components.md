# LightX2V 模型实现参考

复用时从适用任务的脚本和配置走到 runner、model、weights/infer 和 scheduler。下面的差异影响权重选择、张量边界或数值对齐。

## Wan

- `wan2.1`、`wan2.2`（dense）和 `wan2.2_moe` 在同一个模块注册不同 runner。MoE 用 `MultiModelStruct` 按去噪阶段选择高噪声或低噪声 DiT。检查实际任务的 T5、CLIP/VAE 条件、帧数与 patch 网格。 [runner](../../../../lightx2v/models/runners/wan/wan_runner.py) · [配置 loader](../../../../lightx2v/models/runners/wan/config.py)
- `WanModel` 由 `WanPreInfer` 将 latent 转为 `[tokens, dim]`，常规注意力收 `[tokens, heads, head_dim]`。输出 head 位于 `transformer_infer`，`WanPostInfer` 用原网格 unpatchify。CFG 对两个预测做线性合成，`WanScheduler` 的常规路径用 UniPC，特定任务还保留条件 latent。 [model](../../../../lightx2v/models/networks/wan/model.py) · [pre-infer](../../../../lightx2v/models/networks/wan/infer/pre_infer.py) · [scheduler](../../../../lightx2v/models/schedulers/wan/scheduler.py)
- 普通 T2V 启动可参照同一组脚本和配置。 [脚本](../../../../scripts/wan/run_wan_t2v.sh) · [配置](../../../../configs/wan/wan_t2v.json)

## Qwen-Image

- `qwen_image` 在启动时绑定 `t2i` 或 `i2i`，DiT 加载来自 `transformer/` 的权重。`i2i` 的 `image_path` 是列表。VAE 编码参考图后，经 packed latent 作为条件拼到生成 token 序列，预测须裁回生成区域。 [runner](../../../../lightx2v/models/runners/qwen_image/qwen_image_runner.py) · [Spec](../../../../lightx2v/models/runners/qwen_image/request_spec.py) · [model](../../../../lightx2v/models/networks/qwen_image/model.py)
- Scheduler 接收 `InputInfo`，使用 packed latent 和各分支的图文 RoPE，复用 Diffusers 的 Flow Match Euler solver。图文双流在 pre-infer 去掉外层单样本维，joint attention 后各自更新，post-infer 恢复预测的外层维度。CFG 合成后用条件预测的范数归一化。 [scheduler](../../../../lightx2v/models/schedulers/qwen_image/scheduler.py) · [pre-infer](../../../../lightx2v/models/networks/qwen_image/infer/pre_infer.py) · [transformer infer](../../../../lightx2v/models/networks/qwen_image/infer/transformer_infer.py) · [post-infer](../../../../lightx2v/models/networks/qwen_image/infer/post_infer.py)
- 编辑入口可参照 2511 的脚本和配置。Qwen-Image-2.1 有独立的 runner、scheduler 和网络，接入该版本时沿相应目录核对。 [编辑脚本](../../../../scripts/qwen_image/qwen_image_i2i_2511.sh) · [编辑配置](../../../../configs/qwen_image/qwen_image_i2i_2511.json) · [2.1 runner](../../../../lightx2v/models/runners/qwen_image_21/qwen_image_21_runner.py)

## MiniMax-H3

- `--model-variant fl2av` 加载 `transformer/`，仓库示例用于 `t2av`、`i2av`、`l2av`、`fl2av`。`--model-variant ref2av` 加载 `transformer_ref/`，用于 `ref2av`。runner 按 `input_info.task` 选条件编码，权重分区在启动时确定。 [配置 loader](../../../../lightx2v/models/runners/minimax_h3/config.py) · [runner](../../../../lightx2v/models/runners/minimax_h3/minimax_h3_runner.py) · [FL2AV 脚本](../../../../scripts/minimax_h3/run_minimax_h3_fl2av.sh) · [Ref2AV 脚本](../../../../scripts/minimax_h3/run_minimax_h3_ref2av.sh)
- 文本、音频、视频和条件行共享 packed 序列，按 layout 对齐 token tag、位置、模态索引和每行 timestep。guidance-distilled 路径每步执行一次 DiT 前向，写入 `video_noise_pred` 与 `audio_noise_pred`。scheduler 对两种 data-ward velocity 使用各自的 sigma，并只更新目标行。解码前去掉条件行。 [packing](../../../../lightx2v/models/networks/minimax_h3/packing.py) · [参考输入 packing](../../../../lightx2v/models/networks/minimax_h3/packing_ref2av.py) · [model](../../../../lightx2v/models/networks/minimax_h3/model.py) · [scheduler](../../../../lightx2v/models/schedulers/minimax_h3/scheduler.py)
- 使用现有 CPU offload 示例时，按脚本准备与权重变体、配置匹配的 AdaLN cache。 [配置](../../../../configs/minimax_h3/minimax_h3.json) · [模型运行说明](../../../../scripts/minimax_h3/README_zh.md)

## 并行、offload 与缓存

- 需要分布式时，从 `parallel` 中的大小和 `init_parallel()` 取得进程组及运行时 flags。按实际 token/head/weight 分片核对 padding、collective、RoPE/mask 和 gather 后的裁切。CFG 并行还要核对每个 rank 的条件编码和合成顺序。 [并行初始化](../../../../lightx2v/utils/parallel.py)
- 改动 block 布局或执行顺序时，检查 offload 的权重 schema、staging slot、加载顺序与同步，复用匹配的管理器。按实际任务查阅 [CPU block offload](../../support-cpu-block-offload/SKILL.md)、[compile](../../support-model-compile/SKILL.md) 或 [warmup](../../support-model-warmup/SKILL.md)。
- 接入自回归 KV cache 时，以模型的 prefill、去噪和 commit 流程确定状态归属，核对 RoPE 与缓存读写顺序、历史窗口、分支和 step 是否共享缓存。符合契约时复用 `KVCacheManager`，由持有状态的组件创建与重置。验证跨 chunk、窗口滚动及下一次请求。 [现有 AR 实现](../../../../lightx2v/common/kvcache/AR_IMPLEMENTATIONS.md) · [KVCacheManager](../../../../lightx2v/common/kvcache/manager.py)
