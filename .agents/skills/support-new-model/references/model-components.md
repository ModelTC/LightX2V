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

## 混合精度

- Wan 2.1 上游的输出头要求调制条件 `e` 为 FP32，在归一化后做缩放和平移。LightX2V 的 `infer_non_blocks()` 用 `sensitive_layer_dtype` 执行这段乘加，在输出投影 `weights.head.apply()` 前转回 `infer_dtype`。接入其他架构时，按其上游前向代码确定各自的控制位置。 [Wan 2.1 上游输出头](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L343-L346) · [LightX2V 输出头](../../../../lightx2v/models/networks/wan/infer/transformer_infer.py)
- 某段计算的浮点参数也需要升精度时，使用共享 loader 的模型可将对应 checkpoint 键的匹配片段加入 `self.sensitive_layer`。例如 Wan 的时间嵌入在加载权重和进入投影时都使用敏感精度。 [Wan 权重选择](../../../../lightx2v/models/networks/wan/model.py) · [共享 loader](../../../../lightx2v/models/networks/base_model.py) · [时间嵌入投影](../../../../lightx2v/models/networks/wan/infer/pre_infer.py)
- 精度读取函数会缓存进程内的结果。以支持 FP32 敏感计算的路径为例，可固定 `DTYPE=BF16`，分别以 `SENSITIVE_LAYER_DTYPE=None` 和 `SENSITIVE_LAYER_DTYPE=FP32` 启动短进程，用同一小输入观察目标运算的精度和下游输入精度。 [环境变量读取](../../../../lightx2v/utils/envs.py) · [启动默认值](../../../../scripts/base/base.sh)

## 并行、offload 与缓存

- 多卡推理复用 `init_parallel()` 按 `parallel` 配置建立的进程组。以 Wan 双卡序列并行为例，启动进程数与 `seq_p_size` 一致，模型在 transformer 前按 rank 切分 token，之后 gather。按实际 token/head/weight 分片核对 padding、collective、RoPE/mask 和输出裁切。CFG 并行还要核对每个 rank 的条件编码和合成顺序。 [并行初始化](../../../../lightx2v/utils/parallel.py) · [Wan 双卡脚本](../../../../scripts/wan/run_wan_t2v_parallel.sh) · [序列分片与汇集](../../../../lightx2v/models/networks/wan/model.py)
- CPU block offload 可沿 Wan 的配置追到 transformer 的逐 block 执行循环，核对 CPU 权重来源、设备 staging slot、加载顺序与同步。改动 block 布局时，检查 slot 与各类 block 的权重形状和 dtype 是否匹配。 [Wan block 配置](../../../../configs/offload/block/wan_t2v_block.json) · [offload 执行循环](../../../../lightx2v/models/networks/wan/infer/offload/transformer_infer.py) · [CPU block offload](../../support-cpu-block-offload/SKILL.md)
- 接入自回归 KV cache 时，以模型的 prefill、去噪和 commit 流程确定状态归属，核对 RoPE 与缓存读写顺序、历史窗口、分支和 step 是否共享缓存。符合契约时复用 `KVCacheManager`，由持有状态的组件创建与重置。验证跨 chunk、窗口滚动及下一次请求。 [现有 AR 实现](../../../../lightx2v/common/kvcache/AR_IMPLEMENTATIONS.md) · [KVCacheManager](../../../../lightx2v/common/kvcache/manager.py)

## 编译与启动预热

- Wan T2V 的启动 JSON 已启用 `use_compile` 和 `warmup`，可直接用于全开组。全关基线使用同一脚本、checkpoint、输入和 seed，只将这两个值改为 `false`。 [启动脚本](../../../../scripts/wan/inference/run_wan_t2v_compile.sh) · [启动配置](../../../../configs/wan/wan_t2v_compile.json)
- `WanTransformerInfer` 初始化时调用 `init_compile()`，执行每个 block 时调用 `run_block()`。公共实现按 block 缓存编译函数，命中时还核对权重对象身份。复用 offload 缓冲区时，若计算依赖逻辑层号，将层号计入缓存键。 [Wan block 循环](../../../../lightx2v/models/networks/wan/infer/transformer_infer.py) · [编译分派与缓存](../../../../lightx2v/common/transformer_infer/transformer_infer.py)
- `BaseRunner` 在最外层 `init_modules()` 成功后调用一次 `warmup()`。Wan 的 `run_warmup()` 遍历代表性形状，准备条件输入，运行采样步与 VAE 解码，并清理每轮的临时状态。对多卡或 offload 组合，用正式请求的各 rank 输入形状、执行分支和权重调度核对预热覆盖范围。 [初始化钩子](../../../../lightx2v/models/runners/base_runner.py) · [Wan 预热实现](../../../../lightx2v/models/runners/wan/wan_runner.py)
