# Omni Ref2AV DeCFG teacher：两台机器，各 8 卡

这套流程从已有 Omni raw100k 中选择 5,000 条：仅非空 `enhanced_prompt`、仅图片参考、实际参考数 2/3/4/5/6 各 1,000 条。seed=42 按稳定 source ID 的 SHA256 排序采样；每桶交替分给两台机器，因此每台恰好每桶 500 条、共 2,500 条。两台各运行一个 TP=2、SP=4 的服务，不跨机组网；两台均可使用本机 `127.0.0.1:8000`。每条仅保存 29 步推理完成后的最终 clean video/audio latents 和训练 conditioning，不保存中间状态。

尺寸沿用已有训练 condition cache：按 `(source_index, source_id)` 精确连接 `latent_match124_bf16/metadata.jsonl`，使用对应的 `target_height/target_width`，仅接受 `[768,1344]` 或 `[1344,768]`、124 帧。原始记录即使写 4K，也不会生成 4K。`source_index` 是忽略空行后的零基记录序号，与既有 cache builder 一致。ID 优先取 `sample_id/metadata_id/id`，缺失时沿用 cache builder 的 source_index 字符串。

参考顺序沿用 cache builder：优先 `references`、`actual_ordered_references`、`ordered_references`，否则 `reference_images`。支持 `kind/modality/type/role` 与 `local_path/path/media_path/url/rel_path`；相对路径拼接 media root。显式 order/reference_index 必须已经连续，拒绝静默重排。实际列表长度决定分桶，raw 中的旧 `ref_image_count` 不参与分桶；已缓存的 count 必须与实际列表匹配。不重跑 prompt enhancement，不使用原始 prompt 兜底，不自动训练。

## 共同环境

先把本次代码放到两机均可访问的目标仓库，直接使用当前 shell 中的 LightX2V 推理 Python。两台机器执行相同的环境设置：

```bash
export H3_CODE_ROOT=/mnt/devsft_afs_2/gushiqiao/outputs/LightX2V
export H3_PYTHON=python
export H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-DeCFG-score
export H3_RAW_JSONL=/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/omni_r2v_image_only_100000_ir.jsonl
export H3_CACHE_METADATA=/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl
export H3_MEDIA_ROOT=/mnt/lm_data_afs/gushiqiao/datasets/omini-r2v
export H3_OUTPUT_DIR=/mnt/devsft_afs_2/gushiqiao/outputs/data_new2/omni_ref_teacher29_decfg_5k
export H3_CONFIG_JSON="$H3_CODE_ROOT/configs/minimax_h3/minimax_h3_ref2av_teacher29_tp2_sp4.json"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PLATFORM=cuda
cd "$H3_CODE_ROOT"
mkdir -p "$H3_OUTPUT_DIR/logs"
```

启动前验证 `MiniMax-H3-DeCFG-score/decfg_merged.json` 是已完成的 `lightx2v_h3_score_decfg`，含已经合并的 `transformer_ref`。验证 config/index SHA256 和权重 shard 大小；不会回退到普通 H3 base。配置固定 29 步、124 帧、24 fps、CFG=false、CPU offload=false、VAE offload=false、reference resize=match、scheduler=reference_blend。

DeCFG 合并文件可以包含 FP32 的注意力/FFN 权重。H3 loader 在 TP 切片后将这些普通线性层转换为 BF16，与激活一致；保留归一化权重的原始精度，输入、时间和输出投影仍按原实现使用 FP32。磁盘上的合并权重不修改，scheduler 状态和导出的最终 AV latents 仍为 FP32。

每次启动都会用**同一配置、同一个已合并 transformer_ref**验证 AdaLN。缺失时调用仓库现有 `tools/cache_minimax_h3_adaln/builder.py` 构建；不合法的同名 cache 先保留为 `.invalid.*` 再重建。配置的 `adaln_cache_dir` 保持原值 `~/.cache/lightx2v/adaln`，不会使用普通 base 的 cache。现有 cache resolver 会把 checkpoint 路径、大小和 mtime 纳入指纹，因此不同权重分开存储。AdaLN 预处理使用第一张可见 GPU；之后才启动 8 卡推理。

## 先跑 1 条离线 smoke

在尚未运行批量服务的机器上：

```bash
bash scripts/minimax_h3/teacher_data/smoke.sh --node-index 0 --sample-index 0 \
  > "$H3_OUTPUT_DIR/logs/smoke-launch.log" 2>&1
tail -f "$H3_OUTPUT_DIR/logs/smoke-launch.log"
```

这是 `python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 -m lightx2v.infer`，不启动 HTTP server。默认对同一选中样本导出最终 AV latents 并解码 MP4/音频；加 `--no-export` 可只检查解码。样本的提示词、图片顺序和缓存画布通过 subprocess argv 直接传递，不依赖 shell 转义。输出在 `smoke/node0/<时间戳>/sample.mp4`，旁边是 `sample.json`、`command.json`、`final_latents/`。具体 infer 日志见启动日志打印的 `logs/node0/smoke.*.log`。

## 两台同时批量采集

机器 0：

```bash
nohup bash scripts/minimax_h3/teacher_data/run_node.sh 0 \
  > "$H3_OUTPUT_DIR/logs/launch.node0.log" 2>&1 &
echo $!
tail -f "$H3_OUTPUT_DIR/logs/launch.node0.log"
```

机器 1：

```bash
nohup bash scripts/minimax_h3/teacher_data/run_node.sh 1 \
  > "$H3_OUTPUT_DIR/logs/launch.node1.log" 2>&1 &
echo $!
tail -f "$H3_OUTPUT_DIR/logs/launch.node1.log"
```

每个 launcher 自行做 CPU 选样、AdaLN 检查、启动 8 卡服务、逐条提交自己的 2,500 条。批量请求 `teacher_latents_only=true`，跳过视频和音频 VAE 解码。服务在本节点采集完成或失败退出时关闭；对 launcher 发 SIGTERM 也会清理其自己的 torchrun 进程组。无需 MASTER_ADDR、跨机 rendezvous 或不同 HTTP port。若本机 8000 已占用会报错；可以给该节点命令加 `--port 8002`。

健康检查、任务提交和状态轮询使用禁用代理的 HTTP client，直接访问本节点服务，不受 `HTTP_PROXY/http_proxy` 或全局 urllib opener 影响。模型加载完成后启动日志才会打印 `Local service ready`。通信失败时，launcher 先打印 torchrun 当前状态和服务日志末尾，再清理自己的进程组；清理产生的 SIGTERM 不能单独视为原始故障。

`node` 和 `smoke` 在选样、AdaLN 和模型加载之前，按可见 GPU ID 获取本机 `/tmp/lightx2v-h3-teacher-gpu-*.lock` 的非阻塞锁，并持有到子进程清理完成。重复启动或可见 GPU 重叠会立即报错并显示占用 launcher 的 PID；改 HTTP port 不能绕过。锁文件会保留，正常完成或异常退出后锁自动释放，无需删除文件。

选样不足任一桶、选中图片缺失、cache 身份/画布/参考数冲突都会报错。`eligible_counts` 是当前 raw 和 cache 实际连接后的计数；不会把截图中的历史计数硬编码为结果。两台分别原子写入 `selection/node0/` 与 `selection/node1/`，各有完整 `selected.jsonl`、两份分片 `node0.jsonl/node1.jsonl` 和 `manifest.json`。共享 AFS 上不使用 flock；启动时若发现另一台 manifest 则比较 selection/raw/cache SHA256，发现差异立即报错。

## 产物与续跑

每条成功样本在 `samples/imagesN/<sample_key>/<request_fingerprint前16位>/`：

```text
condition.pt
video_latents.pt
audio_latents.pt
manifest.json
completion.json
```

29 次更新完成后，仅保存最终 clean AV latents，均为 FP32。`condition.pt` 保留训练所需 conditioning；`video_latents.pt` 保留 normalized 标记与最终视频张量，`audio_latents.pt` 保存最终音频张量。不生成任何 `state_*.pt`。

`manifest.json` 最后发布，记录 schema=`h3_teacher_final_v1`、export_mode=`final`、num_steps=29、num_states=1、state_paths=[]、final_state_index=29，以及 final_video_sigma=final_audio_sigma=0.0。完整 `video_sigmas/audio_sigmas` 数组仅作为 schedule 元数据保留，同时记录源 ID、提示词、参考顺序、模型与画布。collector 在服务报告 completed，且最终 schema、零噪声端点、schedule 与三个非空张量文件检查通过后才写 `completion.json`。

相同命令可续跑。只有 request/config/model/image 文件身份指纹一致、manifest SHA256 一致、符合最终导出 schema 且三个文件均存在且非空的成功样本才跳过。请求指纹使用 `h3_teacher_request_v2` 并包含 export_mode=`final`；旧轨迹的 receipt/schema 不会被识别为本次成功样本，也不迁移旧数据。未完成/不匹配的目录保留为 `.incomplete.<时间戳>` 后重新生成。张量内容完整校验由训练转换器执行；collector 的续跑检查不逐条重新加载大张量。失败记录在 `failures.nodeN.jsonl`，完整历次记录在 `failure_history.nodeN.jsonl`。请求通信异常或任务超时会停止本机采集，避免在未知任务仍运行时重复提交。

成功索引是 `index.node0.jsonl` 和 `index.node1.jsonl`，仅包含成功样本；每行包含原始 source ID/index、prompt、图片列表、缓存条件路径、画布、seed、export_dir、manifest_path 和三个最终文件路径，可交给训练转换器。`complete.nodeN.json` 记录 expected/completed/failed 与索引 SHA256；完整采集应为两台各 expected=completed=2500、failed=0。采集脚本不发起训练。

CPU 验证命令（在项目 Python 环境中执行）：

```bash
python -m unittest discover -s scripts/minimax_h3/teacher_data -p 'test_*.py' -v
```

覆盖 deterministic 分桶和均分、cache 身份/尺寸/计数、缺图和不足 quota、跨节点 selection 冲突、参考顺序、实际 HTTP schema/CLI 参数、续跑指纹、三个最终文件缺失/空文件、错误模式/端点/schedule、旧轨迹 receipt 拒绝，以及本机代理与 GPU 锁回归。真实 AFS 选样、AdaLN GPU 构建及 8 卡 smoke 需在上述两机环境执行。
