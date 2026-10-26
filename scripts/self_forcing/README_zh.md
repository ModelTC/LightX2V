# Self-Forcing (Wan2.1-T2V-1.3B)

本文说明如何在 LightX2V 中运行 Self-Forcing Wan2.1 1.3B 文生视频模型。

## 准备模型

按 LightX2V [快速开始](../../docs/ZH_CN/source/getting_started/quickstart.md)安装环境，下载 Wan2.1 1.3B 原始权重与 Self-Forcing checkpoint：

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./wan_models/Wan2.1-T2V-1.3B
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

当前 RTX 5090 的两份配置均使用 81 帧、480×832、16 FPS 的请求尺寸和 LightTAE 解码器：

- [BF16 配置](../../configs/self_forcing/wan_t2v_sf_bf16_lighttae_5090.json)
- [FP8-FP16 配置](../../configs/self_forcing/wan_t2v_sf_fp8_f16_accum_lighttae_5090.json)

FP8 配置使用已转换的 DiT/T5 权重。迁移到其他机器时，应在复制后的 JSON 中修改权重、tokenizer 和解码器的绝对路径。两份配置都开启 Sage、warmup、DiT compile/CUDA Graph、流式 LightTAE 与顶层异步解码；KV offload 关闭。`vae_use_compile` 与 DiT 的 `use_compile` 分别控制解码器和 DiT 编译。

## 运行

从仓库根目录运行，并用 `MODEL_PATH` 指向已下载的 Wan 权重。脚本会自动定位仓库，并分别选择 BF16、FP8 配置：

```bash
export MODEL_PATH="$PWD/wan_models/Wan2.1-T2V-1.3B"
bash scripts/self_forcing/run_wan_t2v_sf.sh
bash scripts/self_forcing/run_wan_t2v_sf_fp8.sh
```

如需指定复制后的 JSON，在运行脚本时设置 `CONFIG_PATH`。原始 Self-Forcing checkpoint 路径对应 `dit_original_ckpt`。自注意力在写入 rolling ring 前完成 K 的 RoPE；Sage 仅保存历史编码 K/V，并增量更新新写入的 slot。缓存与 FlashAttention 的具体路径见 [AR KV 实现对照](../../lightx2v/common/kvcache/AR_IMPLEMENTATIONS.md)。
