# EdenLive block offload

在仓库根目录运行 `bash scripts/edenlive/offload/run_edenlive_block_offload.sh`。脚本从 `configs/edenlive/wan21_14b_few_step.json` 派生临时配置，为 Wan2.1 14B AR I2V 开启 block offload、compile 和 warmup，生成 121 帧，不修改原始配置。

可通过 `MODEL_PATH`、`SAMPLE_DIR`、`CUDA_VISIBLE_DEVICES`、`SAVE_RESULT_PATH` 指定本机路径和设备。样本目录需包含带有 `staticPrompt` 的 `prompts.json`、`cover.png` 和 `recording.mp4`。其他模型可从保留的顶层 JSON 派生，并同步调整 `--model_cls`。

两个单卡实时配置见 `configs/edenlive/realtime/README.md`。
