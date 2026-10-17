# EdenLive block offload

Run `bash scripts/edenlive/offload/run_edenlive_block_offload.sh` from the repository root. The launcher derives a temporary configuration from `configs/edenlive/wan21_14b_few_step.json`, enabling block offload, compile and warmup for Wan2.1 14B AR I2V with 121 frames. It does not modify the base configuration.

Set `MODEL_PATH`, `SAMPLE_DIR`, `CUDA_VISIBLE_DEVICES` and `SAVE_RESULT_PATH` for the destination host. The sample directory must contain `prompts.json` with `staticPrompt`, `cover.png` and `recording.mp4`. Use the retained base JSON files to derive other model variants, and update the matching `--model_cls`.

The two single-card realtime profiles are documented in `configs/edenlive/realtime/README.md`.
