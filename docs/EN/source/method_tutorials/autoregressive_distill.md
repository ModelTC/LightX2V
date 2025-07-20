# Autoregressive Distillation

Autoregressive distillation is a technical exploration in LightX2V. By training distilled models, it reduces inference steps from the original 40-50 steps to **8 steps**, achieving inference acceleration while enabling infinite-length video generation through KV Cache technology.

> ⚠️ Warning: Currently, autoregressive distillation has mediocre effects and the acceleration improvement has not met expectations, but it can serve as a long-term research project. Currently, LightX2V only supports autoregressive models for T2V.

## 🔍 Technical Principle

Autoregressive distillation is implemented through [CausVid](https://github.com/tianweiy/CausVid) technology. CausVid performs step distillation and CFG distillation on 1.3B autoregressive models. LightX2V extends it with a series of enhancements:

1. **Larger Models**: Supports autoregressive distillation training for 14B models;
2. **More Complete Data Processing Pipeline**: Generates a training dataset of 50,000 prompt-video pairs;

For detailed implementation, refer to [CausVid-Plus](https://github.com/GoatWu/CausVid-Plus).

## 🛠️ Configuration Files

### Configuration File

Configuration options are provided in the [configs/causvid/](https://github.com/ModelTC/lightx2v/tree/main/configs/causvid) directory:

| Configuration File | Model Address |
|-------------------|---------------|
| [wan_t2v_causvid.json](https://github.com/ModelTC/lightx2v/blob/main/configs/causvid/wan_t2v_causvid.json) | https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid |

### Key Configuration Parameters

```json
{
  "enable_cfg": false,          // Disable CFG for speed improvement
  "num_fragments": 3,           // Number of video segments generated at once, 5s each
  "num_frames": 21,             // Frames per video segment, modify with caution!
  "num_frame_per_block": 3,     // Frames per autoregressive block, modify with caution!
  "num_blocks": 7,              // Autoregressive blocks per video segment, modify with caution!
  "frame_seq_length": 1560,     // Encoding length per frame, modify with caution!
  "denoising_step_list": [      // Denoising timestep list
    999, 934, 862, 756, 603, 410, 250, 140, 74
  ]
}
```

## 📜 Usage

### Model Preparation

Place the downloaded model (`causal_model.pt` or `causal_model.safetensors`) in the `causvid_models/` folder under the Wan model root directory:
- For T2V: `Wan2.1-T2V-14B/causvid_models/`

### Inference Script

```bash
bash scripts/wan/run_wan_t2v_causvid.sh
```
