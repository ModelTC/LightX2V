# Variable Resolution Inference

## Overview

Variable resolution inference is a technical strategy for optimizing the denoising process. It improves computational efficiency while maintaining generation quality by using different resolutions at different stages of the denoising process. The core idea of this method is to use lower resolution for coarse denoising in the early stages and switch to normal resolution for fine processing in the later stages.

## Technical Principles

### Multi-stage Denoising Strategy

Variable resolution inference is based on the following observations:

- **Early-stage denoising**: Mainly handles coarse noise and overall structure, requiring less detailed information
- **Late-stage denoising**: Focuses on detail optimization and high-frequency information recovery, requiring complete resolution information

### Resolution Switching Mechanism

1. **Low-resolution stage** (early stage)
   - Downsample the input to a lower resolution (e.g., 0.75x of original size)
   - Execute initial denoising steps
   - Quickly remove most noise and establish basic structure

2. **Normal resolution stage** (late stage)
   - Upsample the denoising result from the first step back to original resolution
   - Continue executing remaining denoising steps
   - Restore detailed information and complete fine processing

### U-shaped Resolution Strategy

If resolution is reduced at the very beginning of the denoising steps, it may cause significant differences between the final generated video and the video generated through normal inference. Therefore, a U-shaped resolution strategy can be adopted, where the original resolution is maintained for the first few steps, then resolution is reduced for inference.

## Usage

The config files for variable resolution inference are located [here](https://github.com/ModelTC/LightX2V/tree/main/configs/changing_resolution)

You can test variable resolution inference by specifying --config_json to the specific config file.

You can refer to the scripts [here](https://github.com/ModelTC/LightX2V/blob/main/scripts/changing_resolution) to run.

### Example 1:
```
{
    "infer_steps": 50,
    "changing_resolution": true,
    "resolution_rate": [0.75],
    "changing_resolution_steps": [25]
}
```

This means a total of 50 steps, with resolution at 0.75x original resolution from step 1 to 25, and original resolution from step 26 to the final step.

### Example 2:
```
{
    "infer_steps": 50,
    "changing_resolution": true,
    "resolution_rate": [1.0, 0.75],
    "changing_resolution_steps": [10, 35]
}
```

This means a total of 50 steps, with original resolution from step 1 to 10, 0.75x original resolution from step 11 to 35, and original resolution from step 36 to the final step.

Generally, if `changing_resolution_steps` is [A, B, C], the denoising starts at step 1, and the total number of steps is X, then the inference process will be divided into four segments.

Specifically, these segments are (0, A], (A, B], (B, C], and (C, X], where each segment is a left-open, right-closed interval.
