# Changing Resolution Inference

## Overview

Changing resolution inference is a technical strategy for optimizing the denoising process. It improves computational efficiency while maintaining generation quality by using different resolutions at different denoising stages. The core idea is to use lower resolution for rough denoising in the early stages of the denoising process, then switch to normal resolution for fine-tuning in the later stages.

## Technical Principles

### Phased Denoising Strategy

Changing resolution inference is based on the following observations:
- **Early-stage denoising**: Mainly processes rough noise and overall structure, doesn't require excessive detail information
- **Late-stage denoising**: Focuses on detail optimization and high-frequency information recovery, requires complete resolution information

### Resolution Switching Mechanism

1. **Low Resolution Stage** (Early stage)
   - Downsample the input to lower resolution (e.g., 0.75 of original size)
   - Execute initial denoising steps
   - Quickly remove most noise and establish basic structure

2. **Normal Resolution Stage** (Late stage)
   - Upsample the denoising result from the first step back to original resolution
   - Continue executing remaining denoising steps
   - Recover detail information and complete fine-tuning


## Usage

The config files for changing resolution inference are available [here](https://github.com/ModelTC/LightX2V/tree/main/configs/changing_resolution)

By specifying --config_json to the specific config file, you can test changing resolution inference.

You can refer to [this script](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan/run_wan_t2v_changing_resolution.sh).
