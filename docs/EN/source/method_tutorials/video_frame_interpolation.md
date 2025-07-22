# Video Frame Interpolation (VFI)

> **Important Note**: Video frame interpolation is enabled through configuration files, not command-line parameters. Please add a `video_frame_interpolation` configuration block to your JSON config file to enable this feature.

## Overview

Video Frame Interpolation (VFI) is a technique that generates intermediate frames between existing frames to increase the frame rate and create smoother video playback. LightX2V integrates the RIFE (Real-Time Intermediate Flow Estimation) model to provide high-quality frame interpolation capabilities.

## What is RIFE?

RIFE is a state-of-the-art video frame interpolation method that uses optical flow estimation to generate intermediate frames. It can effectively:

- Increase video frame rate (e.g., from 16 FPS to 32 FPS)
- Create smooth motion transitions
- Maintain high visual quality with minimal artifacts
- Process videos in real-time

## Installation and Setup

### Download RIFE Model

First, download the RIFE model weights using the provided script:

```bash
python tools/download_rife.py <target_directory>
```

For example, to download to the location:
```bash
python tools/download_rife.py /path/to/rife/train_log
```

This script will:
- Download RIFEv4.26 model from HuggingFace
- Extract and place the model files in the correct directory
- Clean up temporary files

## Usage

### Configuration File Setup

Video frame interpolation is enabled through configuration files. Add a `video_frame_interpolation` configuration block to your JSON config file:

```json
{
    "infer_steps": 50,
    "target_video_length": 81,
    "target_height": 480,
    "target_width": 832,
    "fps": 16,
    "video_frame_interpolation": {
        "algo": "rife",
        "target_fps": 32,
        "model_path": "/path/to/rife/train_log"
    }
}
```

### Command Line Interface

Run inference using a configuration file that includes VFI settings:

```bash
python lightx2v/infer.py \
    --model_cls wan2.1 \
    --task t2v \
    --model_path /path/to/model \
    --config_json ./configs/video_frame_interpolation/wan_t2v.json \
    --prompt "A beautiful sunset over the ocean" \
    --save_video_path ./output.mp4
```

### Configuration Parameters

In the `video_frame_interpolation` configuration block:

- `algo`: Frame interpolation algorithm, currently supports "rife"
- `target_fps`: Target frame rate for the output video
- `model_path`: RIFE model path, typically "/path/to/rife/train_log"

Other related configurations:
- `fps`: Source video frame rate (default 16)

### Configuration Priority

The system automatically handles video frame rate configuration with the following priority:
1. `video_frame_interpolation.target_fps` - If video frame interpolation is enabled, this frame rate is used as the output frame rate
2. `fps` (default 16) - If video frame interpolation is not enabled, this frame rate is used; it's always used as the source frame rate


## How It Works

### Frame Interpolation Process

1. **Source Video Generation**: The base model generates video frames at the source FPS
2. **Frame Analysis**: RIFE analyzes adjacent frames to estimate optical flow
3. **Intermediate Frame Generation**: New frames are generated between existing frames
4. **Temporal Smoothing**: The interpolated frames create smooth motion transitions

### Technical Details

- **Input Format**: ComfyUI Image tensors [N, H, W, C] in range [0, 1]
- **Output Format**: Interpolated ComfyUI Image tensors [M, H, W, C] in range [0, 1]
- **Processing**: Automatic padding and resolution handling
- **Memory Optimization**: Efficient GPU memory management

## Example Configurations

### Basic Frame Rate Doubling

Create configuration file `wan_t2v_vfi_32fps.json`:

```json
{
    "infer_steps": 50,
    "target_video_length": 81,
    "target_height": 480,
    "target_width": 832,
    "seed": 42,
    "sample_guide_scale": 6,
    "enable_cfg": true,
    "fps": 16,
    "video_frame_interpolation": {
        "algo": "rife",
        "target_fps": 32,
        "model_path": "/path/to/rife/train_log"
    }
}
```

Run command:
```bash
python lightx2v/infer.py \
    --model_cls wan2.1 \
    --task t2v \
    --model_path ./models/wan2.1 \
    --config_json ./wan_t2v_vfi_32fps.json \
    --prompt "A cat playing in the garden" \
    --save_video_path ./output_32fps.mp4
```

### Higher Frame Rate Enhancement

Create configuration file `wan_i2v_vfi_60fps.json`:

```json
{
    "infer_steps": 30,
    "target_video_length": 81,
    "target_height": 480,
    "target_width": 832,
    "seed": 42,
    "sample_guide_scale": 6,
    "enable_cfg": true,
    "fps": 16,
    "video_frame_interpolation": {
        "algo": "rife",
        "target_fps": 60,
        "model_path": "/path/to/rife/train_log"
    }
}
```

Run command:
```bash
python lightx2v/infer.py \
    --model_cls wan2.1 \
    --task i2v \
    --model_path ./models/wan2.1 \
    --config_json ./wan_i2v_vfi_60fps.json \
    --image_path ./input.jpg \
    --prompt "Smooth camera movement" \
    --save_video_path ./output_60fps.mp4
```

## Performance Considerations

### Memory Usage

- RIFE processing requires additional GPU memory
- Memory usage scales with video resolution and length
- Consider using lower resolutions for longer videos

### Processing Time

- Frame interpolation adds processing overhead
- Higher target frame rates require more computation
- Processing time is roughly proportional to the number of interpolated frames

### Quality vs Speed Trade-offs

- Higher interpolation ratios may introduce artifacts
- Optimal range: 2x to 4x frame rate increase
- For extreme interpolation (>4x), consider multiple passes

## Best Practices

### Optimal Use Cases

- **Motion-heavy videos**: Benefit most from frame interpolation
- **Camera movements**: Smoother panning and zooming
- **Action sequences**: Reduced motion blur perception
- **Slow-motion effects**: Create fluid slow-motion videos

### Recommended Settings

- **Source FPS**: 16-24 FPS (generated by base model)
- **Target FPS**: 32-60 FPS (2x to 4x increase)
- **Resolution**: Up to 720p for best performance

### Troubleshooting

#### Common Issues

1. **Out of Memory**: Reduce video resolution or target FPS
2. **Artifacts in output**: Lower the interpolation ratio
3. **Slow processing**: Check GPU memory and consider using CPU offloading

#### Solutions

Solve issues by modifying the configuration file:

```json
{
    // For memory issues, use lower resolution
    "target_height": 480,
    "target_width": 832,

    // For quality issues, use moderate interpolation
    "video_frame_interpolation": {
        "target_fps": 24  // instead of 60
    },

    // For performance issues, enable offloading
    "cpu_offload": true
}
```

## Technical Implementation

The RIFE integration in LightX2V includes:

- **RIFEWrapper**: ComfyUI-compatible wrapper for RIFE model
- **Automatic Model Loading**: Seamless integration with the inference pipeline
- **Memory Optimization**: Efficient tensor management and GPU memory usage
- **Quality Preservation**: Maintains original video quality while adding frames
