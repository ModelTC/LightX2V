# Wan-animate Preprocessing User Guider

## 1. Introductions


Wan-animate offers two generation modes: `animation` and `replacement`. While both modes extract the skeleton from the reference video, they each have a distinct preprocessing pipeline.

### 1.1 Animation Mode

In this mode, it is highly recommended to enable pose retargeting, especially if the body proportions of the reference and driving characters are dissimilar.

 - A simplified version of pose retargeting pipeline is provided to help developers quickly implement this functionality.

 - **NOTE:** Due to the potential complexity of input data, the results from this simplified retargeting version are NOT guaranteed to be perfect. It is strongly advised to verify the preprocessing results before proceeding.

 - Community contributions to improve on this feature are welcome.

### 1.2 Replacement Mode

 - Pose retargeting is DISABLED by default in this mode. This is a deliberate choice to account for potential spatial interactions between the character and the environment.

 - **WARNING**: If there is a significant mismatch in body proportions between the reference and driving characters, artifacts or deformations may appear in the final output.

 - A simplified version for extracting the character's mask is also provided.
 - **WARNING:** This mask extraction process is designed for **single-person videos ONLY** and may produce incorrect results or fail in multi-person videos (incorrect pose tracking). For multi-person video, users are required to either develop their own solution or integrate a suitable open-source tool.

---

## 2. Preprocessing Instructions and Recommendations

### 2.1 Basic Usage

- The preprocessing process requires some additional models, including pose detection (mandatory), and mask extraction and image editing models (optional, as needed). Place them according to the following directory structure:
```
    /path/to/your/ckpt_path/
    ├── det/
    │ └── yolov10m.onnx
    ├── pose2d/
    │ └── vitpose_h_wholebody.onnx
    ├── sam2/
    │ └── sam2_hiera_large.pt
    └── FLUX.1-Kontext-dev/
```
- `video_path`, `refer_path`, and `save_path` correspond to the paths for the input driving video, the character image, and the preprocessed results.

- When using `animation` mode, two videos, `src_face.mp4` and `src_pose.mp4`, will be generated in `save_path`. When using `replacement` mode, two additional videos, `src_bg.mp4` and `src_mask.mp4`, will also be generated.

- The `resolution_area` parameter determines the resolution for both preprocessing and the generation model. Its size is determined by pixel area.

- The `fps` parameter can specify the frame rate for video processing. A lower frame rate can improve generation efficiency, but may cause stuttering or choppiness.

---

### 2.2 Animation Mode

- We support three forms: not using pose retargeting, using basic pose retargeting, and using enhanced pose retargeting based on the `FLUX.1-Kontext-dev` image editing model. These are specified via the `retarget_flag` and `use_flux` parameters.

- Specifying `retarget_flag` to use basic pose retargeting requires ensuring that both the reference character and the character in the first frame of the driving video are in a front-facing, stretched pose.

- Other than that, we recommend using enhanced pose retargeting by specifying both `retarget_flag` and `use_flux`. **NOTE:** Due to the limited capabilities of `FLUX.1-Kontext-dev`, it is NOT guaranteed to produce the expected results (e.g., consistency is not maintained, the pose is incorrect, etc.). It is recommended to check the intermediate results as well as the finally generated pose video; both are stored in `save_path`. Of course, users can also use a better image editing model, or explore the prompts for Flux on their own.

---

### 2.3 Replacement Mode

- Specifying `replace_flag` to enable data preprocessing for this mode. The preprocessing will additionally process a mask for the character in the video, and its size and shape can be adjusted by specifying some parameters.
- `iterations` and `k` can make the mask larger, covering more area.
- `w_len` and `h_len` can adjust the mask's shape. Smaller values will make the outline coarser, while larger values will make it finer.

- A smaller, finer-contoured mask can allow for more of the original background to be preserved, but may potentially limit the character's generation area (considering potential appearance differences, this can lead to some shape leakage). A larger, coarser mask can allow the character generation to be more flexible and consistent, but because it includes more of the background, it might affect the background's consistency. We recommend users to adjust the relevant parameters based on their specific input data.
