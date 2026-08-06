#!/usr/bin/env python3
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch
from PIL import Image

from lightx2v.models.runners.bagel.sensenova_vision_runner import SenseNovaVisionRunner
from lightx2v.utils.input_info import SenseNovaVisionInputInfo
from lightx2v.utils.set_config import set_config


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="/data/nvme0/lhd_codes/SenseNova-Vision/models/SenseNova-Vision-7B-MoT",
    )
    parser.add_argument(
        "--source_path",
        default="/data/nvme0/lhd_codes/sensenova-vision-v2",
    )
    parser.add_argument(
        "--output_dir",
        default=str(repo_root / "save_results/sensenova_vision_example"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--example",
        choices=["all", *(f"{index:02d}" for index in range(1, 15))],
        default="all",
        help="Run all examples or only the selected example number.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    lightx2v_root = Path(__file__).resolve().parents[2]
    source_root = Path(args.source_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(source_root))
    from inference.example_visualize import (  # noqa: E402
        CAMERA_POSE_PROMPT,
        EXAMPLE_08_PANOPTIC_IMAGE,
        EXAMPLE_08_PANOPTIC_QUESTION,
        EXAMPLE_09_INTERSEG_IMAGE,
        EXAMPLE_09_INTERSEG_PROMPT_IMAGE,
        EXAMPLE_09_INTERSEG_QUESTION,
    )
    from utils.visualize import (  # noqa: E402
        VisualizationConfig,
        draw_visual_prompt,
        visualize_binary_segmentation,
        visualize_concat_col,
        visualize_detection,
        visualize_gcg_segmentation,
        visualize_panoptic_segmentation,
    )

    config_args = Namespace(
        model_cls="sensenova_vision",
        task="omni_vision_task",
        model_path=args.model_path,
        config_json=str(lightx2v_root / "configs/sensenova_vision/sensenova_vision.json"),
        seed=args.seed,
    )
    config = set_config(config_args)
    config["sensenova_source_path"] = str(source_root)
    runner = SenseNovaVisionRunner(config)
    runner.init_modules()
    vis_config = VisualizationConfig()

    def selected(example_number):
        return args.example in {"all", example_number}

    def source_file(relative_path):
        return str(source_root / relative_path)

    def run(subtask, image_paths, prompt, save_name="", seed=None, **kwargs):
        info = SenseNovaVisionInputInfo(
            seed=args.seed if seed is None else seed,
            prompt=prompt,
            image_path=",".join(source_file(path) for path in image_paths),
            save_result_path=str(output_dir / save_name) if save_name else "",
            omni_vision_subtask=subtask,
            **kwargs,
        )
        return runner.run_pipeline(info)

    # 1. General understanding.
    if selected("01"):
        result = run(
            "understanding",
            ["examples/images/1.jpg"],
            "<image> What are the main objects in this scene and their relationships?",
            save_name="example_01_understanding.txt",
        )
        print(result["text"])

    # 2. Binary segmentation.
    if selected("02"):
        image_path = "examples/images/2.jpg"
        source = Image.open(source_file(image_path)).convert("RGB")
        result = run(
            "binary_segmentation",
            [image_path],
            "<image> Could you return the binary segmentation masks for the specified categories: <p>person furthest to the right</p>?",
            save_name="example_02_binary_segmentation_raw.png",
        )
        pred = visualize_binary_segmentation(source, result["images"][0], label="person furthest to the right", config=vis_config)
        visualize_concat_col(source, pred, concat_col=2).save(output_dir / "example_02_binary_segmentation.png")

    # 3. Depth and 4. surface normals.
    if selected("03"):
        run(
            "depth",
            ["examples/images/3.jpg"],
            "<image> Estimate relative depth for each pixel in the image, with closer objects appearing brighter and distant objects appearing darker. Output is a grayscale image with pixel values ranging from 0-255.",
            save_name="example_03_depth.png",
        )
    if selected("04"):
        run(
            "normal",
            ["examples/images/2.jpg"],
            "<image> Generate an RGB normal map where R, G, B channels represent X, Y, Z surface directions. The output should show continuous color variations with no discrete regions, unlike segmentation results.",
            save_name="example_04_normal.png",
        )

    # 5. Grounded caption + segmentation.
    if selected("05"):
        image_path = "examples/images/4.jpg"
        source = Image.open(source_file(image_path)).convert("RGB")
        result = run(
            "gcg_segmentation",
            [image_path],
            "<image> Please briefly describe the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.",
            save_name="example_05_gcg_segmentation_raw.png",
        )
        pred = visualize_gcg_segmentation(source, result["images"][0], result["text"], config=vis_config)
        visualize_concat_col(source, pred, concat_col=2).save(output_dir / "example_05_gcg_segmentation.png")

    # 6. Object detection.
    if selected("06"):
        image_path = "examples/images/5.jpg"
        source = Image.open(source_file(image_path)).convert("RGB")
        result = run(
            "object_detection",
            [image_path],
            "<image> Please detect all instances of <p>bird</p>, <p>boat</p>, <p>person</p>, <p>cell phone</p>, <p>backpack</p>, <p>handbag</p> in the image. Output the results as a structured text list with each detection including category and bounding box coordinates in <bbox> format.",
            save_name="example_06_object_detection.txt",
        )
        pred = visualize_detection(source, result["text"], task_name="common_object_detection", config=vis_config)
        visualize_concat_col(source, pred, concat_col=2).save(output_dir / "example_06_object_detection.png")

    # 7. Multi-view 3D reconstruction.
    recon_images = [
        "examples/recon3d/47204575_4847.103.png",
        "examples/recon3d/47204575_4852.001.png",
        "examples/recon3d/47204575_4871.692.png",
        "examples/recon3d/47204575_4873.692.png",
        "examples/recon3d/47204575_4875.791.png",
    ]
    if selected("07"):
        recon = run(
            "recon3d",
            recon_images,
            "",
            save_name="example_07_pred_raw.npy",
            seed=123456,
            raw_output_path=str(output_dir / "example_07_pred_raw.npy"),
            glb_output_path=str(output_dir / "example_07_pred_scene.glb"),
            postprocess_predictions=True,
        )
        print("recon pts3d shape:", recon["pts3d"].shape)

    # 8. Panoptic segmentation.
    if selected("08"):
        image_path = EXAMPLE_08_PANOPTIC_IMAGE
        source = Image.open(source_file(image_path)).convert("RGB")
        result = run(
            "panoptic_segmentation",
            [image_path],
            EXAMPLE_08_PANOPTIC_QUESTION,
            save_name="example_08_panoptic_segmentation_raw.png",
        )
        pred = visualize_panoptic_segmentation(source, result["images"][0], result["text"], question=EXAMPLE_08_PANOPTIC_QUESTION, config=vis_config)
        visualize_concat_col(source, pred, concat_col=2).save(output_dir / "example_08_panoptic_segmentation.png")

    # 9. Interactive segmentation with a visual prompt image.
    if selected("09"):
        image_path = EXAMPLE_09_INTERSEG_IMAGE
        prompt_path = EXAMPLE_09_INTERSEG_PROMPT_IMAGE
        source = Image.open(source_file(image_path)).convert("RGB")
        prompt_image = Image.open(source_file(prompt_path)).convert("L")
        result = run(
            "interactive_segmentation",
            [image_path, prompt_path],
            EXAMPLE_09_INTERSEG_QUESTION,
            save_name="example_09_interactive_segmentation_raw.png",
        )
        prompt_panel = draw_visual_prompt(source, prompt_image, prompt_style="boundary")
        pred = visualize_binary_segmentation(source, result["images"][0], label="box prompt", config=vis_config)
        visualize_concat_col(source, pred, concat_col=3, prompt=prompt_panel).save(output_dir / "example_09_interactive_segmentation.png")

    # 10. Visual-grounded instance segmentation.
    if selected("10"):
        image_path = "examples/images/8.jpg"
        source = Image.open(source_file(image_path)).convert("RGB")
        question = "<image> Identify all objects belonging to the same classes as the visually provided <p>object1</p><bbox>[0.616, 0.049, 0.785, 0.224]</bbox>. Generate an instance segmentation visualization and each identified category <p>object1</p> is colored different. First, enumerate each visible <p>object1</p> instance mentioned in the request and assign each <p>object1</p> a different color. Reformat them in the EXACT format: <p>object1<color>(R,G,B)</color></p>. Then respond with interleaved instance segmentation masks using those instance labels and colors."
        result = run("vgd_segmentation", [image_path], question, save_name="example_10_vgd_segmentation_raw.png")
        pred = visualize_gcg_segmentation(source, result["images"][0], result["text"], config=vis_config)
        visualize_concat_col(source, pred, concat_col=2).save(output_dir / "example_10_vgd_segmentation.png")

    # 11. Relative camera poses.
    if selected("11"):
        pose = run(
            "camera_pose",
            recon_images,
            ("<image>" * len(recon_images)) + CAMERA_POSE_PROMPT,
            save_name="example_11_camera_pose.txt",
        )
        print(pose["text"])

    # 12. Point detection.
    if selected("12"):
        image_path = "examples/point/image.jpg"
        source = Image.open(source_file(image_path)).convert("RGB")
        question = (
            "Locate and identify <p>airplane</p> within the scene. Output detection results as text entries, each containing the object class and pixel coordinates defining the object point location."
        )
        result = run(
            "point_detection",
            [image_path],
            question,
            save_name="example_12_point_detection.txt",
        )
        pred = visualize_detection(
            source,
            result["text"],
            task_name="point_detection",
            prompt=question,
            config=vis_config,
        )
        visualize_concat_col(source, pred, concat_col=2).save(output_dir / "example_12_point_detection.png")

    # 13. Human keypoint detection.
    if selected("13"):
        image_path = "examples/keypoint/person/image.png"
        source = Image.open(source_file(image_path)).convert("RGB")
        question = (
            "Detect all instances of <p>person</p> in the image. Unlike depth or pose visualization tasks, "
            "output structured text with each object class, <bbox>, and nose, left eye, right eye, left ear, "
            "right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left "
            "hip, right hip, left knee, right knee, left ankle, right ankle coordinates for further processing."
        )
        result = run(
            "keypoint",
            [image_path],
            question,
            save_name="example_13_keypoint.txt",
        )
        pred = visualize_detection(
            source,
            result["text"],
            task_name="keypoint",
            prompt=question,
            config=vis_config,
        )
        visualize_concat_col(source, pred, concat_col=2).save(output_dir / "example_13_keypoint.png")

    # 14. Word-level OCR.
    if selected("14"):
        image_path = "examples/OCR/image.jpg"
        source = Image.open(source_file(image_path)).convert("RGB")
        question = (
            "Perform word-level text detection and recognition on the entire image. Output a structured text "
            "list containing every detected word, its bounding box coordinates with <bbox> format, and the "
            "recognized text content."
        )
        result = run(
            "ocr",
            [image_path],
            question,
            save_name="example_14_ocr.txt",
        )
        pred = visualize_detection(
            source,
            result["text"],
            task_name="ocr",
            prompt=question,
            config=vis_config,
        )
        visualize_concat_col(source, pred, concat_col=2).save(output_dir / "example_14_ocr.png")

    print(f"LightX2V SenseNova-Vision example {args.example} outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
