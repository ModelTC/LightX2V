"""Post-processing CLI for Hunyuan3D-2.1 mesh texture (paint) generation."""

import argparse
import os

from loguru import logger
from paint_pipeline import PaintPipeline


def get_postprocess_paint_parser():
    parser = argparse.ArgumentParser(description="Post-processing pipeline for Hunyuan3D mesh texture (paint).")

    parser.add_argument(
        "--hy_repo",
        type=str,
        default=None,
        help="Optional override: Hunyuan3D-2.1 source tree containing hy3dpaint/. Defaults to tools/postprocess/hy3dpaint (symlink to upstream hy3dpaint/).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to Hunyuan3D-2.1 HF weights (hunyuan3d-paintpbr-v2-1, etc.).",
    )
    parser.add_argument("--mesh_path", type=str, required=True, help="Input mesh from shape generation (.glb / .obj).")
    parser.add_argument("--image_path", type=str, required=True, help="Reference image for texture generation.")
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Output path for textured mesh (.glb recommended) or output directory prefix.",
    )

    parser.add_argument("--max_num_view", type=int, default=6, help="Number of multiview diffusion views (6-9).")
    parser.add_argument("--resolution", type=int, default=512, choices=[512, 768], help="Multiview diffusion resolution.")
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("AI_DEVICE", "cuda"),
        help="Torch device for paint models (default: AI_DEVICE env or cuda).",
    )
    parser.add_argument(
        "--multiview_cfg_path",
        type=str,
        default=None,
        help="Override path to hunyuan-paint-pbr.yaml (default: hy3dpaint/cfgs/hunyuan-paint-pbr.yaml).",
    )
    parser.add_argument(
        "--realesrgan_ckpt_path",
        type=str,
        default=None,
        help="Override RealESRGAN checkpoint (default: hy3dpaint/ckpt/RealESRGAN_x4plus.pth).",
    )
    parser.add_argument(
        "--custom_pipeline",
        type=str,
        default=None,
        help="Override custom diffusers pipeline dir (default: hy3dpaint/hunyuanpaintpbr).",
    )
    parser.add_argument(
        "--dino_ckpt_path",
        type=str,
        default="facebook/dinov2-giant",
        help="DINOv2 checkpoint id or local path.",
    )

    parser.add_argument("--use_remesh", action="store_true", default=True, help="Remesh input before texturing.")
    parser.add_argument("--no_remesh", action="store_false", dest="use_remesh", help="Skip remesh step.")
    parser.add_argument("--save_glb", action="store_true", default=True, help="Export textured mesh as .glb.")
    parser.add_argument("--no_save_glb", action="store_false", dest="save_glb", help="Keep textured .obj only.")
    return parser


def process_paint(args):
    args_dict = vars(args)
    logger.info(args_dict)

    pipeline = PaintPipeline(
        model_path=args.model_path,
        hy_repo=args.hy_repo,
        max_num_view=args.max_num_view,
        resolution=args.resolution,
        device=args.device,
        multiview_cfg_path=args.multiview_cfg_path,
        realesrgan_ckpt_path=args.realesrgan_ckpt_path,
        custom_pipeline=args.custom_pipeline,
        dino_ckpt_path=args.dino_ckpt_path,
    )
    result_path = pipeline(
        mesh_path=args.mesh_path,
        image_path=args.image_path,
        save_path=args.save_path,
        use_remesh=args.use_remesh,
        save_glb=args.save_glb,
    )
    print(f"Saved textured mesh: {result_path}")
    # bpy / custom_rasterizer may segfault during normal interpreter teardown after a
    # successful run; exit immediately so shell scripts get a clean status code.
    os._exit(0)


if __name__ == "__main__":
    parser = get_postprocess_paint_parser()
    process_paint(parser.parse_args())
