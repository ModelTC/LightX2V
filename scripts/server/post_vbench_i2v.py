import argparse
import glob
import os
from loguru import logger
from post_multi_servers import get_available_urls, process_tasks_async


def create_i2v_messages(img_files, output_path):
    """Create messages for image-to-video tasks"""
    messages = []
    negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    for img_path in img_files:
        file_name = os.path.basename(img_path)
        prompt = os.path.splitext(file_name)[0]
        save_video_path = os.path.join(output_path, f"{prompt}.mp4")

        message = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_path": img_path,
            "save_video_path": save_video_path,
        }
        messages.append(message)

    return messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to img files.")
    parser.add_argument("--output_path", type=str, default="./vbench_i2v", help="output video path.")
    args = parser.parse_args()

    # Create server URLs
    urls = [f"http://localhost:{port}" for port in range(8000, 8008)]

    # Get available servers
    available_urls = get_available_urls(urls)
    if not available_urls:
        exit(1)

    # Find image files
    if os.path.exists(args.data_path):
        img_files = glob.glob(os.path.join(args.data_path, "*.jpg"))
        logger.info(f"Found {len(img_files)} image files.")

        if not img_files:
            logger.error("No image files found.")
            exit(1)

        # Create messages for all images
        messages = create_i2v_messages(img_files, args.output_path)
        logger.info(f"Created {len(messages)} tasks.")

        # Process tasks asynchronously
        success = process_tasks_async(messages, available_urls, show_progress=True)

        if success:
            logger.info("All image-to-video tasks completed successfully!")
        else:
            logger.error("Some tasks failed.")
            exit(1)
    else:
        logger.error(f"Data path does not exist: {args.data_path}")
        exit(1)
