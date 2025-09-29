import argparse
import os

from loguru import logger
from post_multi_servers import get_available_urls, process_tasks_async


def main():
    parser = argparse.ArgumentParser(description="Generate videos from prompts using multiple servers.")
    parser.add_argument("--prompt_file", type=str, help="Path to a file containing prompts, one per line")
    args = parser.parse_args()

    urls = ["http://localhost:8000", "http://localhost:8001"]
    negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    if args.prompt_file:
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]

            file_name = os.path.splitext(os.path.basename(args.prompt_file))[0]

            output_dir = f"./{file_name}"
            os.makedirs(output_dir, exist_ok=True)

            logger.info(f"Loaded {len(prompts)} prompts from file: {args.prompt_file}")
            logger.info(f"Output will be saved to: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to read prompt file: {e}")
            return
    else:
        prompts = [
            "A cat walks on the grass, realistic style.",
            "A person is riding a bike. Realistic, Natural lighting, Casual.",
            "A car turns a corner. Realistic, Natural lighting, Casual.",
            "An astronaut is flying in space, Van Gogh style. Dark, Mysterious.",
            "A beautiful coastal beach in spring, waves gently lapping on the sand, the camera movement is Zoom In. Realistic, Natural lighting, Peaceful.",
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
        ]
        file_name = ""

    messages = []
    for i, prompt in enumerate(prompts):
        if args.prompt_file:
            save_path = f"./{file_name}/output_lightx2v_wan_t2v_{i + 1}.mp4"
        else:
            save_path = f"./output_lightx2v_wan_t2v_{i + 1}.mp4"

        messages.append({"prompt": prompt, "negative_prompt": negative_prompt, "image_path": "", "save_video_path": save_path})

    logger.info(f"urls: {urls}")

    available_urls = get_available_urls(urls)
    if not available_urls:
        logger.error("No available servers found")
        exit(1)

    success = process_tasks_async(messages, available_urls, show_progress=True)

    if success:
        logger.info("All tasks completed successfully!")
    else:
        logger.error("Some tasks failed.")
        exit(1)


if __name__ == "__main__":
    main()
