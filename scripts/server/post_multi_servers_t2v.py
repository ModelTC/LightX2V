import argparse
from pathlib import Path

from loguru import logger
from post_multi_servers import get_available_urls, process_tasks_async


def load_prompts_from_folder(folder_path):
    """Load prompts from all files in the specified folder.

    Returns:
        tuple: (prompts, filenames) where prompts is a list of prompt strings
               and filenames is a list of corresponding filenames
    """
    prompts = []
    filenames = []
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        logger.error(f"Prompt folder does not exist or is not a directory: {folder_path}")
        return prompts, filenames

    # Get all files in the folder and sort them
    files = sorted(folder.glob("*"))
    files = [f for f in files if f.is_file()]

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:  # Only add non-empty prompts
                    prompts.append(content)
                    filenames.append(file_path.name)
                    # logger.info(f"Loaded prompt from {file_path.name}")
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")

    return prompts, filenames


def load_prompts_from_file(file_path):
    """Load prompts from a file, one prompt per line.

    Returns:
        list: prompts, where each element is a prompt string
    """
    prompts = []
    file = Path(file_path)

    if not file.exists() or not file.is_file():
        logger.error(f"Prompt file does not exist or is not a file: {file_path}")
        return prompts

    try:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                prompt = line.strip()
                if prompt:  # Only add non-empty prompts
                    prompts.append(prompt)
    except Exception as e:
        logger.error(f"Failed to read prompt file {file_path}: {e}")

    return prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post prompts to multiple T2V servers")
    parser.add_argument("--prompt-folder", type=str, default=None, help="Folder containing prompt files. If not specified, use default prompts.")
    parser.add_argument("--prompt-file", type=str, default=None, help="File containing prompts, one prompt per line. Cannot be used together with --prompt-folder.")
    parser.add_argument("--save-folder", type=str, default="./", help="Folder to save output videos. Default is current directory.")
    args = parser.parse_args()

    # Check that --prompt-folder and --prompt-file are not used together
    if args.prompt_folder and args.prompt_file:
        logger.error("Cannot use --prompt-folder and --prompt-file together. Please choose one.")
        exit(1)

    # Generate URLs from IPs (each IP has 8 ports: 8000-8007)
    ips = ["localhost"]
    urls = [f"http://{ip}:{port}" for ip in ips for port in range(8000, 8008)]
    # urls = ["http://localhost:8007"]

    logger.info(f"urls: {urls}")

    # Get available servers
    available_urls = get_available_urls(urls)
    if not available_urls:
        exit(1)

    logger.info(f"Total {len(available_urls)} available servers.")

    # Load prompts from folder, file, or use default prompts
    prompt_filenames = None
    if args.prompt_folder:
        logger.info(f"Loading prompts from folder: {args.prompt_folder}")
        prompts, prompt_filenames = load_prompts_from_folder(args.prompt_folder)
        if not prompts:
            logger.error("No valid prompts loaded from folder.")
            exit(1)
    elif args.prompt_file:
        logger.info(f"Loading prompts from file: {args.prompt_file}")
        prompts = load_prompts_from_file(args.prompt_file)
        if not prompts:
            logger.error("No valid prompts loaded from file.")
            exit(1)
    else:
        logger.info("Using default prompts")
        prompts = [
            "A cat walks on the grass, realistic style.",
            "A person is riding a bike. Realistic, Natural lighting, Casual.",
            "A car turns a corner. Realistic, Natural lighting, Casual.",
            "An astronaut is flying in space, Van Gogh style. Dark, Mysterious.",
            "A beautiful coastal beach in spring, waves gently lapping on the sand, the camera movement is Zoom In. Realistic, Natural lighting, Peaceful.",
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
        ]

    negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    # Prepare save folder
    save_folder = Path(args.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    messages = []
    total_count = len(prompts)
    skipped_count = 0

    for i, prompt in enumerate(prompts):
        # Generate output filename
        if prompt_filenames:
            # Use prompt filename, replace extension with .mp4
            filename = Path(prompt_filenames[i]).stem + ".mp4"
        else:
            # Use default naming
            filename = f"output_lightx2v_wan_t2v_{i + 1}.mp4"

        save_path = save_folder / filename

        # Skip if file already exists (only when using prompt_filenames)
        if prompt_filenames and save_path.exists():
            logger.info(f"Skipping {filename} - file already exists")
            skipped_count += 1
            continue

        messages.append({"seed": 42, "prompt": prompt, "negative_prompt": negative_prompt, "image_path": "", "save_result_path": str(save_path)})

    # Log statistics
    to_process_count = len(messages)
    logger.info("=" * 80)
    logger.info("Task Statistics:")
    logger.info(f"  Total prompts: {total_count}")
    logger.info(f"  Skipped (already exists): {skipped_count}")
    logger.info(f"  To process: {to_process_count}")
    logger.info("=" * 80)

    if to_process_count == 0:
        logger.info("No tasks to process. All files already exist.")
        exit(0)

    # Process tasks asynchronously
    success = process_tasks_async(messages, available_urls, show_progress=True)

    if success:
        logger.info("All tasks completed successfully!")
    else:
        logger.error("Some tasks failed.")
        exit(1)
