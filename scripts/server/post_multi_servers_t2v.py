from loguru import logger
from post_multi_servers import get_available_urls, process_tasks_async

if __name__ == "__main__":
    urls = ["http://localhost:8000", "http://localhost:8001"]
    prompts = [
        "A cat walks on the grass, realistic style.",
        "A person is riding a bike. Realistic, Natural lighting, Casual.",
        "A car turns a corner. Realistic, Natural lighting, Casual.",
        "An astronaut is flying in space, Van Gogh style. Dark, Mysterious.",
        "A beautiful coastal beach in spring, waves gently lapping on the sand, the camera movement is Zoom In. Realistic, Natural lighting, Peaceful.",
        "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    ]
    negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    messages = []
    for i, prompt in enumerate(prompts):
        messages.append({"prompt": prompt, "negative_prompt": negative_prompt, "image_path": "", "save_video_path": f"./output_lightx2v_wan_t2v_{i + 1}.mp4"})

    logger.info(f"urls: {urls}")

    # Get available servers
    available_urls = get_available_urls(urls)
    if not available_urls:
        exit(1)

    # Process tasks asynchronously
    success = process_tasks_async(messages, available_urls, show_progress=True)

    if success:
        logger.info("All tasks completed successfully!")
    else:
        logger.error("Some tasks failed.")
        exit(1)
