from loguru import logger
from post_multi_servers import get_available_urls, process_tasks_async


if __name__ == "__main__":
    urls = [f"http://localhost:{port}" for port in range(8000, 8008)]
    img_prompts = {
        "assets/inputs/imgs/img_0.jpg": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "assets/inputs/imgs/img_2.jpg": "A close-up cinematic view of a person cooking in a warm,sunlit kitchen, using a wooden spatula to stir-fry a colorful mix of freshvegetables—carrots, broccoli, and bell peppers—in a black frying pan on amodern induction stove. The scene captures the glistening texture of thevegetables, steam gently rising, and subtle reflections on the stove surface.In the background, soft-focus jars, fruits, and a window with natural daylightcreate a cozy atmosphere. The hand motions are smooth and rhythmic, with a realisticsense of motion blur and lighting.",
    }
    negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    messages = []
    for i, (image_path, prompt) in enumerate(img_prompts.items()):
        messages.append({"prompt": prompt, "negative_prompt": negative_prompt, "image_path": image_path, "save_video_path": f"./output_lightx2v_wan_i2v_{i + 1}.mp4"})

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
