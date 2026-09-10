import requests
from loguru import logger

if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/image/"

    # Start the service with start_server_i2i_reuse.sh.
    # Run with reuse=False first; after the task completes successfully, set reuse=True.
    # Keep the image file and prompts unchanged; seed may change.
    message = {
        "prompt": "Turn the image into a watercolor painting while preserving the main subject.",
        "negative_prompt": "",
        # Use the same absolute path readable by the server for every request.
        "image_path": "/path/to/LightX2V/assets/inputs/imgs/img_0.jpg",
        "seed": 42,
        "reuse": False,
        "save_result_path": "./output_lightx2v_qwen_i2i_reuse.png",
    }

    logger.info(f"message: {message}")
    response = requests.post(url, json=message)
    response.raise_for_status()
    logger.info(f"response: {response.json()}")
