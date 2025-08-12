import base64
import os
import threading
import time

import requests
from loguru import logger
from tqdm import tqdm


def image_to_base64(image_path):
    """Convert an image file to base64 string"""
    with open(image_path, "rb") as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode("utf-8")


def process_image_path(image_path):
    """处理image_path：如果是本地路径则转换为base64，如果是HTTP链接则保持不变"""
    if not image_path:
        return image_path

    if image_path.startswith(("http://", "https://")):
        return image_path

    if os.path.exists(image_path):
        return image_to_base64(image_path)
    else:
        logger.warning(f"Image path not found: {image_path}")
        return image_path


def send_and_monitor_task(url, message, task_index, complete_bar, complete_lock):
    """Send task to server and monitor until completion"""
    try:
        if "image_path" in message and message["image_path"]:
            message["image_path"] = process_image_path(message["image_path"])

        response = requests.post(f"{url}/v1/tasks/", json=message)
        response_data = response.json()
        task_id = response_data.get("task_id")

        if not task_id:
            logger.error(f"No task_id received from {url}")
            return False

        # Step 2: Monitor task status until completion
        while True:
            try:
                status_response = requests.get(f"{url}/v1/tasks/{task_id}/status")
                status_data = status_response.json()
                task_status = status_data.get("status")

                if task_status == "completed":
                    # Update completion bar safely
                    if complete_bar and complete_lock:
                        with complete_lock:
                            complete_bar.update(1)
                    return True
                elif task_status == "failed":
                    logger.error(f"Task {task_index + 1} (task_id: {task_id}) failed")
                    if complete_bar and complete_lock:
                        with complete_lock:
                            complete_bar.update(1)  # Still update progress even if failed
                    return False
                else:
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed to check status for task_id {task_id}: {e}")
                time.sleep(0.5)

    except Exception as e:
        logger.error(f"Failed to send task to {url}: {e}")
        return False


def get_available_urls(urls):
    """Check which URLs are available and return the list"""
    available_urls = []
    for url in urls:
        try:
            _ = requests.get(f"{url}/v1/service/status").json()
            available_urls.append(url)
        except Exception as e:
            continue

    if not available_urls:
        logger.error("No available urls.")
        return None

    logger.info(f"available_urls: {available_urls}")
    return available_urls


def find_idle_server(available_urls):
    """Find an idle server from available URLs"""
    while True:
        for url in available_urls:
            try:
                response = requests.get(f"{url}/v1/service/status").json()
                if response["service_status"] == "idle":
                    return url
            except Exception as e:
                continue
        time.sleep(3)


def process_tasks_async(messages, available_urls, show_progress=True):
    """Process a list of tasks asynchronously across multiple servers"""
    if not available_urls:
        logger.error("No available servers to process tasks.")
        return False

    active_threads = []

    logger.info(f"Sending {len(messages)} tasks to available servers...")

    complete_bar = None
    complete_lock = None
    if show_progress:
        complete_bar = tqdm(total=len(messages), desc="Completing tasks")
        complete_lock = threading.Lock()  # Thread-safe updates to completion bar

    for idx, message in enumerate(messages):
        # Find an idle server
        server_url = find_idle_server(available_urls)

        # Create and start thread for sending and monitoring task
        thread = threading.Thread(target=send_and_monitor_task, args=(server_url, message, idx, complete_bar, complete_lock))
        thread.daemon = False
        thread.start()
        active_threads.append(thread)

        # Small delay to let thread start
        time.sleep(0.5)

    # Wait for all threads to complete
    for thread in active_threads:
        thread.join()

    # Close completion bar
    if complete_bar:
        complete_bar.close()

    logger.info("All tasks processing completed!")
    return True
