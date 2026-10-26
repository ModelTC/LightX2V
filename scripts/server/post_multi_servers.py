import base64
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from loguru import logger
from tqdm import tqdm


def image_to_base64(image_path):
    """Convert an image file to base64 string"""
    with open(image_path, "rb") as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode("utf-8")


def process_image_path(image_path) -> Any | str:
    """Process image_path: convert to base64 if local path, keep unchanged if HTTP link"""
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

        response = requests.post(f"{url}/v1/tasks/", json=message, timeout=30)
        response.raise_for_status()
        response_data = response.json()
        task_id = response_data.get("task_id")

        if not task_id:
            logger.error(f"No task_id received from {url}")
            return False

        # Step 2: Monitor task status until completion
        while True:
            status_response = requests.get(f"{url}/v1/tasks/{task_id}/status", timeout=15)
            status_response.raise_for_status()
            status_data = status_response.json()
            task_status = status_data["status"]

            if task_status == "completed":
                return True
            if task_status in ("failed", "cancelled"):
                logger.error(f"Task {task_index + 1} (task_id: {task_id}) {task_status}: {status_data.get('error')}")
                return False
            time.sleep(0.5)

    except Exception as e:
        logger.error(f"Task {task_index + 1} at {url} failed: {e}")
        return False
    finally:
        if complete_bar is not None and complete_lock is not None:
            with complete_lock:
                complete_bar.update(1)


def get_available_urls(urls):
    """Check which URLs are available and return the list"""
    available_urls = []
    for url in urls:
        try:
            response = requests.get(f"{url}/v1/service/status", timeout=10)
            response.raise_for_status()
            response.json()
            available_urls.append(url)
        except requests.RequestException as e:
            logger.warning(f"Server {url} is unavailable: {e}")
            continue

    if not available_urls:
        logger.error("No available urls.")
        return None

    logger.info(f"available_urls: {available_urls}")
    return available_urls


def find_idle_server(available_urls):
    """Find an idle server from available URLs"""
    while True:
        reachable = False
        for url in available_urls:
            try:
                response = requests.get(f"{url}/v1/service/status", timeout=10)
                response.raise_for_status()
                status = response.json()["service_status"]
                reachable = True
                if status == "idle":
                    return url
            except requests.RequestException as e:
                logger.warning(f"Server {url} is unavailable: {e}")
                continue
        if not reachable:
            raise RuntimeError("No available servers to process tasks")
        time.sleep(3)


def process_tasks_async(messages, available_urls, show_progress=True):
    """Process a list of tasks asynchronously across multiple servers"""
    if not available_urls:
        logger.error("No available servers to process tasks.")
        return False

    futures = []

    logger.info(f"Sending {len(messages)} tasks to available servers...")

    complete_bar = None
    complete_lock = None
    if show_progress:
        complete_bar = tqdm(total=len(messages), desc="Completing tasks")
        complete_lock = threading.Lock()  # Thread-safe updates to completion bar

    try:
        with ThreadPoolExecutor(max_workers=max(1, len(messages))) as executor:
            for idx, message in enumerate(messages):
                server_url = find_idle_server(available_urls)
                futures.append(executor.submit(send_and_monitor_task, server_url, message, idx, complete_bar, complete_lock))
                time.sleep(0.5)
    finally:
        if complete_bar is not None:
            complete_bar.close()

    logger.info("All tasks processing completed!")
    return all(future.result() for future in futures)
