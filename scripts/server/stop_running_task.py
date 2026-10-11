import requests
from loguru import logger

url = "http://localhost:8000"
response = requests.get(f"{url}/v1/service/status", timeout=10)
response.raise_for_status()
task_id = response.json()["current_task"]
if task_id is None:
    logger.info("No task is currently running")
else:
    response = requests.delete(f"{url}/v1/tasks/{task_id}", timeout=30)
    response.raise_for_status()
    logger.info(response.json())
