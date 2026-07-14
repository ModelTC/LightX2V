import requests
from loguru import logger

response = requests.get("http://localhost:8000/v1/local/video/generate/stop_running_task")
logger.info(response.json())
