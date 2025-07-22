import requests
from loguru import logger


response = requests.get("http://localhost:8000/v1/service/status")
logger.info(response.json())


response = requests.get("http://localhost:8000/v1/tasks/")
logger.info(response.json())


response = requests.get("http://localhost:8000/v1/tasks/test_task_001/status")
logger.info(response.json())
