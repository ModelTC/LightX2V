"""Log startup settings and effective request inputs."""

import json
from dataclasses import fields

from loguru import logger


def print_config(config, title="config"):
    from lightx2v.utils.utils import is_main_process

    if is_main_process():
        logger.info(f"{title}:\n{json.dumps(config, ensure_ascii=False, indent=4, default=str)}")


def print_request(input_info, supported_request_fields):
    request = {input_field.name: getattr(input_info, input_field.name) for input_field in fields(input_info) if input_field.repr and input_field.name in supported_request_fields}
    print_config(request, title="Effective request")
