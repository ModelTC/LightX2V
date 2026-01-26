# -*- coding: utf-8 -*-

import asyncio
import base64
import json
import os
import sys
import time
import uuid

import aiohttp
from loguru import logger


class VolcEngineASRClient:
    """
    VolcEngine ASR Client
    """

    # Error code definitions
    ERROR_CODES = {
        "20000000": "Success",
        "20000001": "Task in progress",
        "20000002": "Task waiting",
        "20000003": "Silent audio",
        "45000001": "Invalid request parameters (missing required fields / invalid field values)",
        "45000002": "Empty audio",
        "45000151": "Incorrect audio format",
        "55000031": "Server busy (service overloaded, unable to process current request)",
    }

    def __init__(self):
        self.url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
        self.appid = os.getenv("VOLCENGINE_ASR_APPID")
        self.access_token = os.getenv("VOLCENGINE_ASR_ACCESS_TOKEN")
        self.proxy = os.getenv("HTTPS_PROXY", None)
        if self.proxy:
            logger.info(f"volcengine asr use proxy: {self.proxy}")

    def _file_to_base64(self, file_path):
        """Convert local file to Base64"""
        with open(file_path, "rb") as file:
            file_data = file.read()
            base64_data = base64.b64encode(file_data).decode("utf-8")
        return base64_data

    async def _download_file(self, file_url):
        """Download file"""
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url, proxy=self.proxy) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    raise Exception(f"Download failed, HTTP status code: {response.status}")

    async def recognize_request(
        self,
        file_url=None,
        file_path=None,
        model_name="bigmodel",
        resource_id="volc.bigasr.auc_turbo",
        enable_itn=False,
        enable_punc=True,
        enable_ddc=False,
        enable_speaker_info=False,
    ):
        """
        Execute ASR recognition request

        Args:
            file_url: Audio file URL (optional)
            file_path: Local audio file path (optional)
            model_name: Model name, default "bigmodel"
            resource_id: Resource ID, default "volc.bigasr.auc_turbo"
            enable_itn: Whether to enable inverse text normalization
            enable_punc: Whether to enable punctuation
            enable_ddc: Whether to enable speaker diarization
            enable_speaker_info: Whether to enable speaker information

        Returns:
            tuple: (success: bool, result: dict or str)
                - success: True indicates success, False indicates failure
                - result: Returns recognition result dict on success, error message string on failure
        """
        if not self.appid:
            error_msg = "VOLCENGINE_APPID is not set"
            logger.error(error_msg)
            return False, error_msg

        if not self.access_token:
            error_msg = "VOLCENGINE_ACCESS_TOKEN is not set"
            logger.error(error_msg)
            return False, error_msg

        headers = {
            "X-Api-App-Key": self.appid,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
            "Content-Type": "application/json",
        }

        # Check whether to use file URL or upload data directly
        audio_data = None
        if file_url:
            audio_data = {"url": file_url}
        elif file_path:
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                logger.error(error_msg)
                return False, error_msg
            base64_data = await asyncio.to_thread(self._file_to_base64, file_path)
            audio_data = {"data": base64_data}
        else:
            error_msg = "Either file_url or file_path must be provided"
            logger.error(error_msg)
            return False, error_msg

        request_payload = {
            "user": {"uid": self.appid},
            "audio": audio_data,
            "request": {
                "model_name": model_name,
                "enable_itn": enable_itn,
                "enable_punc": enable_punc,
                "enable_ddc": enable_ddc,
                "enable_speaker_info": enable_speaker_info,
            },
        }

        try:
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=request_payload, headers=headers, proxy=self.proxy) as response:
                    # Check status code in response headers
                    status_code = response.headers.get("X-Api-Status-Code", "")
                    message = response.headers.get("X-Api-Message", "")
                    logid = response.headers.get("X-Tt-Logid", "")

                    logger.info(f"ASR request status code: {status_code}, message: {message}, logid: {logid}")

                    if status_code == "20000000":  # Success
                        result_data = await response.json()
                        elapsed_time = time.time() - start_time
                        logger.info(f"VolcEngineASRClient recognize request success, elapsed time: {elapsed_time:.3f} seconds")
                        return True, result_data
                    elif status_code in ["20000001", "20000002"]:  # Task in progress or waiting
                        error_msg = f"Task in progress, status: {status_code}, message: {message}"
                        logger.warning(error_msg)
                        return False, error_msg
                    else:  # Task failed
                        result_data = await response.json() if response.content_type == "application/json" else {}

                        # Get detailed error code description
                        error_description = self.ERROR_CODES.get(status_code, "")
                        if error_description:
                            error_msg = f"ASR request failed, code: {status_code} ({error_description}), message: {message}"
                        elif status_code.startswith("550"):
                            error_msg = f"ASR request failed, code: {status_code} (Internal service processing error), message: {message}"
                        else:
                            error_msg = f"ASR request failed, code: {status_code}, message: {message}"

                        if result_data:
                            error_msg += f", response: {result_data}"
                        logger.error(error_msg)
                        return False, error_msg

        except Exception as e:
            error_msg = f"VolcEngineASRClient recognize request failed: {str(e)}"
            logger.warning(error_msg)
            return False, error_msg


async def test(args):
    """
    ASR test function

    Args:
        args: list, e.g. [file_path, file_url, model_name, resource_id, enable_itn, enable_punc, enable_ddc, enable_speaker_info]
              Provide as many as needed, from left to right.

    Parameters:
        - file_path: Local audio file path
        - file_url: Audio file URL
        - model_name: Model name, default "bigmodel"
        - resource_id: Resource ID, default "volc.bigasr.auc_turbo"
        - enable_itn: Whether to enable inverse text normalization (True/False)
        - enable_punc: Whether to enable punctuation (True/False)
        - enable_ddc: Whether to enable speaker diarization (True/False)
        - enable_speaker_info: Whether to enable speaker information (True/False)
    """
    client = VolcEngineASRClient()
    # Set default parameters
    params = {
        "file_path": "/mtc/gongruihao/qinxinyi/lightx2v/lightx2v/deploy/common/sample.wav",
        "file_url": None,
        "model_name": "bigmodel",
        "resource_id": "volc.bigasr.auc_turbo",
        "enable_itn": False,
        "enable_punc": True,
        "enable_ddc": False,
        "enable_speaker_info": False,
    }
    keys = list(params.keys())
    # Override default parameters
    for i, arg in enumerate(args):
        if i < len(keys):
            # Type conversion
            if keys[i] in ["enable_itn", "enable_punc", "enable_ddc", "enable_speaker_info"]:
                # Support multiple boolean input formats
                params[keys[i]] = str(arg).lower() in ("1", "true", "yes", "on")
            else:
                params[keys[i]] = arg

    success, result = await client.recognize_request(
        file_url=params["file_url"],
        file_path=params["file_path"],
        model_name=params["model_name"],
        resource_id=params["resource_id"],
        enable_itn=params["enable_itn"],
        enable_punc=params["enable_punc"],
        enable_ddc=params["enable_ddc"],
        enable_speaker_info=params["enable_speaker_info"],
    )

    if success:
        logger.info(f"ASR recognition successful!")
        logger.info(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
    else:
        logger.warning(f"ASR recognition failed: {result}")


if __name__ == "__main__":
    asyncio.run(test(sys.argv[1:]))
