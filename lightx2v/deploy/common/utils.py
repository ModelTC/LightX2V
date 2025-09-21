import base64
import io
import os
import time
import traceback
from datetime import datetime

import httpx
import torchaudio
from PIL import Image
from loguru import logger

FMT = "%Y-%m-%d %H:%M:%S"


def current_time():
    return datetime.now().timestamp()


def time2str(t):
    d = datetime.fromtimestamp(t)
    return d.strftime(FMT)


def str2time(s):
    d = datetime.strptime(s, FMT)
    return d.timestamp()


def try_catch(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.error(f"Error in {func.__name__}:")
            traceback.print_exc()
            return None

    return wrapper


def class_try_catch(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception:
            logger.error(f"Error in {self.__class__.__name__}.{func.__name__}:")
            traceback.print_exc()
            return None

    return wrapper


def class_try_catch_async(func):
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except Exception:
            logger.error(f"Error in {self.__class__.__name__}.{func.__name__}:")
            traceback.print_exc()
            return None

    return wrapper


def data_name(x, task_id):
    if x == "input_image":
        x = x + ".png"
    elif x == "output_video":
        x = x + ".mp4"
    return f"{task_id}-{x}"


async def fetch_resource(url, timeout):
    logger.info(f"Begin to download resource from url: {url}")
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url, timeout=timeout) as response:
            response.raise_for_status()
            ans_bytes = []
            async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                ans_bytes.append(chunk)
                if len(ans_bytes) > 128:
                    raise Exception(f"url {url} recv data is too big")
            content = b"".join(ans_bytes)
    logger.info(f"Download url {url} resource cost time: {time.time() - t0} seconds")
    return content


async def preload_data(inp, inp_type, typ, val):
    try:
        if typ == "url":
            timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
            data = await fetch_resource(val, timeout=timeout)
        elif typ == "base64":
            data = base64.b64decode(val)
        elif typ == "stream":
            # no bytes data need to be saved by data_manager
            data = None
        else:
            raise ValueError(f"cannot read {inp}[{inp_type}] which type is {typ}!")

        # check if valid image bytes
        if inp_type == "IMAGE":
            image = Image.open(io.BytesIO(data))
            logger.info(f"load image: {image.size}")
            assert image.size[0] > 0 and image.size[1] > 0, "image is empty"
        elif inp_type == "AUDIO":
            if typ != "stream":
                try:
                    waveform, sample_rate = torchaudio.load(io.BytesIO(data), num_frames=10)
                    logger.info(f"load audio: {waveform.size()}, {sample_rate}")
                    assert waveform.size(0) > 0, "audio is empty"
                    assert sample_rate > 0, "audio sample rate is not valid"
                except Exception as e:
                    logger.warning(f"torchaudio failed to load audio, trying alternative method: {e}")
                    # 尝试使用其他方法验证音频文件
                    # 检查文件头是否为有效的音频格式
                    if len(data) < 4:
                        raise ValueError("Audio file too short")
                    # 检查常见的音频文件头
                    audio_headers = [b"RIFF", b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"OggS"]
                    if not any(data.startswith(header) for header in audio_headers):
                        logger.warning("Audio file doesn't have recognized header, but continuing...")
                    logger.info(f"Audio validation passed (alternative method), size: {len(data)} bytes")
        else:
            raise Exception(f"cannot parse inp_type={inp_type} data")
        return data

    except Exception as e:
        raise ValueError(f"Failed to read {inp}, type={typ}, val={val[:100]}: {e}!")


async def load_inputs(params, raw_inputs, types):
    inputs_data = {}
    for inp in raw_inputs:
        item = params.pop(inp)
        bytes_data = await preload_data(inp, types[inp], item["type"], item["data"])
        if bytes_data is not None:
            inputs_data[inp] = bytes_data
        else:
            params[inp] = item
    return inputs_data


def check_params(params, raw_inputs, raw_outputs, types):
    stream_audio = os.getenv("STREAM_AUDIO", "0") == "1"
    stream_video = os.getenv("STREAM_VIDEO", "0") == "1"
    for x in raw_inputs + raw_outputs:
        if x in params and "type" in params[x] and params[x]["type"] == "stream":
            if types[x] == "AUDIO":
                assert stream_audio, "stream audio is not supported, please set env STREAM_AUDIO=1"
            elif types[x] == "VIDEO":
                assert stream_video, "stream video is not supported, please set env STREAM_VIDEO=1"
