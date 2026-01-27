"""
LightLLM Service-based Text Encoder
使用 LightLLM 服务提供的 hidden states 作为 text encoder 输出
"""

import math
import os
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from loguru import logger
from transformers import Qwen2Tokenizer, Qwen2VLProcessor

try:
    from diffusers.image_processor import VaeImageProcessor
except ImportError:
    try:
        from diffusers import VaeImageProcessor
    except ImportError:
        VaeImageProcessor = None


class LightLLMServiceTextEncoder:
    """
    基于 LightLLM 服务的 Text Encoder
    通过 HTTP API 调用 LightLLM 服务获取 hidden states
    """

    def __init__(self, config, device=None):
        from lightx2v_platform.base.global_var import AI_DEVICE

        self.config = config
        self.device = device if device is not None else AI_DEVICE
        self.dtype = torch.bfloat16

        # 配置参数
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = config["prompt_template_encode"]
        self.prompt_template_encode_start_idx = config["prompt_template_encode_start_idx"]

        self.CONDITION_IMAGE_SIZE = config.get("CONDITION_IMAGE_SIZE", 384 * 384)
        self.USE_IMAGE_ID_IN_PROMPT = config.get("USE_IMAGE_ID_IN_PROMPT", True)
        self.VAE_IMAGE_SIZE = 1024 * 1024
        self.is_layered = config.get("layered", False)
        if self.is_layered:
            self.resolution = config.get("resolution", 640)
            self.VAE_IMAGE_SIZE = self.resolution * self.resolution

        # LightLLM 服务配置
        self.service_url = config.get("service_url", "http://localhost:8010")
        self.timeout = config.get("service_timeout", 30)  # 超时时间（秒）
        self.retry_times = config.get("service_retry", 3)  # 重试次数

        # Shared Memory 模式配置（默认开启，仅在同机部署时有效）
        self.use_shm = config.get("use_shm", True)

        logger.info(f"Initializing LightLLM Service Text Encoder")
        logger.info(f"  Service URL: {self.service_url}")
        logger.info(f"  Timeout: {self.timeout}s")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Use Shared Memory: {self.use_shm}")

        # 加载必要的组件
        self.load()

    def load(self):
        """加载必要的组件（tokenizer, processor, image_processor）"""
        logger.info("Loading tokenizer and processors...")

        # 加载 tokenizer
        tokenizer_path = self.config.get("qwen25vl_tokenizer_path", os.path.join(self.config["model_path"], "tokenizer"))
        self.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)

        # 加载 processor 和 image processor（用于 i2i 任务）
        if self.config["task"] == "i2i":
            if VaeImageProcessor is None:
                raise ImportError("VaeImageProcessor could not be imported from diffusers")
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.config.get("vae_scale_factor", 8) * 2)
            processor_path = self.config.get("qwen25vl_processor_path", os.path.join(self.config["model_path"], "processor"))
            self.processor = Qwen2VLProcessor.from_pretrained(processor_path)

        logger.info("Tokenizer and processors loaded successfully")

        # 测试服务连接
        self._test_service_connection()

    def _test_service_connection(self):
        """测试与 LightLLM 服务的连接"""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"✓ Successfully connected to LightLLM service at {self.service_url}")
            else:
                logger.warning(f"⚠ LightLLM service returned status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Failed to connect to LightLLM service: {e}")
            logger.error(f"  Please ensure the service is running at {self.service_url}")
            logger.error(f"  Start with: python -m lightllm.server.api_server --return_input_hidden_states ...")

    def _call_service(self, text: str, images: Optional[List[Image.Image]] = None) -> dict:
        """
        调用 LightLLM 服务获取 hidden states

        Args:
            text: 输入文本
            images: 可选的图像列表

        Returns:
            服务返回的 JSON 响应
        """
        # 参考 test_text_encoder.py 的格式
        payload = {
            "inputs": text,
            "parameters": {
                "do_sample": False,
                "return_details": True,  # 需要此参数才能返回 hidden_states
                "max_new_tokens": 1,  # LightLLM 要求至少为 1
            },
        }

        # 如果有图像，需要按照 LightLLM 的 multimodal_params 格式
        # 参考: lightllm/server/multimodal_params.py
        if images is not None and len(images) > 0:
            import base64
            from io import BytesIO

            # 检查 prompt 中的图像 token 数量
            # Qwen2-VL 使用 <|image_pad|> 作为图像占位符
            image_token_count = text.count("<|image_pad|>")
            logger.debug(f"Found {image_token_count} image tokens in prompt, have {len(images)} images")

            # 确保图像数量与 prompt 中的图像 token 数量匹配
            if image_token_count != len(images):
                logger.warning(f"Image token count ({image_token_count}) != image count ({len(images)}), adjusting to match prompt")
                # 如果 prompt 中有多个图像 token，但只提供了 1 个图像，重复使用该图像
                if len(images) == 1 and image_token_count > 1:
                    images = images * image_token_count
                    logger.debug(f"Repeated image {image_token_count} times to match prompt")
                elif image_token_count == 0:
                    logger.warning("No image tokens found in prompt, skipping image transmission")
                    images = []

            # LightLLM 期望的格式：multimodal_params.images 是 ImageItem 列表
            # 每个 ImageItem 需要 {"type": "base64", "data": "base64_string"}
            # 优化：使用 JPEG 格式（编码更快，传输更小）
            image_items = []
            for idx, img in enumerate(images):
                buffered = BytesIO()
                # BMP 格式：无损且编码极快 (也就是直接内存拷贝)，适合 Localhost 高带宽场景
                # 相比 PNG (CPU 压缩慢) 和 JPEG (有损)，BMP 是由于 Service Mode 的最佳选择
                img.save(buffered, format="BMP")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                image_items.append({"type": "base64", "data": img_str})
                logger.debug(f"Encoded image {idx + 1}/{len(images)}: {len(img_str)} chars base64 (JPEG)")

            # 使用 multimodal_params 格式
            if len(image_items) > 0:
                payload["multimodal_params"] = {"images": image_items}
                logger.debug(f"Added {len(image_items)} images to multimodal_params")

        # 尝试使用更快的 JSON 库
        try:
            import orjson

            def fast_json_loads(data):
                return orjson.loads(data)

            logger.debug("Using orjson for fast JSON parsing")
        except ImportError:
            try:
                import ujson

                def fast_json_loads(data):
                    return ujson.loads(data)

                logger.debug("Using ujson for JSON parsing")
            except ImportError:
                import json

                def fast_json_loads(data):
                    return json.loads(data)

                logger.debug("Using standard json for JSON parsing")

        # 重试机制
        last_error = None
        for attempt in range(self.retry_times):
            try:
                logger.debug(f"Calling LightLLM service (attempt {attempt + 1}/{self.retry_times})...")
                response = requests.post(f"{self.service_url}/generate", json=payload, timeout=self.timeout)
                response.raise_for_status()

                # 使用更快的 JSON 库解析响应
                result = fast_json_loads(response.content)
                logger.debug(f"✓ Service call successful")
                return result

            except requests.exceptions.Timeout:
                last_error = f"Request timeout after {self.timeout}s"
                logger.warning(f"⚠ Attempt {attempt + 1} failed: {last_error}")
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                # 记录详细的错误信息
                if hasattr(e, "response") and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        logger.warning(f"⚠ Attempt {attempt + 1} failed: {last_error}")
                        logger.debug(f"  Error detail: {error_detail}")
                    except Exception:
                        logger.warning(f"⚠ Attempt {attempt + 1} failed: {last_error}")
                        logger.debug(f"  Response text: {e.response.text[:200] if hasattr(e.response, 'text') else 'N/A'}")
                else:
                    logger.warning(f"⚠ Attempt {attempt + 1} failed: {last_error}")

            if attempt < self.retry_times - 1:
                import time

                time.sleep(1)  # 重试前等待1秒

        # 所有重试都失败
        raise RuntimeError(f"Failed to call LightLLM service after {self.retry_times} attempts. Last error: {last_error}")

    @torch.inference_mode()
    def infer(self, text: List[str], image_list: Optional[List] = None) -> Tuple:
        """
        推理方法 - 调用 LightLLM 服务获取 hidden states

        Args:
            text: 文本提示列表
            image_list: 可选的图像列表（用于 i2i 任务）

        Returns:
            (prompt_embeds, prompt_embeds_mask, image_info)
        """
        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx

        # 准备图像信息
        if image_list is not None:
            condition_image_list = []
            vae_image_list = []
            condition_image_info_list = []
            vae_image_info_list = []

            if self.USE_IMAGE_ID_IN_PROMPT:
                base_img_prompt = ""
                img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
                for i, image in enumerate(image_list):
                    base_img_prompt += img_prompt_template.format(i + 1)
                    condition_image, vae_image, condition_image_info, vae_image_info = self.preprocess_image(image)
                    condition_image_list.append(condition_image)
                    vae_image_list.append(vae_image)
                    condition_image_info_list.append(condition_image_info)
                    vae_image_info_list.append(vae_image_info)
            else:
                base_img_prompt = "<|vision_start|><|image_pad|><|vision_end|>"
                for i, image in enumerate(image_list):
                    condition_image, vae_image, condition_image_info, vae_image_info = self.preprocess_image(image)
                    condition_image_list.append(condition_image)
                    vae_image_list.append(vae_image)
                    condition_image_info_list.append(condition_image_info)
                    vae_image_info_list.append(vae_image_info)

            image_info = {
                "vae_image_list": vae_image_list,
                "vae_image_info_list": vae_image_info_list,
            }
        else:
            image_info = {}
            base_img_prompt = ""
            condition_image_list = None

        # 准备文本
        if self.config["task"] == "i2i" and not self.is_layered and image_list is not None:
            txt = template.format(base_img_prompt + text[0])
        else:
            txt = template.format(text[0])

        # 调用 LightLLM 服务
        logger.debug(f"Calling LightLLM service with text: {txt[:100]}...")
        logger.debug(f"  Image count: {len(condition_image_list) if condition_image_list else 0}")
        logger.debug(f"  Base image prompt: {base_img_prompt[:50] if base_img_prompt else 'None'}...")
        result = self._call_service(txt, condition_image_list)

        # 解析返回的 hidden states
        # 优先使用 Shared Memory 模式（零拷贝，最快）
        if self.use_shm and "shm_hidden_states_name" in result:
            from .shm_client import get_shm_client

            shm_name = result["shm_hidden_states_name"]
            shape = tuple(result["shm_hidden_states_shape"])
            try:
                shm_client = get_shm_client()
                hidden_states_np = shm_client.read_hidden_states(shm_name, shape, dtype=np.uint8)
                hidden_states = torch.from_numpy(hidden_states_np).to(device=self.device, non_blocking=True)
                logger.debug(f"✓ Read hidden states from shared memory: {shm_name}")
            except Exception as e:
                logger.warning(f"Failed to read from shared memory '{shm_name}': {e}, falling back to HTTP mode")
                # Fallback to base64 mode
                if "hidden_states_base64" in result:
                    import base64
                    data_bytes = base64.b64decode(result["hidden_states_base64"])
                    shape = result["hidden_states_shape"]
                    hidden_states_np = np.frombuffer(data_bytes, dtype=np.uint8).reshape(shape)
                    hidden_states = torch.from_numpy(hidden_states_np).to(device=self.device, non_blocking=True)
                else:
                    raise

        elif "hidden_states_base64" in result:
            import base64

            # Decode base64 to bytes
            data_bytes = base64.b64decode(result["hidden_states_base64"])
            shape = result["hidden_states_shape"]
            # Create numpy array from buffer (zero copy if possible, but base64 decode creates bytes)
            hidden_states_np = np.frombuffer(data_bytes, dtype=np.uint8).reshape(shape)
            hidden_states = torch.from_numpy(hidden_states_np).to(device=self.device, non_blocking=True)

        elif "hidden_states" in result:
            # Legacy path
            hidden_states_data = result["hidden_states"]

            # 优化：根据数据类型选择最快的转换方式
            if isinstance(hidden_states_data, list):
                # 列表格式：检查是否是扁平列表或嵌套列表
                if len(hidden_states_data) > 0 and isinstance(hidden_states_data[0], list):
                    # 嵌套列表：使用 numpy 转换（比 torch.tensor 快）
                    hidden_states_np = np.array(hidden_states_data, dtype=np.uint8)
                else:
                    # 扁平列表：使用 numpy frombuffer 更快（如果数据支持）
                    try:
                        # 尝试使用 memoryview 加速
                        hidden_states_np = np.array(hidden_states_data, dtype=np.uint8)
                    except Exception:
                        hidden_states_np = np.array(hidden_states_data, dtype=np.uint8)
                hidden_states = torch.from_numpy(hidden_states_np).to(device=self.device, non_blocking=True)
            elif isinstance(hidden_states_data, np.ndarray):
                # numpy array 格式：直接转换
                if hidden_states_data.dtype != np.uint8:
                    hidden_states_data = hidden_states_data.astype(np.uint8)
                hidden_states = torch.from_numpy(hidden_states_data).to(device=self.device, non_blocking=True)
            else:
                # 其他格式
                hidden_states_np = np.array(hidden_states_data, dtype=np.uint8)
                hidden_states = torch.from_numpy(hidden_states_np).to(device=self.device, non_blocking=True)
        else:
            raise ValueError(f"LightLLM service response missing 'hidden_states' or 'hidden_states_base64'. Response keys: {result.keys()}")

        # 关键步骤：将 uint8 tensor 通过 view 转换为 bfloat16
        hidden_states = hidden_states.view(torch.bfloat16)

        logger.debug(f"Converted hidden states: shape={hidden_states.shape}, dtype={hidden_states.dtype}, device={hidden_states.device}")

        # 后处理：去除 drop_idx 和调整形状
        # 假设 hidden_states 形状为 [batch, seq_len, hidden_dim]
        if len(hidden_states.shape) == 2:
            # 如果是 [seq_len, hidden_dim]，添加 batch 维度
            hidden_states = hidden_states.unsqueeze(0)

        # 去除 prompt template 的前缀 tokens
        if drop_idx > 0 and hidden_states.shape[1] > drop_idx:
            hidden_states = hidden_states[:, drop_idx:, :]

        # 创建 attention mask
        seq_len = hidden_states.shape[1]
        attention_mask = torch.ones(hidden_states.shape[0], seq_len, dtype=torch.long, device=self.device)

        prompt_embeds = hidden_states
        prompt_embeds_mask = attention_mask

        logger.info(f"✓ LightLLM service inference complete: prompt_embeds shape={prompt_embeds.shape}")

        return prompt_embeds, prompt_embeds_mask, image_info

    def _calculate_dimensions(self, target_area, ratio):
        """计算目标尺寸"""
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height

    def preprocess_image(self, image):
        """预处理图像 (带简单缓存)"""
        # 使用 image id 作为缓存键 (假设同一对象内容不变，适用于 diffusers pipeline 循环)
        img_id = id(image)
        if hasattr(self, "_image_cache") and img_id in self._image_cache:
            return self._image_cache[img_id]

        image_width, image_height = image.size
        condition_width, condition_height = self._calculate_dimensions(self.CONDITION_IMAGE_SIZE, image_width / image_height)
        vae_width, vae_height = self._calculate_dimensions(self.VAE_IMAGE_SIZE, image_width / image_height)
        condition_image = self.image_processor.resize(image, condition_height, condition_width)
        vae_image = self.image_processor.preprocess(image, vae_height, vae_width).unsqueeze(2)

        result = (condition_image, vae_image, (condition_height, condition_width), (vae_height, vae_width))

        # 初始化缓存 (如果不存在)
        if not hasattr(self, "_image_cache"):
            self._image_cache = {}

        # 简单缓存管理：如果太大则清空
        if len(self._image_cache) > 50:
            self._image_cache.clear()

        self._image_cache[img_id] = result
        return result

    def offload_to_cpu(self):
        """服务化版本无需 offload"""
        logger.debug("Service-based encoder: offload_to_cpu() is a no-op")

    def reload_to_device(self):
        """服务化版本无需 reload"""
        logger.debug("Service-based encoder: reload_to_device() is a no-op")
