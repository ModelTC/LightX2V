"""Bilibili RTMP output with text padding or a translucent in-frame panel."""

import subprocess
import threading

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

from lightx2v.utils.va_recorder import VARecorder

from .config import json_option, validate_padding_config


class BiliLiveVARecorder(VARecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_config = json_option(self.stream_config, "padding_config", {})
        validate_padding_config(self.padding_config)
        self.text_mode = self.padding_config.get("mode", "padding")
        self.padding_direction = "right"
        self.padding_width = 0
        self.padding_height = 0
        self.padding_tensor = None
        self.overlay_alpha = None
        self.overlay_x = 0
        self.overlay_y = 0
        self.padding_lock = threading.Lock()
        self.start_texts = self.padding_config.get("start_texts", ["AI虚拟主播", "发送弹幕与我互动"])

    def config_video_padding(self) -> None:
        if self.text_mode == "overlay":
            self.overlay_x = int(self.padding_config.get("x", 20))
            self.overlay_y = int(self.padding_config.get("y", 20))
            if self.overlay_x >= self.width or self.overlay_y >= self.height:
                raise ValueError("padding_config.x and y must be inside the video frame")
            self.padding_width = min(int(self.padding_config.get("width", 400)), self.width - self.overlay_x)
            self.padding_height = min(int(self.padding_config.get("height", self.height - self.overlay_y)), self.height - self.overlay_y)
            self.update_padding_texts([])
            return
        ratio = float(self.padding_config.get("target_ratio", 0))
        if ratio == 0:
            return
        if not 0.33 <= abs(ratio) <= 3:
            raise ValueError("padding_config.target_ratio must have absolute value in [0.33, 3]")
        if abs(ratio) > self.width / self.height:
            self.padding_width = max(0, round(self.height * abs(ratio)) - self.width)
            self.padding_width += self.padding_width % 2
            self.padding_height = self.height
            self.padding_direction = "right" if ratio > 0 else "left"
        else:
            self.padding_height = max(0, round(self.width / abs(ratio)) - self.height)
            self.padding_height += self.padding_height % 2
            self.padding_width = self.width
            self.padding_direction = "bottom" if ratio > 0 else "top"
        if min(self.padding_width, self.padding_height) > 0:
            self.update_padding_texts([])

    def update_padding_texts(self, texts: list[str]) -> None:
        if min(self.padding_width, self.padding_height) <= 0:
            return
        config = self.padding_config
        overlay = self.text_mode == "overlay"
        font_size = int(config.get("font_size", 23))
        font_path = config.get("font_path", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            logger.warning("[Bili] Text font unavailable: {}; install fonts-noto-cjk or set padding_config.font_path", font_path)
            font = ImageFont.load_default()
        x, y = int(config.get("start_x", 10)), int(config.get("start_y", 10))
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        lines = []
        for text in self.start_texts + list(texts):
            for paragraph in str(text).split("\n"):
                line = ""
                for character in paragraph:
                    if line and draw.textlength(line + character, font=font) > max(1, self.padding_width - 2 * x):
                        lines.append(line)
                        line = ""
                    line += character
                lines.append(line)
        bbox = font.getbbox("Ag国")
        text_height = max(font_size, bbox[3] - bbox[1])
        spacing = int(config.get("spacing_line", 10))
        line_height = text_height + spacing
        max_lines = max(0, (self.padding_height - 2 * y + spacing) // line_height)
        lines = lines[:max_lines]
        height = self.padding_height
        if overlay and "height" not in config:
            height = min(height, max(1, 2 * y + len(lines) * line_height - (spacing if lines else 0)))
        bg_color = tuple(config.get("bg_color", [0, 0, 0] if overlay else [255, 255, 255]))
        color = tuple(config.get("text_color", [255, 255, 255] if overlay else [0, 0, 0]))
        if overlay:
            bg_color += (round(255 * float(config.get("bg_opacity", 0.55))),)
            color += (255,)
        image = Image.new("RGBA" if overlay else "RGB", (self.padding_width, height), bg_color)
        draw = ImageDraw.Draw(image)
        for line in lines:
            draw.text((x, y), line, font=font, fill=color, anchor="lt")
            y += line_height
        tensor = torch.from_numpy(np.asarray(image).copy()).unsqueeze(0).float().div_(255)
        with self.padding_lock:
            self.padding_tensor = tensor[..., :3]
            self.overlay_alpha = tensor[..., 3:] if overlay else None

    def padding_video_frames(self, frames: torch.Tensor) -> torch.Tensor:
        with self.padding_lock:
            padding = self.padding_tensor
            alpha = self.overlay_alpha
        if padding is None:
            return frames
        if self.text_mode == "overlay":
            padding = padding.to(device=frames.device, dtype=frames.dtype)
            alpha = alpha.to(device=frames.device, dtype=frames.dtype)
            height, width = padding.shape[1:3]
            output = frames.clone()
            region = output[:, self.overlay_y : self.overlay_y + height, self.overlay_x : self.overlay_x + width]
            region.copy_(region * (1 - alpha) + padding * alpha)
            return output
        padding = padding.to(device=frames.device, dtype=frames.dtype).expand(frames.shape[0], -1, -1, -1)
        dimension = 2 if self.padding_direction in {"left", "right"} else 1
        items = [padding, frames] if self.padding_direction in {"left", "top"} else [frames, padding]
        return torch.cat(items, dim=dimension)

    def start_ffmpeg_process_rtmp(self) -> None:
        width, height = self.width, self.height
        if self.text_mode == "padding" and self.padding_tensor is not None:
            if self.padding_direction in {"left", "right"}:
                width += self.padding_width
            else:
                height += self.padding_height
        command = [
            "ffmpeg",
            "-re",
            "-f",
            "s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-i",
            "tcp://127.0.0.1:%s" % self.audio_port,
            "-f",
            "rawvideo",
            "-re",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-s",
            "%sx%s" % (width, height),
            "-i",
            "tcp://127.0.0.1:%s" % self.video_port,
        ]
        if self.stream_config.get("rtmp_vf"):
            command.extend(["-vf", self.stream_config["rtmp_vf"]])
        command.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-ar",
                "44100",
                "-c:v",
                "libx264",
                "-b:v",
                "2M",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-g",
                str(round(self.fps)),
                "-pix_fmt",
                "yuv420p",
                "-f",
                "flv",
                self.livestream_url,
                "-y",
                "-loglevel",
                self.ffmpeg_log_level,
            ]
        )
        self.ffmpeg_process = subprocess.Popen(command)
        logger.info("Bilibili RTMP encoder started, PID={}, input_audio_sr={}, fps={}", self.ffmpeg_process.pid, self.sample_rate, self.fps)
