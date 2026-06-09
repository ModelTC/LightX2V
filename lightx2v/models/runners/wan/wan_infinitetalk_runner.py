import gc
import math
import os
import subprocess
import tempfile

import imageio_ffmpeg as ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from loguru import logger

from lightx2v.models.input_encoders.hf.infinitetalk.audio_encoder import InfiniteTalkAudioEncoder
from lightx2v.models.networks.wan.infinitetalk_model import WanInfiniteTalkModel
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.infinitetalk.scheduler import InfiniteTalkScheduler
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.audio_io import load_audio_file
from lightx2v.utils.envs import GET_DTYPE, GET_RECORDER_MODE
from lightx2v.utils.input_info import UNSET
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import is_main_process, save_to_video, wan_vae_to_comfy
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)

try:
    import librosa
except ImportError:
    librosa = None

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None


VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".mpeg", ".mpg")

ASPECT_RATIO_627 = {
    "0.26": ([320, 1216], 1),
    "0.38": ([384, 1024], 1),
    "0.50": ([448, 896], 1),
    "0.67": ([512, 768], 1),
    "0.82": ([576, 704], 1),
    "1.00": ([640, 640], 1),
    "1.22": ([704, 576], 1),
    "1.50": ([768, 512], 1),
    "1.86": ([832, 448], 1),
    "2.00": ([896, 448], 1),
    "2.50": ([960, 384], 1),
    "2.83": ([1088, 384], 1),
    "3.60": ([1152, 320], 1),
    "3.80": ([1216, 320], 1),
    "4.00": ([1280, 320], 1),
}

ASPECT_RATIO_960 = {
    "0.22": ([448, 2048], 1),
    "0.29": ([512, 1792], 1),
    "0.36": ([576, 1600], 1),
    "0.45": ([640, 1408], 1),
    "0.55": ([704, 1280], 1),
    "0.63": ([768, 1216], 1),
    "0.76": ([832, 1088], 1),
    "0.88": ([896, 1024], 1),
    "1.00": ([960, 960], 1),
    "1.14": ([1024, 896], 1),
    "1.31": ([1088, 832], 1),
    "1.50": ([1152, 768], 1),
    "1.58": ([1216, 768], 1),
    "1.82": ([1280, 704], 1),
    "1.91": ([1344, 704], 1),
    "2.20": ([1408, 640], 1),
    "2.30": ([1472, 640], 1),
    "2.67": ([1536, 576], 1),
    "2.89": ([1664, 576], 1),
    "3.62": ([1856, 512], 1),
    "3.75": ([1920, 512], 1),
}


def _is_video(path):
    return os.path.splitext(path)[1].lower() in VID_EXTENSIONS


@RUNNER_REGISTER("infinitetalk")
class InfiniteTalkRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)
        assert self.config["task"] == "s2v", "InfiniteTalk runner expects task=s2v"
        self.audio_sample_rate = int(self.config.get("audio_sample_rate", 16000))
        self.target_fps = int(self.config.get("target_fps", 25))
        self.video_audio_path = None

    def init_scheduler(self):
        self.scheduler = InfiniteTalkScheduler(self.config)

    def init_modules(self):
        logger.info("Initializing InfiniteTalk runner modules...")
        self.load_model()
        self.model.set_scheduler(self.scheduler)
        self.run_input_encoder = self._run_input_encoder_local_s2v
        self.config.lock()

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.image_encoder = self.load_image_encoder()
        self.vae_encoder, self.vae_decoder = self.load_vae()
        self.audio_encoder = self.load_audio_encoder()
        self.vfi_model = None
        self.vsr_model = None

    def load_transformer(self):
        return WanInfiniteTalkModel(self.config["model_path"], self.config, self.init_device)

    def load_audio_encoder(self):
        audio_encoder_path = self.config.get("audio_encoder_path", None)
        if audio_encoder_path is None:
            raise ValueError("InfiniteTalk requires audio_encoder_path in config.")
        device = self.config.get("wav2vec_device", "cpu")
        return InfiniteTalkAudioEncoder(audio_encoder_path, device=device, fps=self.target_fps, sample_rate=self.audio_sample_rate)

    @ProfilingContext4DebugL1(
        "Run Text Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_text_encode_duration,
        metrics_labels=["InfiniteTalkRunner"],
    )
    def run_text_encoder(self, input_info):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_encoders = self.load_text_encoder()

        prompt = input_info.prompt_enhanced if self.config["use_prompt_enhancer"] else input_info.prompt
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_input_prompt_len.observe(len(prompt))

        context = self.text_encoders[0].infer([prompt])
        context = torch.stack([torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))]) for u in context])
        if self.config.get("enable_cfg", False):
            context_null = self.text_encoders[0].infer([input_info.negative_prompt])
            context_null = torch.stack([torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))]) for u in context_null])
        else:
            context_null = None

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.text_encoders[0]
            torch_device_module.empty_cache()
            gc.collect()

        return {
            "context": context,
            "context_null": context_null,
        }

    def _load_input_data(self):
        cfg_input = self.config.get("infinitetalk_input", None)
        if cfg_input is not None:
            data = dict(cfg_input)
            data["cond_audio"] = dict(data["cond_audio"])
        else:
            cond_audio = self.config.get("cond_audio", None)
            if cond_audio is not None:
                cond_audio = dict(cond_audio)
            else:
                audio_path = getattr(self.input_info, "audio_path", "") or self.config.get("audio_path", "")
                audio_paths = [item.strip() for item in audio_path.split(",") if item.strip()]
                if len(audio_paths) == 1:
                    cond_audio = {"person1": audio_paths[0]}
                elif len(audio_paths) == 2:
                    cond_audio = {"person1": audio_paths[0], "person2": audio_paths[1]}
                else:
                    cond_audio = {}

            data = {
                "prompt": getattr(self.input_info, "prompt", "") or self.config.get("prompt", ""),
                "cond_video": getattr(self.input_info, "image_path", "") or self.config.get("cond_video", "") or self.config.get("image_path", ""),
                "cond_audio": cond_audio,
            }
            if self.config.get("audio_type", None):
                data["audio_type"] = self.config["audio_type"]
            if self.config.get("bbox", None):
                data["bbox"] = self.config["bbox"]

        if not data.get("prompt"):
            raise ValueError("InfiniteTalk requires prompt from --prompt or config infinitetalk_input/prompt.")
        if not data.get("cond_video"):
            raise ValueError("InfiniteTalk requires cond_video from --image_path or config.")
        if not data.get("cond_audio"):
            raise ValueError("InfiniteTalk requires cond_audio from --audio_path or config.")

        return data

    @staticmethod
    def _loudness_norm(audio_array, sr=16000, lufs=-23):
        if pyln is None:
            return audio_array
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > 100:
            return audio_array
        return pyln.normalize.loudness(audio_array, loudness, lufs)

    def _extract_audio_from_video(self, filename):
        audio_dir = self.config.get("audio_save_dir", os.path.join(os.getcwd(), "save_results", "infinitetalk_audio"))
        os.makedirs(audio_dir, exist_ok=True)
        raw_audio_path = os.path.join(audio_dir, os.path.splitext(os.path.basename(filename))[0] + "_raw.wav")
        cmd = [
            ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(filename),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self.audio_sample_rate),
            "-ac",
            "2",
            raw_audio_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        audio = self._load_audio_array(raw_audio_path)
        try:
            os.remove(raw_audio_path)
        except OSError:
            pass
        return audio

    def _load_audio_array(self, audio_path):
        if _is_video(audio_path):
            return self._extract_audio_from_video(audio_path)
        if librosa is not None:
            audio_array, sr = librosa.load(audio_path, sr=self.audio_sample_rate)
            return self._loudness_norm(audio_array, sr)

        audio_tensor, sr = load_audio_file(audio_path)
        audio_tensor = audio_tensor.float().mean(0)
        if sr != self.audio_sample_rate:
            target_len = int(round(audio_tensor.numel() * self.audio_sample_rate / sr))
            audio_tensor = F.interpolate(audio_tensor.view(1, 1, -1), size=target_len, mode="linear", align_corners=True).view(-1)
        return self._loudness_norm(audio_tensor.cpu().numpy(), self.audio_sample_rate)

    def _audio_prepare_single(self, audio_path):
        return self._load_audio_array(audio_path)

    def _audio_prepare_multi(self, left_path, right_path, audio_type):
        if not (left_path == "None" or right_path == "None"):
            speech1 = self._audio_prepare_single(left_path)
            speech2 = self._audio_prepare_single(right_path)
        elif left_path == "None":
            speech2 = self._audio_prepare_single(right_path)
            speech1 = np.zeros(speech2.shape[0], dtype=speech2.dtype)
        else:
            speech1 = self._audio_prepare_single(left_path)
            speech2 = np.zeros(speech1.shape[0], dtype=speech1.dtype)

        if audio_type == "para":
            new_speech1 = speech1
            new_speech2 = speech2
        elif audio_type == "add":
            new_speech1 = np.concatenate([speech1[: speech1.shape[0]], np.zeros(speech2.shape[0], dtype=speech1.dtype)])
            new_speech2 = np.concatenate([np.zeros(speech1.shape[0], dtype=speech2.dtype), speech2[: speech2.shape[0]]])
        else:
            raise ValueError(f"Unsupported InfiniteTalk audio_type: {audio_type}")
        return new_speech1, new_speech2, new_speech1 + new_speech2

    def _write_sum_audio(self, input_data, audio_arrays):
        if sf is not None:
            fd, audio_path = tempfile.mkstemp(prefix="infinitetalk_sum_", suffix=".wav")
            os.close(fd)
            sf.write(audio_path, audio_arrays, self.audio_sample_rate)
            self.video_audio_path = audio_path
        else:
            logger.warning("soundfile is unavailable; generated video will be saved without muxed audio.")
            self.video_audio_path = None

    def _remove_video_audio_path(self):
        audio_path = self.video_audio_path
        self.video_audio_path = None
        if audio_path and os.path.isfile(audio_path):
            try:
                os.remove(audio_path)
            except OSError as exc:
                logger.warning(f"Failed to remove temporary audio file {audio_path}: {exc}")

    def _load_or_encode_audio(self, audio_path_or_array):
        if isinstance(audio_path_or_array, np.ndarray):
            return self.audio_encoder.infer(audio_path_or_array)
        if str(audio_path_or_array).endswith((".pt", ".pth")):
            return torch.load(audio_path_or_array, map_location="cpu")
        return self.audio_encoder.infer(self._audio_prepare_single(audio_path_or_array))

    def _prepare_audio_embeddings(self, input_data):
        cond_audio = input_data["cond_audio"]
        if len(cond_audio) == 2:
            audio_type = input_data.get("audio_type", "para")
            speech1, speech2, sum_speech = self._audio_prepare_multi(cond_audio["person1"], cond_audio["person2"], audio_type)
            self._write_sum_audio(input_data, sum_speech)
            return [self._load_or_encode_audio(speech1), self._load_or_encode_audio(speech2)]

        speech = self._audio_prepare_single(cond_audio["person1"])
        self._write_sum_audio(input_data, speech)
        return [self._load_or_encode_audio(speech)]

    def _extract_specific_frame(self, video_path, frame_id):
        if not _is_video(video_path):
            return Image.open(video_path).convert("RGB")
        if VideoReader is None:
            raise ImportError("decord is required for InfiniteTalk video cond_video inputs.")
        vr = VideoReader(video_path, ctx=cpu(0))
        if frame_id < len(vr):
            frame = vr[frame_id].asnumpy()
        else:
            frame = vr[-1].asnumpy()
        del vr
        gc.collect()
        return Image.fromarray(frame)

    @staticmethod
    def _resize_and_centercrop(cond_image, target_size):
        if isinstance(cond_image, torch.Tensor):
            _, orig_h, orig_w = cond_image.shape
        else:
            orig_h, orig_w = cond_image.height, cond_image.width

        target_h, target_w = target_size
        scale = max(target_h / orig_h, target_w / orig_w)
        final_h = math.ceil(scale * orig_h)
        final_w = math.ceil(scale * orig_w)

        if isinstance(cond_image, torch.Tensor):
            resized = F.interpolate(cond_image.unsqueeze(0), size=(final_h, final_w), mode="nearest").contiguous()
            return TF.center_crop(resized, target_size).squeeze(0)

        resized = cond_image.resize((final_w, final_h), resample=Image.BILINEAR)
        resized_tensor = torch.from_numpy(np.array(resized))[None].permute(0, 3, 1, 2).contiguous()
        cropped = TF.center_crop(resized_tensor, target_size)
        return cropped[:, :, None, :, :]

    def _select_target_size(self, image):
        bucket_config = ASPECT_RATIO_960 if self.config.get("infinitetalk_size", "infinitetalk-720") == "infinitetalk-720" else ASPECT_RATIO_627
        src_h, src_w = image.height, image.width
        ratio = src_h / src_w
        closest_bucket = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x) - ratio))[0]
        target_h, target_w = bucket_config[closest_bucket][0]
        return src_h, src_w, target_h, target_w

    def _prepare_cond_image(self, frame_id):
        image = self._extract_specific_frame(self.cond_file_path, frame_id)
        image = self._resize_and_centercrop(image, (self.target_h, self.target_w))
        image = image.float() / 255.0
        image = (image - 0.5) * 2
        return image.to(AI_DEVICE)

    def _build_ref_target_masks(self, human_num, latent_h, latent_w):
        human_masks = []
        if human_num == 1:
            background_mask = torch.ones([self.src_h, self.src_w])
            human_mask1 = torch.ones([self.src_h, self.src_w])
            human_mask2 = torch.ones([self.src_h, self.src_w])
            human_masks = [human_mask1, human_mask2, background_mask]
        else:
            if "bbox" in self.input_data:
                background_mask = torch.zeros([self.src_h, self.src_w])
                for _, person_bbox in self.input_data["bbox"].items():
                    x_min, y_min, x_max, y_max = person_bbox
                    human_mask = torch.zeros([self.src_h, self.src_w])
                    human_mask[int(x_min) : int(x_max), int(y_min) : int(y_max)] = 1
                    background_mask += human_mask
                    human_masks.append(human_mask)
            else:
                face_scale = float(self.config.get("face_scale", 0.05))
                x_min, x_max = int(self.src_h * face_scale), int(self.src_h * (1 - face_scale))
                human_mask1 = torch.zeros([self.src_h, self.src_w])
                human_mask2 = torch.zeros([self.src_h, self.src_w])
                background_mask = torch.zeros([self.src_h, self.src_w])
                lefty_min, lefty_max = int((self.src_w // 2) * face_scale), int((self.src_w // 2) * (1 - face_scale))
                righty_min = int((self.src_w // 2) * face_scale + (self.src_w // 2))
                righty_max = int((self.src_w // 2) * (1 - face_scale) + (self.src_w // 2))
                human_mask1[x_min:x_max, lefty_min:lefty_max] = 1
                human_mask2[x_min:x_max, righty_min:righty_max] = 1
                background_mask += human_mask1 + human_mask2
                human_masks = [human_mask1, human_mask2]
            background_mask = torch.where(background_mask > 0, torch.tensor(0), torch.tensor(1))
            human_masks.append(background_mask)

        masks = torch.stack(human_masks, dim=0)
        masks = self._resize_and_centercrop(masks, (self.target_h, self.target_w))
        masks = F.interpolate(masks.unsqueeze(0), size=(latent_h, latent_w), mode="nearest").squeeze(0)
        return (masks > 0).float().to(AI_DEVICE)

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_s2v(self):
        input_data = self._load_input_data()
        if self.input_info.prompt:
            input_data["prompt"] = self.input_info.prompt
        self.input_info.prompt = input_data["prompt"]

        self.input_data = input_data
        self.cond_file_path = input_data["cond_video"]
        first_image = self._extract_specific_frame(self.cond_file_path, 0)
        self.src_h, self.src_w, self.target_h, self.target_w = self._select_target_size(first_image)

        full_audio_embs = self._prepare_audio_embeddings(input_data)
        if any(audio_emb.shape[0] <= 0 for audio_emb in full_audio_embs):
            raise ValueError("InfiniteTalk audio embeddings must be non-empty.")

        text_encoder_output = self.run_text_encoder(self.input_info)
        return {
            "text_encoder_output": text_encoder_output,
            "full_audio_embs": full_audio_embs,
            "human_num": len(full_audio_embs),
            "seed": self.input_info.seed,
        }

    def _slice_audio_embeddings(self, full_audio_embs, audio_start_idx, audio_end_idx):
        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        audio_embs = []
        for full_audio_emb in full_audio_embs:
            center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
            center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
            audio_embs.append(full_audio_emb[center_indices][None])
        return torch.concat(audio_embs, dim=0).to(AI_DEVICE, GET_DTYPE())

    def _build_vae_encoder_out(self, cond_image, frame_num):
        video_frames = torch.zeros(1, cond_image.shape[1], frame_num - cond_image.shape[2], self.target_h, self.target_w, device=AI_DEVICE)
        padding_frames = torch.concat([cond_image, video_frames], dim=2)
        y = self.vae_encoder.encode(padding_frames.to(GET_DTYPE())).to(GET_DTYPE())

        latent_h, latent_w = y.shape[-2:]
        msk = torch.ones(1, frame_num, latent_h, latent_w, device=AI_DEVICE)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, latent_h, latent_w)
        msk = msk.transpose(1, 2).to(GET_DTYPE())[0]
        return torch.concat([msk, y], dim=0)

    def _build_clip_context(self, cond_image):
        return self.run_image_encoder(cond_image[:, :, -1])

    def _run_dit_clip(self, dit_inputs):
        infer_steps = self.scheduler.infer_steps
        for step_index in range(infer_steps):
            logger.info(f"==> step_index: {step_index + 1} / {infer_steps}")
            with ProfilingContext4DebugL1("step_pre"):
                self.scheduler.step_pre(step_index)
            with ProfilingContext4DebugL1("infer_main"):
                self.model.infer(dit_inputs)
            with ProfilingContext4DebugL1("step_post"):
                self.scheduler.step_post()

    def _resolve_video_duration(self):
        video_duration = getattr(self.input_info, "video_duration", UNSET)
        if video_duration is UNSET or video_duration is None:
            video_duration = self.config.get("video_duration", None)
        if video_duration is None:
            return None
        return float(video_duration)

    def _resolve_expected_frames(self):
        audio_frames = min(int(audio_emb.shape[0]) for audio_emb in self.full_audio_embs)
        if audio_frames <= 0:
            raise ValueError("InfiniteTalk audio embeddings must be non-empty.")

        video_duration = self._resolve_video_duration()
        if video_duration is None:
            requested_frames = audio_frames
        else:
            requested_frames = max(1, int(video_duration * self.target_fps))

        expected_frames = min(requested_frames, audio_frames)
        if expected_frames < requested_frames:
            logger.warning(
                f"Input video_duration is greater than actual audio duration, using audio duration instead: audio_duration={audio_frames / self.target_fps}, video_duration={video_duration}"
            )
        return expected_frames

    def _segment_start_frame(self, segment_idx):
        return segment_idx * self.segment_stride

    def _ensure_audio_padding(self, audio_end_idx):
        for idx, full_audio_emb in enumerate(self.full_audio_embs):
            if audio_end_idx < full_audio_emb.shape[0]:
                continue
            miss_length = audio_end_idx - full_audio_emb.shape[0] + 3
            add_audio_emb = torch.flip(full_audio_emb[-miss_length:], dims=[0])
            self.full_audio_embs[idx] = torch.cat([full_audio_emb, add_audio_emb], dim=0)

    def init_run(self):
        self.frame_num = int(self.config["target_video_length"])
        input_frame_num = getattr(self.input_info, "target_video_length", UNSET)
        if input_frame_num is not UNSET and input_frame_num is not None and input_frame_num > 0:
            self.frame_num = int(input_frame_num)
        self.motion_frame = int(self.config.get("motion_frame", 9))
        self.segment_stride = self.frame_num - self.motion_frame
        if self.segment_stride <= 0:
            raise ValueError(f"motion_frame must be smaller than target_video_length, got motion_frame={self.motion_frame}, target_video_length={self.frame_num}")

        self.full_audio_embs = list(self.inputs["full_audio_embs"])
        self.human_num = int(self.inputs["human_num"])
        self.expected_frames = self._resolve_expected_frames()
        self.seed = self.scheduler.seed_everything(self.inputs["seed"])
        logger.info(f"InfiniteTalk seed: {self.seed}")
        logger.info(f"InfiniteTalk expected_frames: {self.expected_frames}, fps: {self.target_fps}, duration: {self.expected_frames / self.target_fps:.3f}s")

        self.cond_image = self._prepare_cond_image(0)
        self.cond_frame = None
        self.gen_video_list = []

    def get_video_segment_num(self):
        if self.expected_frames <= self.frame_num:
            self.video_segment_num = 1
        else:
            self.video_segment_num = 1 + math.ceil((self.expected_frames - self.frame_num) / self.segment_stride)
        logger.info(f"InfiniteTalk video segments: {self.video_segment_num}")

    @ProfilingContext4DebugL1(
        "Init run segment",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_init_run_segment_duration,
        metrics_labels=["InfiniteTalkRunner"],
    )
    def init_run_segment(self, segment_idx):
        self.segment_idx = segment_idx
        self.is_first_segment = segment_idx == 0
        self.current_motion_frames_num = 1 if self.is_first_segment else self.motion_frame
        self.audio_start_idx = self._segment_start_frame(segment_idx)
        self.audio_end_idx = self.audio_start_idx + self.frame_num
        self._ensure_audio_padding(self.audio_end_idx)

        audio_embs = self._slice_audio_embeddings(self.full_audio_embs, self.audio_start_idx, self.audio_end_idx)
        clip_context = self._build_clip_context(self.cond_image)
        vae_encoder_out = self._build_vae_encoder_out(self.cond_image, self.frame_num)
        latent_h, latent_w = vae_encoder_out.shape[-2:]
        cur_motion_frames_latent_num = int(1 + (self.current_motion_frames_num - 1) // 4)

        if self.is_first_segment:
            latent_motion_input = self.cond_image
        else:
            if self.cond_frame is None:
                raise RuntimeError("InfiniteTalk non-first segment requires previous decoded motion frames.")
            latent_motion_input = self.cond_frame
        latent_motion_frames = self.vae_encoder.encode(latent_motion_input.to(GET_DTYPE()))

        ref_target_masks = self._build_ref_target_masks(self.human_num, latent_h, latent_w)
        latent_shape = (16, (self.frame_num - 1) // 4 + 1, latent_h, latent_w)
        self.scheduler.prepare(
            seed=self.seed,
            latent_shape=latent_shape,
            latent_motion_frames=latent_motion_frames,
            is_first_clip=self.is_first_segment,
            cur_motion_frames_latent_num=cur_motion_frames_latent_num,
        )
        self.dit_inputs = {
            "text_encoder_output": self.inputs["text_encoder_output"],
            "image_encoder_output": {
                "clip_encoder_out": clip_context,
                "vae_encoder_out": vae_encoder_out,
            },
            "audio_encoder_output": audio_embs,
            "ref_target_masks": ref_target_masks,
        }

    def run_segment(self, segment_idx=0):
        self._run_dit_clip(self.dit_inputs)
        return self.scheduler.latents

    @ProfilingContext4DebugL1(
        "End run segment",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_end_run_segment_duration,
        metrics_labels=["InfiniteTalkRunner"],
    )
    def end_run_segment(self, segment_idx, latents):
        videos = self.run_vae_decoder(latents).cpu()
        if self.is_first_segment:
            self.gen_video_list.append(videos)
        else:
            self.gen_video_list.append(videos[:, :, self.current_motion_frames_num :])

        if segment_idx < self.video_segment_num - 1:
            self.cond_frame = videos[:, :, -self.motion_frame :].to(torch.float32).to(AI_DEVICE)
            self.cond_image = self._prepare_cond_image(self._segment_start_frame(segment_idx + 1))

        del videos
        torch.cuda.empty_cache()
        gc.collect()

    @ProfilingContext4DebugL2("Run DiT + decode")
    def run_main(self):
        self.init_run()
        self.get_video_segment_num()

        for segment_idx in range(self.video_segment_num):
            logger.info(f"start InfiniteTalk segment {segment_idx + 1}/{self.video_segment_num}")
            with ProfilingContext4DebugL1(f"segment end2end {segment_idx + 1}/{self.video_segment_num}"):
                self.init_run_segment(segment_idx)
                latents = self.run_segment(segment_idx)
                self.end_run_segment(segment_idx, latents)

        self.gen_video = torch.cat(self.gen_video_list, dim=2)[:, :, : self.expected_frames].to(torch.float32)
        return self.process_images_after_vae_decoder()

    @ProfilingContext4DebugL1("Process after vae decoder")
    def process_images_after_vae_decoder(self):
        self.gen_video_final = wan_vae_to_comfy(self.gen_video)
        if self.input_info.return_result_tensor:
            return {"video": self.gen_video_final}
        if self.input_info.save_result_path is not None and is_main_process():
            out_path = self.input_info.save_result_path
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            save_to_video(self.gen_video_final, out_path, fps=self.target_fps, method="ffmpeg")
            if self.video_audio_path and os.path.isfile(self.video_audio_path):
                try:
                    self._mux_audio(out_path, self.video_audio_path)
                finally:
                    self._remove_video_audio_path()
            logger.info(f"Video saved to {out_path}")
        return {"video": None}

    @staticmethod
    def _mux_audio(video_path, audio_path):
        tmp_path = video_path + ".tmp.mp4"
        cmd = [
            ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            tmp_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.replace(tmp_path, video_path)
            logger.info(f"Muxed audio from {audio_path}")
        except Exception as exc:
            logger.warning(f"Audio mux failed: {exc}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def end_run(self):
        self._remove_video_audio_path()
        if hasattr(self, "inputs"):
            del self.inputs
        torch.cuda.empty_cache()
        gc.collect()

    @ProfilingContext4DebugL1(
        "RUN pipeline",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_worker_request_duration,
        metrics_labels=["InfiniteTalkRunner"],
    )
    def run_pipeline(self, input_info):
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_count.inc()
        self.input_info = input_info
        self.inputs = self.run_input_encoder()
        result = self.run_main()
        self.end_run()
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        return result
