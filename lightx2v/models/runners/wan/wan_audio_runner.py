import os
import gc
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.utils.profiler import ProfilingContext4Debug, ProfilingContext
from lightx2v.models.input_encoders.hf.t5.model import T5EncoderModel
from lightx2v.models.input_encoders.hf.xlm_roberta.model import CLIPModel, WanVideoIPHandler
from lightx2v.models.networks.wan.audio_model import WanAudioModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE

from lightx2v.models.networks.wan.audio_adapter import AudioAdapter, AudioAdapterPipe, rank0_load_state_dict_from_path

from lightx2v.models.schedulers.wan.step_distill.scheduler import WanStepDistillScheduler
from lightx2v.models.schedulers.wan.audio.scheduler import EulerSchedulerTimestepFix

from loguru import logger
import torch.distributed as dist
from einops import rearrange
import torchaudio as ta
from transformers import AutoFeatureExtractor

from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

import subprocess
import warnings
from typing import Optional, Tuple, Union


def add_mask_to_frames(
    frames: np.ndarray,
    mask_rate: float = 0.1,
    rnd_state: np.random.RandomState = None,
) -> np.ndarray:
    if mask_rate is None:
        return frames

    if rnd_state is None:
        rnd_state = np.random.RandomState()

    h, w = frames.shape[-2:]
    mask = rnd_state.rand(h, w) > mask_rate
    frames = frames * mask
    return frames


def add_noise_to_frames(
    frames: np.ndarray,
    noise_mean: float = -3.0,
    noise_std: float = 0.5,
    rnd_state: np.random.RandomState = None,
) -> np.ndarray:
    if noise_mean is None or noise_std is None:
        return frames

    if rnd_state is None:
        rnd_state = np.random.RandomState()

    shape = frames.shape
    bs = 1 if len(shape) == 4 else shape[0]
    sigma = rnd_state.normal(loc=noise_mean, scale=noise_std, size=(bs,))
    sigma = np.exp(sigma)
    sigma = np.expand_dims(sigma, axis=tuple(range(1, len(shape))))
    noise = rnd_state.randn(*shape) * sigma
    frames = frames + noise
    return frames


def get_crop_bbox(ori_h, ori_w, tgt_h, tgt_w):
    tgt_ar = tgt_h / tgt_w
    ori_ar = ori_h / ori_w
    if abs(ori_ar - tgt_ar) < 0.01:
        return 0, ori_h, 0, ori_w
    if ori_ar > tgt_ar:
        crop_h = int(tgt_ar * ori_w)
        y0 = (ori_h - crop_h) // 2
        y1 = y0 + crop_h
        return y0, y1, 0, ori_w
    else:
        crop_w = int(ori_h / tgt_ar)
        x0 = (ori_w - crop_w) // 2
        x1 = x0 + crop_w
        return 0, ori_h, x0, x1


def isotropic_crop_resize(frames: torch.Tensor, size: tuple):
    """
    frames: (T, C, H, W)
    size: (H, W)
    """
    ori_h, ori_w = frames.shape[2:]
    h, w = size
    y0, y1, x0, x1 = get_crop_bbox(ori_h, ori_w, h, w)
    cropped_frames = frames[:, :, y0:y1, x0:x1]
    resized_frames = resize(cropped_frames, size, InterpolationMode.BICUBIC, antialias=True)
    return resized_frames


def adaptive_resize(img):
    bucket_config = {
        0.667: (np.array([[480, 832], [544, 960], [720, 1280]], dtype=np.int64), np.array([0.2, 0.5, 0.3])),
        1.0: (np.array([[480, 480], [576, 576], [704, 704], [960, 960]], dtype=np.int64), np.array([0.1, 0.1, 0.5, 0.3])),
        1.5: (np.array([[480, 832], [544, 960], [720, 1280]], dtype=np.int64)[:, ::-1], np.array([0.2, 0.5, 0.3])),
    }
    ori_height = img.shape[-2]
    ori_weight = img.shape[-1]
    ori_ratio = ori_height / ori_weight
    aspect_ratios = np.array(np.array(list(bucket_config.keys())))
    closet_aspect_idx = np.argmin(np.abs(aspect_ratios - ori_ratio))
    closet_ratio = aspect_ratios[closet_aspect_idx]
    if ori_ratio < 1.0:
        target_h, target_w = 480, 832
    elif ori_ratio == 1.0:
        target_h, target_w = 480, 480
    else:
        target_h, target_w = 832, 480
    for resolution in bucket_config[closet_ratio][0]:
        if ori_height * ori_weight >= resolution[0] * resolution[1]:
            target_h, target_w = resolution
    cropped_img = isotropic_crop_resize(img, (target_h, target_w))
    return cropped_img, target_h, target_w


def array_to_video(
    image_array: np.ndarray,
    output_path: str,
    fps: int | float = 30,
    resolution: tuple[int, int] | tuple[float, float] | None = None,
    disable_log: bool = False,
    lossless: bool = True,
    output_pix_fmt: str = "yuv420p",
) -> None:
    """Convert an array to a video directly, gif not supported.

    Args:
        image_array (np.ndarray): shape should be (f * h * w * 3).
        output_path (str): output video file path.
        fps (Union[int, float, optional): fps. Defaults to 30.
        resolution (Optional[Union[Tuple[int, int], Tuple[float, float]]],
            optional): (height, width) of the output video.
            Defaults to None.
        disable_log (bool, optional): whether close the ffmepg command info.
            Defaults to False.
        output_pix_fmt (str): output pix_fmt in ffmpeg command.
    Raises:
        FileNotFoundError: check output path.
        TypeError: check input array.

    Returns:
        None.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("Input should be np.ndarray.")
    assert image_array.ndim == 4
    assert image_array.shape[-1] == 3
    if resolution:
        height, width = resolution
        width += width % 2
        height += height % 2
    else:
        image_array = pad_for_libx264(image_array)
        height, width = image_array.shape[1], image_array.shape[2]
    if lossless:
        command = [
            "/usr/bin/ffmpeg",
            "-y",  # (optional) overwrite output file if it exists
            "-f",
            "rawvideo",
            "-s",
            f"{int(width)}x{int(height)}",  # size of one frame
            "-pix_fmt",
            "bgr24",
            "-r",
            f"{fps}",  # frames per second
            "-loglevel",
            "error",
            "-threads",
            "4",
            "-i",
            "-",  # The input comes from a pipe
            "-vcodec",
            "libx264rgb",
            "-crf",
            "0",
            "-an",  # Tells FFMPEG not to expect any audio
            output_path,
        ]
    else:
        output_pix_fmt = output_pix_fmt or "yuv420p"
        command = [
            "/usr/bin/ffmpeg",
            "-y",  # (optional) overwrite output file if it exists
            "-f",
            "rawvideo",
            "-s",
            f"{int(width)}x{int(height)}",  # size of one frame
            "-pix_fmt",
            "bgr24",
            "-r",
            f"{fps}",  # frames per second
            "-loglevel",
            "error",
            "-threads",
            "4",
            "-i",
            "-",  # The input comes from a pipe
            "-vcodec",
            "libx264",
            "-pix_fmt",
            f"{output_pix_fmt}",
            "-an",  # Tells FFMPEG not to expect any audio
            output_path,
        ]

    if output_pix_fmt is not None:
        command += ["-pix_fmt", output_pix_fmt]

    if not disable_log:
        print(f'Running "{" ".join(command)}"')
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stderr is None:
        raise BrokenPipeError("No buffer received.")
    index = 0
    while True:
        if index >= image_array.shape[0]:
            break
        process.stdin.write(image_array[index].tobytes())
        index += 1
    process.stdin.close()
    process.stderr.close()
    process.wait()


def pad_for_libx264(image_array):
    if image_array.ndim == 2 or (image_array.ndim == 3 and image_array.shape[2] == 3):
        hei_index = 0
        wid_index = 1
    elif image_array.ndim == 4 or (image_array.ndim == 3 and image_array.shape[2] != 3):
        hei_index = 1
        wid_index = 2
    else:
        return image_array
    hei_pad = image_array.shape[hei_index] % 2
    wid_pad = image_array.shape[wid_index] % 2
    if hei_pad + wid_pad > 0:
        pad_width = []
        for dim_index in range(image_array.ndim):
            if dim_index == hei_index:
                pad_width.append((0, hei_pad))
            elif dim_index == wid_index:
                pad_width.append((0, wid_pad))
            else:
                pad_width.append((0, 0))
        values = 0
        image_array = np.pad(image_array, pad_width, mode="constant", constant_values=values)
    return image_array


def generate_unique_path(path):
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    index = 1
    new_path = f"{root}-{index}{ext}"
    while os.path.exists(new_path):
        index += 1
        new_path = f"{root}-{index}{ext}"
    return new_path


def save_to_video(gen_lvideo, out_path, target_fps):
    gen_lvideo = rearrange(gen_lvideo, "B C T H W -> B T H W C")
    gen_lvideo = (gen_lvideo[0].cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    gen_lvideo = gen_lvideo[..., ::-1].copy()
    generate_unique_path(out_path)
    array_to_video(gen_lvideo, output_path=out_path, fps=target_fps, lossless=False, output_pix_fmt="yuv444p")


def save_audio(
    audio_array,
    audio_name: str,
    video_name: str,
    sr: int = 16000,
    output_path: Optional[str] = None,
):
    logger.info(f"Saving audio to {audio_name} type: {type(audio_array)}")

    ta.save(
        audio_name,
        torch.tensor(audio_array[None]),
        sample_rate=sr,
    )

    if output_path is None:
        out_video = f"{video_name[:-4]}_with_audio.mp4"
    else:
        out_video = output_path

    parent_dir = os.path.dirname(out_video)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(out_video):
        os.remove(out_video)

    subprocess.call(["/usr/bin/ffmpeg", "-y", "-i", video_name, "-i", audio_name, out_video])

    return out_video


@RUNNER_REGISTER("wan2.1_audio")
class WanAudioRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)

    def init_scheduler(self):
        scheduler = EulerSchedulerTimestepFix(self.config)
        self.model.set_scheduler(scheduler)

    def load_audio_models(self):
        ##音频特征提取器
        self.audio_preprocess = AutoFeatureExtractor.from_pretrained(self.config["model_path"], subfolder="audio_encoder")

        ##音频驱动视频生成adapter
        audio_adapter_path = self.config["model_path"] + "/audio_adapter.safetensors"
        audio_adaper = AudioAdapter.from_transformer(
            self.model,
            audio_feature_dim=1024,
            interval=1,
            time_freq_dim=256,
            projection_transformer_layers=4,
        )
        audio_adapter = rank0_load_state_dict_from_path(audio_adaper, audio_adapter_path, strict=False)

        ##音频特征编码器
        device = self.model.device
        audio_encoder_repo = self.config["model_path"] + "/audio_encoder"
        audio_adapter_pipe = AudioAdapterPipe(audio_adapter, audio_encoder_repo=audio_encoder_repo, dtype=torch.bfloat16, device=device, generator=torch.Generator(device), weight=1.0)

        return audio_adapter_pipe

    def load_transformer(self):
        base_model = WanAudioModel(self.config.model_path, self.config, self.init_device)

        if self.config.get("lora_configs") and self.config.lora_configs:
            assert not self.config.get("dit_quantized", False) or self.config.mm_config.get("weight_auto_quant", False)
            lora_wrapper = WanLoraWrapper(base_model)
            for lora_config in self.config.lora_configs:
                lora_path = lora_config["path"]
                strength = lora_config.get("strength", 1.0)
                lora_name = lora_wrapper.load_lora(lora_path)
                lora_wrapper.apply_lora(lora_name, strength)
                logger.info(f"Loaded LoRA: {lora_name} with strength: {strength}")

        return base_model

    def load_image_encoder(self):
        clip_model_dir = self.config["model_path"] + "/image_encoder"
        image_encoder = WanVideoIPHandler("CLIPModel", repo_or_path=clip_model_dir, require_grad=False, mode="eval", device=self.init_device, dtype=torch.float16)

        return image_encoder

    def run_image_encoder(self, config, vae_model):
        ref_img = Image.open(config.image_path)
        ref_img = (np.array(ref_img).astype(np.float32) - 127.5) / 127.5
        ref_img = torch.from_numpy(ref_img).to(vae_model.device)
        ref_img = rearrange(ref_img, "H W C -> 1 C H W")
        ref_img = ref_img[:, :3]

        # resize and crop image
        cond_frms, tgt_h, tgt_w = adaptive_resize(ref_img)
        config.tgt_h = tgt_h
        config.tgt_w = tgt_w
        clip_encoder_out = self.image_encoder.encode(cond_frms).squeeze(0).to(torch.bfloat16)

        cond_frms = rearrange(cond_frms, "1 C H W -> 1 C 1 H W")
        lat_h, lat_w = tgt_h // 8, tgt_w // 8
        config.lat_h = lat_h
        config.lat_w = lat_w
        vae_encode_out = vae_model.encode(cond_frms.to(torch.float), config)
        if isinstance(vae_encode_out, list):  #
            # list转tensor
            vae_encode_out = torch.stack(vae_encode_out, dim=0).to(torch.bfloat16)

        return vae_encode_out, clip_encoder_out

    def run_input_encoder_internal(self):
        image_encoder_output = None
        if os.path.isfile(self.config.image_path):
            with ProfilingContext("Run Img Encoder"):
                vae_encode_out, clip_encoder_out = self.run_image_encoder(self.config, self.vae_encoder)
                image_encoder_output = {
                    "clip_encoder_out": clip_encoder_out,
                    "vae_encode_out": vae_encode_out,
                }
                logger.info(f"clip_encoder_out:{clip_encoder_out.shape} vae_encode_out:{vae_encode_out.shape}")

        with ProfilingContext("Run Text Encoder"):
            logger.info(f"Prompt: {self.config['prompt']}")
            img = Image.open(self.config["image_path"]).convert("RGB")
            text_encoder_output = self.run_text_encoder(self.config["prompt"], img)

        self.set_target_shape()
        self.inputs = {"text_encoder_output": text_encoder_output, "image_encoder_output": image_encoder_output}

        # del self.image_encoder  # 删除ref的clip模型，只使用一次
        gc.collect()
        torch.cuda.empty_cache()

    def set_target_shape(self):
        ret = {}
        num_channels_latents = 16
        if self.config.task == "i2v":
            self.config.target_shape = (
                num_channels_latents,
                (self.config.target_video_length - 1) // self.config.vae_stride[0] + 1,
                self.config.lat_h,
                self.config.lat_w,
            )
            ret["lat_h"] = self.config.lat_h
            ret["lat_w"] = self.config.lat_w
        else:
            error_msg = "t2v task is not supported in WanAudioRunner"
            assert 1 == 0, error_msg

        ret["target_shape"] = self.config.target_shape
        return ret

    def run(self):
        def load_audio(in_path: str, sr: float = 16000):
            audio_array, ori_sr = ta.load(in_path)
            audio_array = ta.functional.resample(audio_array.mean(0), orig_freq=ori_sr, new_freq=sr)
            return audio_array.numpy()

        def get_audio_range(start_frame: int, end_frame: int, fps: float, audio_sr: float = 16000):
            audio_frame_rate = audio_sr / fps
            return round(start_frame * audio_frame_rate), round((end_frame + 1) * audio_frame_rate)

        def wan_mask_rearrange(mask: torch.Tensor):
            # mask: 1, T, H, W, where 1 means the input mask is one-channel
            if mask.ndim == 3:
                mask = mask[None]
            assert mask.ndim == 4
            _, t, h, w = mask.shape
            assert t == ((t - 1) // 4 * 4 + 1)
            mask_first_frame = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
            mask = torch.concat([mask_first_frame, mask[:, 1:]], dim=1)
            mask = mask.view(mask.shape[1] // 4, 4, h, w)
            return mask.transpose(0, 1)  # 4, T // 4, H, W

        self.inputs["audio_adapter_pipe"] = self.load_audio_models()

        # process audio
        audio_sr = self.config.get("audio_sr", 16000)
        max_num_frames = self.config.get("target_video_length", 81)  # wan2.1一段最多81帧，5秒，16fps
        target_fps = self.config.get("target_fps", 16)  # 音视频同步帧率
        video_duration = self.config.get("video_duration", 5)  # 期望视频输出时长
        audio_array = load_audio(self.config["audio_path"], sr=audio_sr)
        audio_len = int(audio_array.shape[0] / audio_sr * target_fps)
        prev_frame_length = 5
        prev_token_length = (prev_frame_length - 1) // 4 + 1
        max_num_audio_length = int((max_num_frames + 1) / target_fps * audio_sr)

        interval_num = 1
        # expected_frames
        expected_frames = min(max(1, int(float(video_duration) * target_fps)), audio_len)
        res_frame_num = 0
        if expected_frames <= max_num_frames:
            interval_num = 1
        else:
            interval_num = max(int((expected_frames - max_num_frames) / (max_num_frames - prev_frame_length)) + 1, 1)
            res_frame_num = expected_frames - interval_num * (max_num_frames - prev_frame_length)
            if res_frame_num > 5:
                interval_num += 1

        audio_start, audio_end = get_audio_range(0, expected_frames, fps=target_fps, audio_sr=audio_sr)
        audio_array_ori = audio_array[audio_start:audio_end]

        gen_video_list = []
        cut_audio_list = []
        # reference latents

        tgt_h = self.config.tgt_h
        tgt_w = self.config.tgt_w
        device = self.model.scheduler.latents.device
        dtype = torch.bfloat16
        vae_dtype = torch.float

        for idx in range(interval_num):
            self.config.seed = self.config.seed + idx
            torch.manual_seed(self.config.seed)
            logger.info(f"###  manual_seed: {self.config.seed} ####")
            useful_length = -1
            if idx == 0:  # 第一段 Condition padding0
                prev_frames = torch.zeros((1, 3, max_num_frames, tgt_h, tgt_w), device=device)
                prev_latents = self.vae_encoder.encode(prev_frames.to(vae_dtype), self.config)[0].to(dtype)
                prev_len = 0
                audio_start, audio_end = get_audio_range(0, max_num_frames, fps=target_fps, audio_sr=audio_sr)
                audio_array = audio_array_ori[audio_start:audio_end]
                if expected_frames < max_num_frames:
                    useful_length = audio_array.shape[0]
                    audio_array = np.concatenate((audio_array, np.zeros(max_num_audio_length)[: max_num_audio_length - useful_length]), axis=0)
                audio_input_feat = self.audio_preprocess(audio_array, sampling_rate=audio_sr, return_tensors="pt").input_values.squeeze(0)

            elif res_frame_num > 5 and idx == interval_num - 1:  # 最后一段可能不够81帧
                prev_frames = torch.zeros((1, 3, max_num_frames, tgt_h, tgt_w), device=device)
                last_frames = gen_video_list[-1][:, :, -prev_frame_length:].clone().to(device)

                last_frames = last_frames.cpu().detach().numpy()
                last_frames = add_noise_to_frames(last_frames)
                last_frames = add_mask_to_frames(last_frames, mask_rate=0.1)  # mask 0.10
                last_frames = torch.from_numpy(last_frames).to(dtype=dtype, device=device)

                prev_frames[:, :, :prev_frame_length] = last_frames
                prev_latents = self.vae_encoder.encode(prev_frames.to(vae_dtype), self.config)[0].to(dtype)
                prev_len = prev_token_length
                audio_start, audio_end = get_audio_range(idx * max_num_frames - idx * prev_frame_length, expected_frames, fps=target_fps, audio_sr=audio_sr)
                audio_array = audio_array_ori[audio_start:audio_end]
                useful_length = audio_array.shape[0]
                audio_array = np.concatenate((audio_array, np.zeros(max_num_audio_length)[: max_num_audio_length - useful_length]), axis=0)
                audio_input_feat = self.audio_preprocess(audio_array, sampling_rate=audio_sr, return_tensors="pt").input_values.squeeze(0)

            else:  # 中间段满81帧带pre_latens
                prev_frames = torch.zeros((1, 3, max_num_frames, tgt_h, tgt_w), device=device)
                last_frames = gen_video_list[-1][:, :, -prev_frame_length:].clone().to(device)

                last_frames = last_frames.cpu().detach().numpy()
                last_frames = add_noise_to_frames(last_frames)
                last_frames = add_mask_to_frames(last_frames, mask_rate=0.1)  # mask 0.10
                last_frames = torch.from_numpy(last_frames).to(dtype=dtype, device=device)

                prev_frames[:, :, :prev_frame_length] = last_frames
                prev_latents = self.vae_encoder.encode(prev_frames.to(vae_dtype), self.config)[0].to(dtype)
                prev_len = prev_token_length
                audio_start, audio_end = get_audio_range(idx * max_num_frames - idx * prev_frame_length, (idx + 1) * max_num_frames - idx * prev_frame_length, fps=target_fps, audio_sr=audio_sr)
                audio_array = audio_array_ori[audio_start:audio_end]
                audio_input_feat = self.audio_preprocess(audio_array, sampling_rate=audio_sr, return_tensors="pt").input_values.squeeze(0)

            self.inputs["audio_encoder_output"] = audio_input_feat.to(device)

            if idx != 0:
                self.model.scheduler.reset()

            if prev_latents is not None:
                ltnt_channel, nframe, height, width = self.model.scheduler.latents.shape
                # bs = 1
                frames_n = (nframe - 1) * 4 + 1
                prev_frame_len = max((prev_len - 1) * 4 + 1, 0)
                prev_mask = torch.ones((1, frames_n, height, width), device=device, dtype=dtype)
                prev_mask[:, prev_frame_len:] = 0
                prev_mask = wan_mask_rearrange(prev_mask).unsqueeze(0)
                previmg_encoder_output = {
                    "prev_latents": prev_latents,
                    "prev_mask": prev_mask,
                }
                self.inputs["previmg_encoder_output"] = previmg_encoder_output

            for step_index in range(self.model.scheduler.infer_steps):
                logger.info(f"==> step_index: {step_index} / {self.model.scheduler.infer_steps}")

                with ProfilingContext4Debug("step_pre"):
                    self.model.scheduler.step_pre(step_index=step_index)

                with ProfilingContext4Debug("infer"):
                    self.model.infer(self.inputs)

                with ProfilingContext4Debug("step_post"):
                    self.model.scheduler.step_post()

            latents = self.model.scheduler.latents
            generator = self.model.scheduler.generator
            gen_video = self.vae_decoder.decode(latents, generator=generator, config=self.config)
            gen_video = torch.clamp(gen_video, -1, 1)
            start_frame = 0 if idx == 0 else prev_frame_length
            start_audio_frame = 0 if idx == 0 else int((prev_frame_length + 1) * audio_sr / target_fps)

            if res_frame_num > 5 and idx == interval_num - 1:
                gen_video_list.append(gen_video[:, :, start_frame:res_frame_num].cpu())
                cut_audio_list.append(audio_array[start_audio_frame:useful_length])
            elif expected_frames < max_num_frames and useful_length != -1:
                gen_video_list.append(gen_video[:, :, start_frame:expected_frames].cpu())
                cut_audio_list.append(audio_array[start_audio_frame:useful_length])
            else:
                gen_video_list.append(gen_video[:, :, start_frame:].cpu())
                cut_audio_list.append(audio_array[start_audio_frame:])

        gen_lvideo = torch.cat(gen_video_list, dim=2).float()
        merge_audio = np.concatenate(cut_audio_list, axis=0).astype(np.float32)
        out_path = os.path.join("./", "video_merge.mp4")
        audio_file = os.path.join("./", "audio_merge.wav")
        save_to_video(gen_lvideo, out_path, target_fps)
        save_audio(merge_audio, audio_file, out_path, output_path=self.config.get("save_video_path", None))
        os.remove(out_path)
        os.remove(audio_file)

    def run_pipeline(self):
        if self.config["use_prompt_enhancer"]:
            self.config["prompt_enhanced"] = self.post_prompt_enhancer()

        self.run_input_encoder_internal()
        self.set_target_shape()

        self.init_scheduler()
        self.model.scheduler.prepare(self.inputs["image_encoder_output"])
        self.run()
        self.end_run()

        gc.collect()
        torch.cuda.empty_cache()
