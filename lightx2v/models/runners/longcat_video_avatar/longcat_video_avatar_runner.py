import os
from contextlib import closing

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageOps
from diffusers.image_processor import VaeImageProcessor
from loguru import logger

from lightx2v.models.input_encoders.hf.wan.t5.model import T5EncoderModel
from lightx2v.models.networks.longcat_video_avatar.model import LongCatVideoAvatarModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.longcat_video_avatar.buckets import select_image_size
from lightx2v.models.schedulers.longcat_video_avatar.scheduler import LongCatVideoAvatarScheduler
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE, convert_to_channels_last_3d
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import GET_DTYPE, GET_RECORDER_MODE
from lightx2v.utils.profiler import ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.streaming_video import StreamingVideoWriter, video_chunk_to_uint8
from lightx2v.utils.utils import wan_vae_to_comfy
from lightx2v_platform.base.global_var import AI_DEVICE

OVERLAP_FRAMES = 13


@RUNNER_REGISTER("longcat_video_avatar")
class LongCatVideoAvatarRunner(DefaultRunner):
    def init_scheduler(self):
        self.scheduler = LongCatVideoAvatarScheduler(self.config)

    def load_model(self):
        if AI_DEVICE != "cuda" or GET_DTYPE() != torch.bfloat16:
            raise ValueError("LongCat-Video-Avatar requires a CUDA device with BF16 inference")
        config = self.config
        world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", 1))
        if world_size != (config.get("parallel") or {}).get("seq_p_size", 1):
            raise ValueError("Launch one process per device configured for LongCat sequence parallelism")
        self.model = LongCatVideoAvatarModel(config, self.init_device)
        self.spatial_alignment = tuple(stride * parts for stride, parts in zip(self._spatial_token_size(), self.model.spatial_split))
        self.vae = WanVAE(
            vae_path=config["vae_path"],
            dtype=torch.bfloat16,
            device=self.init_device,
        )
        if config.get("vae_decoder_channels_last_3d", False):
            convert_to_channels_last_3d(self.vae.model.decoder)
        if config.get("vae_use_compile", False):
            self.vae.compile_decoder()
        # The release normalizes the reference in FP32, then decodes in BF16.
        mean, std = torch.tensor([config["vae_latents_mean"], config["vae_latents_std"]], dtype=torch.float32, device=self.init_device)
        self.vae.scale = (mean.bfloat16(), std.bfloat16().reciprocal())

        self.text_tp_group = self.model.group if config.get("text_encoder_tensor_parallel") else None
        if self.text_tp_group is not None or self.model.rank == 0:
            self.text_encoder = T5EncoderModel(
                text_len=config["model_max_length"],
                device=self.init_device,
                checkpoint_path=config["t5_original_ckpt"],
                tokenizer_path=config["t5_tokenizer_path"],
                tp_group=self.text_tp_group,
            )
        if self.model.rank == 0:
            from lightx2v.models.input_encoders.hf.longcat_video_avatar.audio import LongCatWhisperEncoder

            self.image_processor = VaeImageProcessor(vae_scale_factor=config["vae_scale_factor"])
            self.reference_scale = (mean, std.reciprocal())
            vocal_path = config["vocal_separator_path"] if config["audio_vocal_separation"] else None
            self.audio_encoder = LongCatWhisperEncoder(config["audio_encoder_path"], self.init_device, vocal_checkpoint=vocal_path)

    def _spatial_token_size(self):
        _, patch_h, patch_w = self.config["patch_size"]
        scale = self.config["vae_scale_factor"]
        return scale * patch_h, scale * patch_w

    def create_input_info(self, request_data):
        info = super().create_input_info(request_data)
        if isinstance(info.audio_path, str):
            info.audio_path = [info.audio_path]
        if not isinstance(info.audio_path, list) or len(info.audio_path) not in (1, 2) or not all(isinstance(path, str) and path.strip() for path in info.audio_path):
            raise ValueError("audio_path requires one or two audio paths")
        if info.audio_type not in ("para", "add"):
            raise ValueError("audio_type must be para or add")
        if info.audio_type == "add" and len(info.audio_path) != 2:
            raise ValueError("audio_type=add requires two audio paths")
        if len(info.audio_path) == 2 and not info.image_path:
            raise ValueError("Two-speaker generation requires a reference image with both speakers")
        if info.bbox is not None:
            if len(info.audio_path) != 2:
                raise ValueError("bbox requires two audio paths and a reference image")
            if not isinstance(info.bbox, dict) or info.bbox.keys() - {"person1", "person2", "others"}:
                raise ValueError("bbox accepts person1, person2 and others")
            if ("person1" in info.bbox) != ("person2" in info.bbox):
                raise ValueError("bbox.person1 and bbox.person2 must be supplied together")
            for name, coords in info.bbox.items():
                if not isinstance(coords, list) or any(type(value) is not int for value in coords):
                    raise ValueError(f"bbox.{name} requires a list of integer pixel coordinates")
                if name == "others" and len(coords) % 4:
                    raise ValueError("bbox.others requires groups of four pixel coordinates")
                if name != "others" and len(coords) != 4:
                    raise ValueError(f"bbox.{name} requires four pixel coordinates")
        if info.resolution not in ("480p", "720p"):
            raise ValueError("resolution must be 480p or 720p")
        if "resolution" in request_data and "size" not in request_data:
            # A requested preset supersedes a deployment's default pixel size.
            info.size = []
        if type(info.num_frames) is not int or info.num_frames < 5 or (info.num_frames - 1) % self.config["vae_scale"]:
            raise ValueError("num_frames must be an integer of the form 4k+1 with at least 5 frames")
        if type(info.num_segments) is not int or info.num_segments < 1:
            raise ValueError("num_segments must be a positive integer")
        if info.num_segments > 1 and info.num_frames <= OVERLAP_FRAMES:
            raise ValueError(f"num_frames must exceed {OVERLAP_FRAMES} when num_segments is greater than one")
        if not isinstance(info.size, (list, tuple)) or len(info.size) not in (0, 2):
            raise ValueError("size must be [height, width] or empty for automatic sizing")
        alignment = self.spatial_alignment
        if not info.image_path and not info.size:
            height, width = (768, 1280) if info.resolution == "720p" else (480, 832)
            if height % alignment[0] or width % alignment[1]:
                height, width = select_image_size(info.resolution, max(alignment), height / width)
            info.size = [height, width]
        if any(type(side) is not int or side <= 0 or side % stride for side, stride in zip(info.size, alignment)):
            raise ValueError(f"size must contain positive integer [height, width] divisible by {alignment}")
        return info

    def _run_input_encoder_local_s2v(self):
        if self.model.rank != 0:
            return {}
        info = self.input_info
        fps = self.config["fps"]
        total_frames = info.num_frames + (info.num_segments - 1) * (info.num_frames - OVERLAP_FRAMES)
        silent_background = bool(info.bbox and info.bbox.get("others"))
        audio, waveform, sample_rate = self.audio_encoder.prepare_tracks(info.audio_path, total_frames, fps, info.audio_type, silent_background)
        frame_size = info.size
        inputs = {"audio": audio}
        if self.text_tp_group is None:
            inputs["text"] = self.text_encoder.infer([info.prompt])[0]
        if info.image_path:
            with Image.open(info.image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                ratio = image.height / image.width
                frame_size = frame_size or select_image_size(info.resolution, max(self.spatial_alignment), ratio)
                height, width = frame_size
                if len(info.audio_path) == 2:
                    inputs["speaker_masks"] = self._speaker_masks(image, height, width, info.bbox or {})
                pixels = self.image_processor.preprocess(image, height=height, width=width, resize_mode="crop")
            # Contiguous NCTHW selects the same BF16 cuDNN path as the reference VAE.
            pixels = pixels.unsqueeze(2).to(self.init_device, torch.bfloat16, memory_format=torch.contiguous_format)
            inputs["reference"] = self.vae.model.encode(pixels, self.reference_scale)
        if info.save_result_path and not info.return_result_tensor:
            self.video_writer = StreamingVideoWriter(info.save_result_path, fps)
            self.video_writer.set_audio(torch.from_numpy(waveform), sample_rate, num_frames=total_frames)
            self.video_writer.start(*frame_size)
        return inputs

    def _speaker_masks(self, image, height, width, bbox):
        if "person1" in bbox:
            boxes = [bbox["person1"], bbox["person2"]]
        else:
            # The release assigns the inner 80% of each image half to its speaker.
            half = image.width // 2
            top, bottom = int(image.height * 0.1), int(image.height * 0.9)
            left, right = int(half * 0.1), int(half * 0.9)
            boxes = [(top, left, bottom, right), (top, half + left, bottom, half + right)]
        others = bbox.get("others", [])
        boxes.extend(others[i : i + 4] for i in range(0, len(others), 4))
        people = torch.zeros((len(boxes), image.height, image.width), device=self.init_device)
        for mask, (y1, x1, y2, x2) in zip(people, boxes):
            if not (0 <= y1 < y2 <= image.height and 0 <= x1 < x2 <= image.width):
                raise ValueError(f"Invalid bbox {[y1, x1, y2, x2]} for reference image of height {image.height}, width {image.width}")
            mask[y1:y2, x1:x2] = 1
        background = 1 - torch.maximum(people[0], people[1])
        masks = torch.cat((people[:2], background[None], people[2:]))[None]
        resized_h = max(height, image.height * width // image.width)
        resized_w = max(width, image.width * height // image.height)
        masks = F.interpolate(masks, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
        y, x = (resized_h - height) // 2, (resized_w - width) // 2
        masks = masks[:, :, y : y + height, x : x + width]
        token_h, token_w = self._spatial_token_size()
        masks = F.interpolate(masks, size=(height // token_h, width // token_w), mode="nearest")[0].flatten(1) > 0
        if not masks.any(dim=1).all():
            raise ValueError("Each person and the background must be visible in the selected crop and output size")
        return masks.to(torch.bfloat16)

    def _run_and_broadcast(self, operation, *args):
        model = self.model
        if model.group is None:
            return operation(*args)

        metadata = [None]
        try:
            tensors = operation(*args)
            if model.rank == 0:
                tensors = {key: tensor.contiguous() for key, tensor in tensors.items()}
                metadata[0] = {key: (tensor.shape, tensor.dtype) for key, tensor in tensors.items()}
        except Exception as exc:
            if model.rank == 0:
                dist.broadcast_object_list([str(exc)], src=0, group=model.group)
            raise
        dist.broadcast_object_list(metadata, src=0, group=model.group)
        if model.rank != 0:
            if isinstance(metadata[0], str):
                raise RuntimeError(f"LongCat primary rank failed: {metadata[0]}")
            tensors = {key: torch.empty(shape, dtype=dtype, device=self.init_device) for key, (shape, dtype) in metadata[0].items()}
        for tensor in tensors.values():
            dist.broadcast(tensor, src=0, group=model.group)
        return tensors

    def _decode_chunks(self, latents):
        """Decode on every rank and yield the blended frames on rank 0."""
        rows, cols = self.model.spatial_split
        if (rows, cols) == (1, 1):
            yield from self.vae.decode_stream(latents)
            return

        vae = self.vae.model
        scale = vae.spatial_compression_ratio
        halo = (vae.tile_sample_min_height - vae.tile_sample_stride_height) // scale
        height, width = latents.shape[-2:]

        def axis_tile(length, parts, index):
            core = length // parts
            start = max(0, index * core - halo)
            stop = min(length, (index + 1) * core + halo)
            weights = torch.ones((stop - start) * scale, dtype=torch.float32, device=latents.device)
            blend = min(2 * halo * scale, weights.numel())
            if index > 0:
                weights[:blend] = torch.linspace(0, 1, blend, device=latents.device)
            if index + 1 < parts:
                weights[-blend:] *= torch.linspace(1, 0, blend, device=latents.device)
            return slice(start, stop), slice(start * scale, stop * scale), weights

        rank, group = self.model.rank, self.model.group
        row, col = divmod(rank, cols)
        latent_y, pixel_y, weight_y = axis_tile(height, rows, row)
        latent_x, pixel_x, weight_x = axis_tile(width, cols, col)
        weight = weight_y[:, None] * weight_x
        pixel_shape = (height * scale, width * scale)
        weight_sum = weight.new_zeros(pixel_shape)
        weight_sum[pixel_y, pixel_x] = weight
        dist.reduce(weight_sum, dst=0, group=group)
        if rank == 0:
            weight_sum.clamp_min_(1e-6)

        tile = latents[..., latent_y, latent_x].contiguous()
        with closing(self.vae.decode_stream(tile)) as chunks:
            for chunk in chunks:
                # WanVAE clamps each tile before the FP32 overlap blend.
                frames = weight.new_zeros((*chunk.shape[:-2], *pixel_shape))
                frames[..., pixel_y, pixel_x] = chunk.float() * weight
                dist.reduce(frames, dst=0, group=group)
                if rank == 0:
                    yield frames.div_(weight_sum).to(chunk.dtype)

    def _decode_segment(self, latents, segment_idx, has_next_segment, result_chunks):
        info = self.input_info
        skip = OVERLAP_FRAMES if segment_idx else 0
        overlap_start = info.num_frames - OVERLAP_FRAMES
        tail = []
        position = 0
        chunks = self._decode_chunks(latents[0].to(torch.bfloat16))
        try:
            for chunk in chunks:
                end = position + chunk.shape[2]
                if has_next_segment and end > overlap_start:
                    overlap = chunk[:, :, max(0, overlap_start - position) :]
                    # The release feeds truncated uint8 RGB back into the VAE after BF16 denormalization.
                    rgb = (overlap * 0.5 + 0.5).float().mul_(255).to(torch.uint8)
                    tail.append(rgb)
                if end > skip:
                    frames = chunk[:, :, max(0, skip - position) :]
                    if info.return_result_tensor:
                        result_chunks.append(wan_vae_to_comfy(frames.float()).cpu())
                    elif self.video_writer:
                        self.video_writer.submit(video_chunk_to_uint8(frames, device=frames.device))
                position = end
        except Exception:
            # Finish the reductions peers are waiting on before propagating a writer error.
            if self.model.group is not None:
                for _ in chunks:
                    pass
            raise
        finally:
            chunks.close()
        if has_next_segment and self.model.rank == 0:
            # Center before dividing to match the PIL/NumPy normalization at every uint8 value.
            pixels = torch.cat(tail, dim=2).float(memory_format=torch.channels_last_3d).mul_(2).sub_(255).div_(255).bfloat16()
            return {"reference": self.vae.model.encode(pixels, self.vae.scale)}
        if self.video_writer:
            self.video_writer.close()
            logger.info(f"Saved LongCat-Video-Avatar result to {info.save_result_path}")
        return {}

    @torch.no_grad()
    @ProfilingContext4DebugL1(
        "RUN pipeline",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_worker_request_duration,
        metrics_labels=["LongCatVideoAvatarRunner"],
    )
    def run_pipeline(self, input_info):
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_count.inc()
        self.input_info = input_info
        self.video_writer = None
        self.video_segment_num = input_info.num_segments
        result_chunks = []
        try:
            self.check_stop()
            encoded = self._run_and_broadcast(self.run_input_encoder)
            text = self.text_encoder.infer([input_info.prompt])[0] if self.text_tp_group is not None else encoded.pop("text")
            projected_text = self.model.pre_infer.project_text(text)
            audio = encoded["audio"]
            radius = self.config["audio_window"] // 2
            indices = torch.arange(input_info.num_frames, device=self.init_device)[:, None] + torch.arange(-radius, radius + 1, device=self.init_device)
            generator = torch.Generator(device=self.init_device).manual_seed(input_info.seed)
            reference = encoded.pop("reference", None)
            anchor = None
            for index in range(input_info.num_segments):
                self.check_stop()
                self.scheduler.prepare(input_info, reference, generator, anchor)
                start = index * (input_info.num_frames - OVERLAP_FRAMES)
                segment_audio = audio[:, (indices + start).clamp(0, audio.shape[1] - 1)]
                # All continuation segments in this request share the same anchored grid.
                self.inputs = self.model.prepare_condition(projected_text, segment_audio, encoded.get("speaker_masks"), rope=self.inputs["rope"] if index > 1 else None)
                latents = self.run_segment(index)
                self.check_stop()
                has_next_segment = index + 1 < input_info.num_segments
                if index:
                    latents = latents[:, :, 1:]
                elif has_next_segment:
                    anchor = latents[:, :, :1].clone()
                if has_next_segment or input_info.save_result_path or input_info.return_result_tensor:
                    outputs = self._run_and_broadcast(self._decode_segment, latents, index, has_next_segment, result_chunks)
                    reference = outputs.get("reference")
            video = torch.cat(result_chunks) if result_chunks else None
        except BaseException:
            if self.video_writer:
                self.video_writer.abort()
            raise
        finally:
            self.video_writer = None
            self.end_run()
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        return {"video": video}
