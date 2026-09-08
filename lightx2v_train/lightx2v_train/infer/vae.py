"""Video reconstruction with a distilled VAE encoder or decoder."""

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

from lightx2v_train.runtime.distributed import barrier, is_main_process
from lightx2v_train.utils.registry import INFERENCER_REGISTER
from lightx2v_train.utils.video import save_mp4, video_psnr, video_ssim

from .base import BaseInferencer


def _metadata_value(value):
    if torch.is_tensor(value):
        return value.item()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _pil_frames(video):
    for frame_index in range(video.shape[1]):
        frame = video[:, frame_index].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        yield Image.fromarray(np.round(frame * 255.0).astype(np.uint8))


def _chunked_video_metric(metric, prediction, target, frame_batch_size):
    weighted_sum = 0.0
    num_frames = prediction.shape[1]
    for start in range(0, num_frames, frame_batch_size):
        stop = min(start + frame_batch_size, num_frames)
        weighted_sum += metric(prediction[:, start:stop], target[:, start:stop]) * (stop - start)
    return weighted_sum / num_frames


def _save_lossless_comparisons(target, reconstruction, output_dir, stem, frame_count):
    indices = torch.linspace(0, target.shape[1] - 1, min(frame_count, target.shape[1])).round().long().unique()
    frame_dir = output_dir / f"{stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    comparison = torch.cat((target, reconstruction), dim=-1)
    for frame_index in indices.tolist():
        frame = next(_pil_frames(comparison[:, frame_index : frame_index + 1]))
        frame.save(frame_dir / f"frame_{frame_index:04d}_source_student.png")


@INFERENCER_REGISTER("minimax_h3_vae_reconstruction")
@INFERENCER_REGISTER("wan21_vae_reconstruction")
class MiniMaxH3VAEReconstructionInferencer(BaseInferencer):
    scheduler_cls = None

    @torch.no_grad()
    def infer(self):
        if self.output_infer_dir is None:
            raise ValueError("inference.output_dir is required for VAE reconstruction.")

        output_dir = Path(self.output_infer_dir)
        fps = int(self.infer_config.get("fps", 24))
        metric_frame_batch_size = int(self.infer_config.get("metric_frame_batch_size", 4))
        save_comparison = bool(self.infer_config.get("save_comparison", False))
        save_png_frame_count = int(self.infer_config.get("save_png_frame_count", 0))
        if metric_frame_batch_size < 1:
            raise ValueError("inference.metric_frame_batch_size must be positive.")
        max_samples = self.infer_config.get("max_samples")
        metrics = []
        self.model.set_denoiser_eval()

        for index, sample in enumerate(self.dataloader_eval):
            if max_samples is not None and index >= int(max_samples):
                break
            video = sample["inputs"]["video"].to(self.model.device, dtype=torch.float32)
            normalized_latents = sample["inputs"].get("latents")
            reconstruction = (
                self.model.reconstruct(video)
                if normalized_latents is None
                else self.model.decode_latents(normalized_latents)
            )
            source_frames = int(_metadata_value(sample["meta"]["source_num_frames"]))
            target = video[0, :, :source_frames]
            reconstructed = reconstruction[0, :, :source_frames]

            psnr = _chunked_video_metric(video_psnr, reconstructed, target, metric_frame_batch_size)
            ssim = _chunked_video_metric(video_ssim, reconstructed, target, metric_frame_batch_size)
            metrics.append((psnr, ssim))

            if is_main_process():
                source_path = Path(str(_metadata_value(sample["meta"]["video_path"])))
                output_path = output_dir / f"{index:05d}_{source_path.stem}_student.mp4"
                save_mp4(_pil_frames(reconstructed), str(output_path), fps=fps)
                comparison_path = None
                if save_comparison:
                    comparison_path = output_dir / f"{index:05d}_{source_path.stem}_source_student.mp4"
                    save_mp4(_pil_frames(torch.cat((target, reconstructed), dim=-1)), str(comparison_path), fps=fps)
                if save_png_frame_count:
                    _save_lossless_comparisons(
                        target,
                        reconstructed,
                        output_dir,
                        f"{index:05d}_{source_path.stem}",
                        save_png_frame_count,
                    )
                logger.info(
                    "[vae-infer] sample={}/{} frames={} psnr={:.4f} ssim={:.6f} path={} comparison={}",
                    index + 1,
                    len(self.dataloader_eval),
                    source_frames,
                    psnr,
                    ssim,
                    output_path,
                    comparison_path,
                )

        barrier()
        if is_main_process() and metrics:
            mean_psnr = sum(value[0] for value in metrics) / len(metrics)
            mean_ssim = sum(value[1] for value in metrics) / len(metrics)
            logger.info("[vae-infer] finished samples={} psnr={:.4f} ssim={:.6f}", len(metrics), mean_psnr, mean_ssim)
        return metrics
