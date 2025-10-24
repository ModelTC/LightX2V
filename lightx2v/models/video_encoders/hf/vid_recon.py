import argparse

import cv2
import torch
from loguru import logger

from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import Wan2_2_VAE
from lightx2v.models.video_encoders.hf.wan.vae_tiny import Wan2_2_VAE_tiny, WanVAE_tiny


class VideoTensorReader:
    def __init__(self, video_file_path):
        self.cap = cv2.VideoCapture(video_file_path)
        assert self.cap.isOpened(), f"Could not load {video_file_path}"
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration  # End of video or error
        return torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)  # BGR HWC -> RGB CHW


class VideoTensorWriter:
    def __init__(self, video_file_path, width_height, fps=30):
        self.writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, width_height)
        assert self.writer.isOpened(), f"Could not create writer for {video_file_path}"

    def write(self, frame_tensor):
        assert frame_tensor.ndim == 3 and frame_tensor.shape[0] == 3, f"{frame_tensor.shape}??"
        self.writer.write(cv2.cvtColor(frame_tensor.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))  # RGB CHW -> BGR HWC

    def __del__(self):
        if hasattr(self, "writer"):
            self.writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode and decode videos using the TaeHV model for reconstruction")
    parser.add_argument("video_paths", nargs="+", help="Paths to input video files (multiple allowed)")
    parser.add_argument("--checkpoint", "-c", help=f"Path to the model checkpoint file")
    parser.add_argument("--device", "-d", default="cuda", help=f'Computing device (e.g., "cuda", "mps", "cpu"; default: auto-detect available device)')
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="Data type for model computation (default: bfloat16)")
    parser.add_argument("--model_type", choices=["taew2_1", "taew2_2", "vaew2_1", "vaew2_2"], required=True, help="Type of the model to use (choices: taew2_1, taew2_2)")
    parser.add_argument("--use_lightvae", default=False, action="store_true")

    args = parser.parse_args()
    if args.use_lightvae:
        assert args.model_type in ["vaew2_1"]

    if args.device:
        dev = torch.device(args.device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    model_map = {"taew2_1": WanVAE_tiny, "taew2_2": Wan2_2_VAE_tiny, "vaew2_1": WanVAE, "vaew2_2": Wan2_2_VAE}

    dtype = dtype_map[args.dtype]

    model_args = {"vae_path": args.checkpoint, "dtype": dtype, "device": dev}
    if args.model_type in "vaew2_1":
        model_args.update({"use_lightvae": args.use_lightvae})

    model = model_map[args.model_type](**model_args)
    if args.model_type.startswith("tae"):
        model = model_map[args.model_type](**model_args).to(dev)

    # Process each input video
    for idx, video_path in enumerate(args.video_paths):
        logger.info(f"Processing video {video_path}...")
        # Read video
        video_in = VideoTensorReader(video_path)
        video = torch.stack(list(video_in), 0)[None]  # Add batch dimension
        vid_dev = video.to(dev, dtype).div_(255.0)  # Normalize to [0,1]
        # Encode
        vid_enc = model.encode_video(vid_dev)
        if isinstance(vid_enc, tuple):
            vid_enc = vid_enc[0]
        # Decode
        vid_dec = model.decode_video(vid_enc)
        # Save reconstructed video
        video_out_path = f"{video_path}.reconstructed_{idx}.mp4"
        frame_size = (vid_dec.shape[-1], vid_dec.shape[-2])
        fps = int(round(video_in.fps))
        video_out = VideoTensorWriter(video_out_path, frame_size, fps)
        for frame in vid_dec.clamp_(0, 1).mul_(255).round_().byte().cpu()[0]:
            video_out.write(frame)
        logger.info(f"  Reconstructed video saved to {video_out_path}")
