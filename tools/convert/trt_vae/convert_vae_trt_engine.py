from pathlib import Path
import os
import argparse

import torch
from loguru import logger

from lightx2v.models.video_encoders.hf.autoencoder_kl_causal_3d.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from lightx2v.models.video_encoders.trt.autoencoder_kl_causal_3d.trt_vae_infer import HyVaeTrtModelInfer


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", help="", type=str)
    args.add_argument("--dtype", default=torch.float16)
    args.add_argument("--device", default="cuda", type=str)
    return args.parse_args()


def convert_vae_trt_engine(args):
    vae_path = os.path.join(args.model_path, "hunyuan-video-t2v-720p/vae")
    assert Path(vae_path).exists(), f"{vae_path} not exists."
    config = AutoencoderKLCausal3D.load_config(vae_path)
    model = AutoencoderKLCausal3D.from_config(config)
    assert Path(os.path.join(vae_path, "pytorch_model.pt")).exists(), f"{os.path.join(vae_path, 'pytorch_model.pt')} not exists."
    ckpt = torch.load(os.path.join(vae_path, "pytorch_model.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model = model.to(dtype=args.dtype, device=args.device)
    onnx_path = HyVaeTrtModelInfer.export_to_onnx(model.decoder, vae_path)
    del model
    torch.cuda.empty_cache()
    engine_path = onnx_path.replace(".onnx", ".engine")
    HyVaeTrtModelInfer.convert_to_trt_engine(onnx_path, engine_path)
    logger.info(f"ONNX: {onnx_path}")
    logger.info(f"TRT Engine: {engine_path}")
    return


def main():
    args = parse_args()
    convert_vae_trt_engine(args)


if __name__ == "__main__":
    main()
