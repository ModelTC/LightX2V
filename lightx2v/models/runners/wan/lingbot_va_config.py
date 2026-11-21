from loguru import logger

from lightx2v.models.runners.wan.config import load_config as load_wan_config


def load_config(deployment_config):
    config = load_wan_config(deployment_config)
    if "num_frames" not in config:
        ar_config = config.get("ar_config", {})
        missing_keys = [key for key in ("num_frame_per_chunk", "num_chunks") if key not in ar_config]
        if missing_keys:
            raise ValueError(f"LingBot-VA requires ar_config.{', ar_config.'.join(missing_keys)} to derive num_frames.")
        latent_frames = int(ar_config["num_frame_per_chunk"]) * int(ar_config["num_chunks"])
        temporal_stride = int(config["vae_stride"][0])
        if latent_frames <= 0 or temporal_stride <= 0:
            raise ValueError(f"LingBot-VA requires positive latent frame count and VAE temporal stride, got latent_frames={latent_frames}, temporal_stride={temporal_stride}.")
        config["num_frames"] = (latent_frames - 1) * temporal_stride + 1
        logger.info(f"Auto-set LingBot-VA num_frames={config['num_frames']} from {latent_frames} latent frames and temporal stride {temporal_stride}.")
    return config
