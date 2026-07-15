from lightx2v_train.utils.registry import build_data


def __getattr__(name):
    if name == "build_image_dataset":
        from .image_dataset import build_image_dataset

        return build_image_dataset
    if name in {"build_latent_dataset", "build_prompt_dataset", "build_video_dataset"}:
        from .video_dataset import build_latent_dataset, build_prompt_dataset, build_video_dataset

        return {
            "build_latent_dataset": build_latent_dataset,
            "build_prompt_dataset": build_prompt_dataset,
            "build_video_dataset": build_video_dataset,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_data",
    "build_image_dataset",
    "build_latent_dataset",
    "build_prompt_dataset",
    "build_video_dataset",
]
