from lightx2v_train.utils.registry import build_data, build_sample_processor


def __getattr__(name):
    if name == "build_image_dataset":
        from .image_dataset import build_image_dataset

        return build_image_dataset
    if name == "build_training_cache_dataset":
        from .training_cache_dataset import build_training_cache_dataset

        return build_training_cache_dataset
    if name in {"build_prompt_dataset", "build_video_dataset"}:
        from .video_dataset import build_prompt_dataset, build_video_dataset

        return {
            "build_prompt_dataset": build_prompt_dataset,
            "build_video_dataset": build_video_dataset,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_data",
    "build_sample_processor",
    "build_image_dataset",
    "build_prompt_dataset",
    "build_training_cache_dataset",
    "build_video_dataset",
]
