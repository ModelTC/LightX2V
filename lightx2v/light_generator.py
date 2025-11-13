import torch
import torch.distributed as dist
from loguru import logger
from typing import Union, Optional, Dict, Any
from pathlib import Path


from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_animate_runner import WanAnimateRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_audio_runner import Wan22AudioRunner, WanAudioRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner, WanRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_sf_runner import WanSFRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_vace_runner import WanVaceRunner  # noqa: F401


from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import set_config, set_parallel_config, print_config
from lightx2v.utils.input_info import set_input_info
from lightx2v.utils.utils import seed_all, get_configs_dir


class LightGenerator:
    """
    A clean interface for LightX2V video generation.

    This class provides a simple API that separates model initialization from inference,
    following the same configuration pattern as lightx2v/infer.py.

    Example:
        >>> generator = LightGenerator(
        ...     model_path="/path/to/model",
        ...     model_cls="wan2.1", # wan2.1, wan2.1_distill, wan2.2_moe, wan2.2_moe_distill, ...
        ...     task="i2v" # t2v, i2v, flf2v, ...
        ... )
        >>> video_path = generator.generate(
        ...     prompt="Two anthropomorphic cats...",
        ...     image_path="input.jpg",
        ...     save_result_path="output_video.mp4"
        ... )
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        model_cls: str = "wan2.1",
        task: str = "t2v",
        config_json: Optional[Union[str, Path, Dict[str, Any]]] = None,
    ):
        """
        Initialize the LightGenerator.

        Args:
            model_path: Path to the pre-trained model
            model_cls: Model class name (from lightx2v/infer.py choices)
            task: Task type ("t2v", "i2v", "t2i", "i2i", "flf2v", "vace", "animate", "s2v")
            config_json: Path to config file or config dict. Follows LightX2V config pattern.
        """

        # Auto-select default config based on model_cls and task if not provided
        if config_json is None:
            config_json = self._get_default_config_path(model_cls, task)

        # Create args-like object for set_config compatibility
        class Args:
            def __init__(self):
                self.model_path = str(model_path)
                self.model_cls = model_cls
                self.task = task
                self.config_json = str(config_json)

        args = Args()

        # Set config using the same logic as infer.py
        config = set_config(args)

        if config["parallel"]:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(dist.get_rank())
            set_parallel_config(config)

        print_config(config)

        # Initialize runner
        logger.info(f"Initializing {model_cls} runner for {task} task...")
        logger.info(f"Model path: {model_path}")

        self.runner = self._init_runner(config)
        self.config = config

        logger.info("LightGenerator initialized successfully!")

    def generate(
        self,
        seed: int = 42,
        prompt: str = "",
        negative_prompt: str = "",
        image_path: str = "",
        last_frame_path: str = "",
        audio_path: str = "",
        src_ref_images: Optional[str] = None,
        src_video: Optional[str] = None,
        src_mask: Optional[str] = None,
        save_result_path: Optional[str] = None,
        return_result_tensor: bool = False,
    ) -> str:

        # Create args-like object for set_config compatibility
        class Args:
            def __init__(self):
                self.seed = seed
                self.prompt = prompt
                self.negative_prompt = negative_prompt
                self.image_path = image_path
                self.last_frame_path = last_frame_path
                self.audio_path = audio_path
                self.src_ref_images = src_ref_images
                self.src_video = src_video
                self.src_mask = src_mask
                self.save_result_path = save_result_path
                self.return_result_tensor = return_result_tensor

        args = Args()
        args.task = self.config["task"]

        output_path = Path(args.save_result_path)
        if output_path.suffix:
            logger.info(f"save_result_path: {output_path}")
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / "output.mp4"
            args.save_result_path = str(output_path)
            logger.info(f"Created folder and set file path: {output_path}")

        # Run inference (following LightX2V pattern)
        seed_all(args.seed)
        input_info = set_input_info(args)
        self.runner.run_pipeline(input_info)

        logger.info("Video generated successfully!")
        return args.save_result_path

    @staticmethod
    def _init_runner(config):
        torch.set_grad_enabled(False)
        runner = RUNNER_REGISTER[config["model_cls"]](config)
        runner.init_modules()
        return runner

    @staticmethod
    def _get_default_config_path(model_cls: str, task: str) -> str:
        """Get default config file path based on model class and task."""
        from pathlib import Path

        # Explicit model_cls and task combinations to their default config paths
        # This provides clear mappings for all supported model and task combinations
        explicit_config_mapping = {
            # Wan2.1 configurations
            "wan2.1_t2v": "wan/wan_t2v.json",
            "wan2.1_i2v": "wan/wan_i2v.json",
            "wan2.1_flf2v": "wan/wan_flf2v.json",

            # Wan2.2 MOE configurations
            "wan2.2_moe_t2v": "wan22/wan_moe_t2v.json",
            "wan2.2_moe_i2v": "wan22/wan_moe_i2v.json",
        }

        # Create the composite key
        composite_key = f"{model_cls}_{task}"

        # Check if we have an explicit mapping for this combination
        if composite_key in explicit_config_mapping:
            config_path = Path(get_configs_dir()) / \
                explicit_config_mapping[composite_key]
            if config_path.exists():
                return str(config_path)
            else:
                raise FileNotFoundError(
                    "Default config file not found, please specify the config_json")
