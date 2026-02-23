import os
import torch
import torch.distributed as dist
import tempfile
from PIL import Image
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

# 导入 MOVA 官方模块（确保已在环境中安装）
from mova.diffusion.pipelines.pipeline_mova import MOVA
from mova.datasets.transforms.custom import crop_and_resize
from mova.utils.data import save_video_with_audio


@RUNNER_REGISTER("mova")
class MOVARunner(DefaultRunner):
    """
    LightX2V Runner for MOVA, directly using the official inference pipeline.
    Ensures distributed process group is initialized before calling the pipeline.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config["task"] = "i2v"  # 避免 i2av 方法缺失
        self._dist_initialized = False  # 记录分布式初始化状态

        # 推理参数
        self.num_frames = self.config.get("num_frames", 193)
        self.height = self.config.get("height", 352)
        self.width = self.config.get("width", 640)
        self.video_fps = self.config.get("video_fps", 24.0)
        self.num_inference_steps = self.config.get("num_inference_steps", 50)
        self.cfg_scale = self.config.get("cfg_scale", 5.0)
        self.sigma_shift = self.config.get("sigma_shift", 5.0)

    def _init_distributed(self):
        """初始化分布式进程组（如果尚未初始化）"""
        if self._dist_initialized:
            return True
        if dist.is_initialized():
            print("Distributed already initialized externally.")
            self._dist_initialized = True
            return True

        try:
            # 检查是否由 torchrun 启动（环境变量通常已设置）
            if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
                dist.init_process_group(backend="nccl", init_method="env://")
                print("Initialized distributed from environment (torchrun).")
                self._dist_initialized = True
                return True
            else:
                # 单卡模式：使用临时文件初始化（避免端口冲突）
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(local_rank)
                # 创建临时文件作为共享文件
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                init_method = f"file://{temp_file.name}"
                temp_file.close()
                dist.init_process_group(
                    backend="nccl",
                    init_method=init_method,
                    rank=0,
                    world_size=1
                )
                print("Initialized single-process distributed group with file init_method.")
                self._dist_initialized = True
                return True
        except Exception as e:
            print(f"Failed to initialize distributed group: {e}")
            print("Will attempt to proceed without distributed (may still fail if pipeline requires it).")
            # 这里不设置标志，后续调用仍会尝试
            return False

    def load_model(self):
        """加载官方 MOVA pipeline，并确保分布式初始化"""
        ckpt_path = self.config["model_path"]
        torch_dtype = GET_DTYPE()
        self.pipeline = MOVA.from_pretrained(ckpt_path, torch_dtype=torch_dtype)
        self.pipeline.to(AI_DEVICE)

        # 立即初始化分布式组
        self._init_distributed()

        print(f"MOVA pipeline loaded from {ckpt_path}")

    def run_pipeline(self, input_info):
        # 再次检查分布式初始化（如果之前失败，这里再试一次）
        if not self._dist_initialized:
            if not self._init_distributed():
                os.environ.setdefault("MASTER_ADDR", "localhost")
                os.environ.setdefault("MASTER_PORT", "29500")
                try:
                    dist.init_process_group(
                        backend="nccl",
                        init_method="env://",
                        rank=0,
                        world_size=1
                    )
                    print("Initialized distributed with env:// fallback.")
                    self._dist_initialized = True
                except Exception as e:
                    print(f"Critical: cannot initialize distributed: {e}")
                    raise RuntimeError(
                        "Failed to initialize distributed process group. "
                        "Please run with torchrun: torchrun --nproc_per_node=1 mova_t2v.py"
                    ) from e

        # 获取输入参数
        prompt = input_info.prompt
        negative_prompt = input_info.negative_prompt or ""
        image_path = input_info.image_path
        save_path = input_info.save_result_path
        seed = input_info.seed

        if not image_path:
            raise ValueError("input_info.image_path must be provided.")
        if not save_path:
            raise ValueError("input_info.save_result_path must be provided.")

        # 图像预处理
        img = Image.open(image_path).convert("RGB")
        img = crop_and_resize(img, self.height, self.width)

        # 调用官方 pipeline
        with torch.no_grad():
            video, audio = self.pipeline(
                prompt=prompt,
                image=img,
                negative_prompt=negative_prompt,
                seed=seed,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                video_fps=self.video_fps,
                num_inference_steps=self.num_inference_steps,
                cfg_scale=self.cfg_scale,
                sigma_shift=self.sigma_shift,
            )

        # 处理返回的视频格式
        print(f"Raw video type: {type(video)}")
        print(f"Raw video length: {len(video)}")
        if len(video) > 0:
            print(f"First element type: {type(video[0])}")
            # 如果第一项是列表，说明 video 是嵌套列表 [[frames]]，需要解包
            if isinstance(video[0], list):
                print("Detected nested list, unpacking...")
                video = video[0]
                print(f"Unpacked video length: {len(video)}")
            # 确保每个帧是 PIL Image
            if len(video) > 0 and hasattr(video[0], 'mode'):
                print(f"First frame mode: {video[0].mode}")
            else:
                print("Warning: first element is not a PIL Image, trying to convert...")
        # 统一转换为 RGB（解决 imageio 通道数错误）
        video = [frame.convert('RGB') for frame in video]

        # 保存视频
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sample_rate = self.pipeline.audio_sample_rate
        save_video_with_audio(
            video,
            audio.cpu().squeeze(),
            save_path,
            fps=self.video_fps,
            sample_rate=sample_rate,
            quality=9,
        )
        print(f"Video saved to {save_path}")

        # 兼容 LightX2V 返回值
        self.gen_video_final = torch.zeros(1)
        return self.gen_video_final

    # 以下占位方法
    def run_text_encoder(self, input_info):
        pass

    def run_vae_encoder(self, img=None):
        pass

    def process_images_after_vae_decoder(self):
        pass

    def init_run(self):
        pass