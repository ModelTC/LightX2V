import argparse
import importlib.machinery
import os
import sys
import tempfile
import types
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_ROOT = os.path.join(REPO_ROOT, "app")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

lightx2v_pkg = types.ModuleType("lightx2v")
lightx2v_pkg.__path__ = [os.path.join(REPO_ROOT, "lightx2v")]
sys.modules.setdefault("lightx2v", lightx2v_pkg)

lightx2v_platform_pkg = types.ModuleType("lightx2v_platform")
lightx2v_platform_pkg.__path__ = [os.path.join(REPO_ROOT, "lightx2v_platform")]
sys.modules.setdefault("lightx2v_platform", lightx2v_platform_pkg)

lightx2v_platform_base_pkg = types.ModuleType("lightx2v_platform.base")
lightx2v_platform_base_pkg.__path__ = [os.path.join(REPO_ROOT, "lightx2v_platform", "base")]
sys.modules.setdefault("lightx2v_platform.base", lightx2v_platform_base_pkg)

global_var_module = types.ModuleType("lightx2v_platform.base.global_var")
global_var_module.AI_DEVICE = "cuda"
sys.modules.setdefault("lightx2v_platform.base.global_var", global_var_module)

lightx2v_utils_module = types.ModuleType("lightx2v.utils.utils")
lightx2v_utils_module.is_main_process = lambda: True
sys.modules.setdefault("lightx2v.utils.utils", lightx2v_utils_module)

if "loguru" not in sys.modules:
    sys.modules["loguru"] = types.SimpleNamespace(
        logger=types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None)
    )
if "psutil" not in sys.modules:
    sys.modules["psutil"] = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=0))
if "huggingface_hub" not in sys.modules:
    hf_module = types.ModuleType("huggingface_hub")
    hf_module.HfApi = object
    hf_module.list_repo_files = lambda *a, **k: []
    hf_module.__spec__ = importlib.machinery.ModuleSpec("huggingface_hub", loader=None)
    sys.modules["huggingface_hub"] = hf_module
if "modelscope" not in sys.modules:
    modelscope_module = types.ModuleType("modelscope")
    modelscope_module.__spec__ = importlib.machinery.ModuleSpec("modelscope", loader=None)
    hub_module = types.ModuleType("modelscope.hub")
    hub_module.__spec__ = importlib.machinery.ModuleSpec("modelscope.hub", loader=None)
    api_module = types.ModuleType("modelscope.hub.api")
    api_module.__spec__ = importlib.machinery.ModuleSpec("modelscope.hub.api", loader=None)
    api_module.HubApi = object
    sys.modules["modelscope"] = modelscope_module
    sys.modules["modelscope.hub"] = hub_module
    sys.modules["modelscope.hub.api"] = api_module
if "torch" not in sys.modules:
    torch_module = types.ModuleType("torch")
    torch_module.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    torch_module.float16 = "float16"
    torch_module.float32 = "float32"
    torch_module.bfloat16 = "bfloat16"
    torch_module.Tensor = object
    torch_module._scaled_mm = object()
    torch_module.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        get_device_capability=lambda *_: (0, 0),
        get_device_name=lambda *_: "",
        empty_cache=lambda: None,
        synchronize=lambda: None,
    )
    torch_module.device = lambda value: value

    dist_module = types.ModuleType("torch.distributed")
    dist_module.is_initialized = lambda: False
    dist_module.get_rank = lambda: 0
    dist_module.get_world_size = lambda: 1
    dist_module.all_reduce = lambda *_args, **_kwargs: None

    tensor_module = types.ModuleType("torch.distributed.tensor")
    device_mesh_module = types.ModuleType("torch.distributed.tensor.device_mesh")
    device_mesh_module.init_device_mesh = lambda *_args, **_kwargs: None

    torch_module.distributed = dist_module
    sys.modules["torch"] = torch_module
    sys.modules["torch.distributed"] = dist_module
    sys.modules["torch.distributed.tensor"] = tensor_module
    sys.modules["torch.distributed.tensor.device_mesh"] = device_mesh_module

from utils.model_utils import get_model_configs
from lightx2v.utils.set_config import set_config


class HeliosDistilledSupportTest(unittest.TestCase):
    def test_get_model_configs_detects_helios_distilled_variant(self):
        config = get_model_configs(
            model_type_input="Helios",
            model_path_input="/data1/models/BestWishYSH/Helios-Distilled",
            dit_path_input=None,
            high_noise_path_input=None,
            low_noise_path_input=None,
            t5_path_input=None,
            clip_path_input=None,
            vae_path_input=None,
            qwen_image_dit_path_input=None,
            qwen_image_vae_path_input=None,
            qwen_image_scheduler_path_input=None,
            qwen25vl_encoder_path_input=None,
            z_image_dit_path_input=None,
            z_image_vae_path_input=None,
            z_image_scheduler_path_input=None,
            qwen3_encoder_path_input=None,
            quant_op="triton",
        )

        self.assertEqual(config["model_cls"], "helios")
        self.assertEqual(config["model_variant"], "distilled")
        self.assertEqual(config["scheduler_type"], "HeliosDMDScheduler")
        self.assertEqual(config["model_path"], "/data1/models/BestWishYSH/Helios-Distilled")
        self.assertEqual(config["transformer_model_path"], "/data1/models/BestWishYSH/Helios-Distilled/transformer")
        self.assertEqual(config["text_encoder_path"], "/data1/models/BestWishYSH/Helios-Distilled/text_encoder")
        self.assertEqual(config["tokenizer_path"], "/data1/models/BestWishYSH/Helios-Distilled/tokenizer")
        self.assertEqual(config["vae_path"], "/data1/models/BestWishYSH/Helios-Distilled/vae")
        self.assertEqual(config["scheduler_path"], "/data1/models/BestWishYSH/Helios-Distilled/scheduler")
        self.assertTrue(config["is_distilled"])

    def test_set_config_loads_helios_transformer_and_scheduler_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = os.path.join(tmpdir, "Helios-Distilled")
            os.makedirs(os.path.join(model_root, "transformer"))
            os.makedirs(os.path.join(model_root, "scheduler"))
            os.makedirs(os.path.join(model_root, "text_encoder"))
            os.makedirs(os.path.join(model_root, "tokenizer"))
            os.makedirs(os.path.join(model_root, "vae"))

            with open(os.path.join(model_root, "configuration.json"), "w", encoding="utf-8") as f:
                f.write('{"model_type": "helios"}')
            with open(os.path.join(model_root, "model_index.json"), "w", encoding="utf-8") as f:
                f.write(
                    '{"_class_name":"HeliosPyramidPipeline","is_distilled":true,'
                    '"scheduler":["diffusers","HeliosDMDScheduler"],'
                    '"transformer":["diffusers","HeliosTransformer3DModel"],'
                    '"text_encoder":["transformers","UMT5EncoderModel"],'
                    '"tokenizer":["transformers","T5TokenizerFast"],'
                    '"vae":["diffusers","AutoencoderKLWan"]}'
                )
            with open(os.path.join(model_root, "transformer", "config.json"), "w", encoding="utf-8") as f:
                f.write('{"num_layers": 40, "patch_size": [1, 2, 2], "in_channels": 16, "out_channels": 16}')
            with open(os.path.join(model_root, "scheduler", "scheduler_config.json"), "w", encoding="utf-8") as f:
                f.write('{"_class_name":"HeliosDMDScheduler","stages":3}')
            with open(os.path.join(model_root, "vae", "config.json"), "w", encoding="utf-8") as f:
                f.write('{"temperal_downsample":[false,true,true]}')

            args = argparse.Namespace(
                model_cls="helios",
                model_variant="distilled",
                task="t2v",
                model_path=model_root,
                target_video_length=99,
            )

            config = set_config(args)
        self.assertEqual(config["scheduler_type"], "HeliosDMDScheduler")
        self.assertEqual(config["num_layers"], 40)
        self.assertEqual(config["patch_size"], [1, 2, 2])
        self.assertEqual(config["vae_scale_factor"], 8)
        self.assertTrue(config["is_distilled"])

    def test_helios_runner_is_native_not_pipeline_bridge(self):
        runner_path = os.path.join(REPO_ROOT, "lightx2v", "models", "runners", "helios", "helios_runner.py")
        with open(runner_path, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn("class HeliosRunner", source)
        self.assertNotIn("HeliosPyramidPipeline", source)
        self.assertNotIn("HeliosPipeline", source)

    def test_infer_cli_exposes_helios_model_cls(self):
        infer_path = os.path.join(REPO_ROOT, "lightx2v", "infer.py")
        with open(infer_path, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn('"helios"', source)

    def test_validate_config_paths_has_helios_branch(self):
        utils_path = os.path.join(REPO_ROOT, "lightx2v", "utils", "utils.py")
        with open(utils_path, "r", encoding="utf-8") as f:
            source = f.read()

        self.assertIn('config.get("model_cls") == "helios"', source)


if __name__ == "__main__":
    unittest.main()
