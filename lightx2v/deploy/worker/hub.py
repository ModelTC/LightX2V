import asyncio
import ctypes
import gc
import json
import os
import sys
import tempfile
import threading
import traceback

import torch
import torch.distributed as dist
from loguru import logger

import lightx2v
from lightx2v.deploy.common.utils import class_try_catch_async
from lightx2v.infer import init_runner  # noqa
from lightx2v.utils.input_info import init_empty_input_info
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import set_config, set_parallel_config
from lightx2v.utils.utils import seed_all


def init_tools_preprocess():
    preprocess_path = os.path.abspath(os.path.join(lightx2v.__path__[0], "..", "tools", "preprocess"))
    assert os.path.exists(preprocess_path), f"lightx2v tools preprocess path not found: {preprocess_path}"
    sys.path.append(preprocess_path)


class BaseWorker:
    @ProfilingContext4DebugL1("Init Worker Worker Cost:")
    def __init__(self, args):
        config = set_config(args)
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        seed_all(args.seed)
        self.rank = 0
        self.world_size = 1
        if config["parallel"]:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            set_parallel_config(config)
        # same as va_recorder rank
        self.out_video_rank = int(os.getenv("RECORDER_RANK", "0")) % self.world_size
        torch.set_grad_enabled(False)
        self.runner = RUNNER_REGISTER[config["model_cls"]](config)
        self.input_info = init_empty_input_info(args.task)

    def update_input_info(self, kwargs):
        for k, v in kwargs.items():
            setattr(self.input_info, k, v)

    def set_inputs(self, params):
        self.input_info.prompt = params["prompt"]
        self.input_info.negative_prompt = params.get("negative_prompt", "")
        self.input_info.image_path = params.get("image_path", "")
        self.input_info.save_result_path = params.get("save_result_path", "")
        self.input_info.seed = params.get("seed", self.input_info.seed)
        self.input_info.audio_path = params.get("audio_path", "")
        for k, v in params.get("processed_video_paths", {}).items():
            logger.info(f"set {k} to {v}")
            setattr(self.input_info, k, v)
        self.input_info.last_frame_path = params.get("last_frame_path", "")

        if "stream_config" in self.input_info.__dataclass_fields__:
            self.input_info.stream_config = params.get("stream_config", {})

        if "aspect_ratio" in self.input_info.__dataclass_fields__:
            self.input_info.aspect_ratio = params.get("aspect_ratio", "")

        if "target_shape" in self.input_info.__dataclass_fields__:
            if self.is_t2v_task():
                custom_shape = params.get("custom_shape", [])
                if len(custom_shape) == 2:
                    height, width = int(custom_shape[0]), int(custom_shape[1])
                    if (width <= 1280 and height <= 720 and width >= 256 and height >= 256) or (width <= 720 and height <= 1280 and width >= 256 and height >= 256):
                        self.input_info.target_shape = [height, width]
                        logger.info(f"Set target_height={height}, target_width={width} for t2v task")
                    else:
                        logger.warning(f"Custom shape {width}x{height} is too large or too smallfor t2v task, will use default shape")
            else:
                self.input_info.target_shape = params.get("custom_shape", [])

    async def prepare_input_image(self, params, inputs, tmp_dir, data_manager):
        input_image_path = inputs.get("input_image", "")
        tmp_image_path = os.path.join(tmp_dir, input_image_path)

        # prepare tmp image
        if "image_path" in self.input_info.__dataclass_fields__:
            extra_image_inputs = params.get("extra_inputs", {}).get("input_image", [])

            # for multi image input
            if len(extra_image_inputs) > 0:
                tmp_paths = []
                # xxx-input_image.png -> xxx-input_image
                base_image_path = tmp_image_path.rsplit(".", 1)[0]
                os.makedirs(base_image_path, exist_ok=True)
                for inp in extra_image_inputs:
                    tmp_paths.append(os.path.join(tmp_dir, inputs[inp]))
                    inp_data = await data_manager.load_bytes(inputs[inp])
                    with open(tmp_paths[-1], "wb") as fout:
                        fout.write(inp_data)
                params["image_path"] = ",".join(tmp_paths)
            else:
                img_datas = await data_manager.load_bytes(input_image_path)
                with open(tmp_image_path, "wb") as fout:
                    fout.write(img_datas)
                params["image_path"] = tmp_image_path

    async def prepare_input_video(self, params, inputs, tmp_dir, data_manager):
        if not self.is_animate_model():
            return
        init_tools_preprocess()
        from preprocess_data import get_preprocess_parser, process_input_video

        result_paths = {}
        if self.rank == 0:
            tmp_image_path = params.get("image_path", "")
            assert os.path.exists(tmp_image_path), f"input_image should be save by prepare_input_image but not valid: {tmp_image_path}"

            # prepare tmp input video
            input_video_path = inputs.get("input_video", "")
            tmp_video_path = os.path.join(tmp_dir, input_video_path)
            processed_video_path = os.path.join(tmp_dir, "processe_results")
            video_data = await data_manager.load_bytes(input_video_path)
            with open(tmp_video_path, "wb") as fout:
                fout.write(video_data)

            # prepare preprocess args
            pre_args = get_preprocess_parser().parse_args([])
            pre_args.ckpt_path = self.runner.config["model_path"] + "/process_checkpoint"
            pre_args.video_path = tmp_video_path
            pre_args.refer_path = tmp_image_path
            pre_args.save_path = processed_video_path
            pre_args.replace_flag = self.runner.config.get("replace_flag", False)
            pre_config = self.runner.config.get("preprocess_config", {})
            pre_keys = ["resolution_area", "fps", "replace_flag", "retarget_flag", "use_flux", "iterations", "k", "w_len", "h_len"]
            for k in pre_keys:
                if k in pre_config:
                    setattr(pre_args, k, pre_config[k])

            logger.info(f"Starting video preprocessing in thread pool (this may take a while)...")
            await asyncio.to_thread(process_input_video, pre_args)
            logger.info(f"Video preprocessing completed successfully")

            result_paths = {
                "src_pose_path": os.path.join(processed_video_path, "src_pose.mp4"),
                "src_face_path": os.path.join(processed_video_path, "src_face.mp4"),
                "src_ref_images": os.path.join(processed_video_path, "src_ref.png"),
            }
            if pre_args.replace_flag:
                result_paths["src_bg_path"] = os.path.join(processed_video_path, "src_bg.mp4")
                result_paths["src_mask_path"] = os.path.join(processed_video_path, "src_mask.mp4")

        # for dist, broadcast the video processed result to all ranks
        result_paths = await self.broadcast_data(result_paths, 0)
        for p in result_paths.values():
            assert os.path.exists(p), f"Input video processed result not found: {p}!"
        params["processed_video_paths"] = result_paths

    async def prepare_input_last_frame(self, params, inputs, tmp_dir, data_manager):
        input_last_frame_path = inputs.get("input_last_frame", "")
        tmp_last_frame_path = os.path.join(tmp_dir, input_last_frame_path)

        # prepare tmp last frame
        if "last_frame_path" in self.input_info.__dataclass_fields__:
            img_data = await data_manager.load_bytes(input_last_frame_path)
            with open(tmp_last_frame_path, "wb") as fout:
                fout.write(img_data)
            params["last_frame_path"] = tmp_last_frame_path

    async def prepare_input_audio(self, params, inputs, tmp_dir, data_manager):
        input_audio_path = inputs.get("input_audio", "")
        tmp_audio_path = os.path.join(tmp_dir, input_audio_path)

        # for stream audio input, value is dict
        stream_audio_path = params.get("input_audio", None)
        if stream_audio_path is not None:
            tmp_audio_path = stream_audio_path

        if input_audio_path and self.is_audio_model() and isinstance(tmp_audio_path, str):
            extra_audio_inputs = params.get("extra_inputs", {}).get("input_audio", [])

            # for multi-person audio directory input
            if len(extra_audio_inputs) > 0:
                os.makedirs(tmp_audio_path, exist_ok=True)
                for inp in extra_audio_inputs:
                    tmp_path = os.path.join(tmp_dir, inputs[inp])
                    inp_data = await data_manager.load_bytes(inputs[inp])
                    with open(tmp_path, "wb") as fout:
                        fout.write(inp_data)
            else:
                audio_data = await data_manager.load_bytes(input_audio_path)
                with open(tmp_audio_path, "wb") as fout:
                    fout.write(audio_data)

        params["audio_path"] = tmp_audio_path

    def prepare_output_video(self, params, outputs, tmp_dir, data_manager):
        output_video_path = outputs.get("output_video", "")
        tmp_video_path = os.path.join(tmp_dir, output_video_path)
        if data_manager.name == "local":
            tmp_video_path = os.path.join(data_manager.local_dir, output_video_path)
        # for stream video output, value is dict
        stream_video_path = params.get("output_video", None)
        if stream_video_path is not None:
            tmp_video_path = stream_video_path

        params["save_result_path"] = tmp_video_path
        return tmp_video_path, output_video_path

    def prepare_output_image(self, params, outputs, tmp_dir, data_manager):
        output_image_path = outputs.get("output_image", "")
        tmp_image_path = os.path.join(tmp_dir, output_image_path)
        if data_manager.name == "local":
            tmp_image_path = os.path.join(data_manager.local_dir, output_image_path)
        params["save_result_path"] = tmp_image_path
        return tmp_image_path, output_image_path

    async def prepare_dit_inputs(self, inputs, data_manager):
        device = torch.device("cuda", self.rank)
        text_out = inputs["text_encoder_output"]
        text_encoder_output = await data_manager.load_object(text_out, device)
        image_encoder_output = None

        if "image_path" in self.input_info.__dataclass_fields__:
            clip_path = inputs["clip_encoder_output"]
            vae_path = inputs["vae_encoder_output"]
            clip_encoder_out = await data_manager.load_object(clip_path, device)
            vae_encoder_out = await data_manager.load_object(vae_path, device)
            image_encoder_output = {
                "clip_encoder_out": clip_encoder_out,
                "vae_encoder_out": vae_encoder_out["vals"],
            }
            # apploy the config changes by vae encoder
            self.update_input_info(vae_encoder_out["kwargs"])

        self.runner.inputs = {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
        }

        if self.is_audio_model():
            audio_segments, expected_frames = self.runner.read_audio_input()
            self.runner.inputs["audio_segments"] = audio_segments
            self.runner.inputs["expected_frames"] = expected_frames

    async def save_output_video(self, tmp_video_path, output_video_path, data_manager):
        # save output video
        if data_manager.name != "local" and self.rank == self.out_video_rank and isinstance(tmp_video_path, str):
            video_data = open(tmp_video_path, "rb").read()
            await data_manager.save_bytes(video_data, output_video_path)

    async def save_output_image(self, tmp_image_path, output_image_path, data_manager):
        # save output image
        if data_manager.name != "local" and self.rank == self.out_video_rank and isinstance(tmp_image_path, str):
            image_data = open(tmp_image_path, "rb").read()
            await data_manager.save_bytes(image_data, output_image_path)

    def is_audio_model(self):
        return "audio" in self.runner.config["model_cls"] or "seko_talk" in self.runner.config["model_cls"]

    def is_animate_model(self):
        return self.runner.config.get("task") == "animate"

    def is_image_task(self):
        return self.runner.config.get("task") == "i2i" or self.runner.config.get("task") == "t2i"

    def is_t2v_task(self):
        return self.runner.config.get("task") == "t2v"

    async def broadcast_data(self, data, src_rank=0):
        if self.world_size <= 1:
            return data

        if self.rank == src_rank:
            val = json.dumps(data, ensure_ascii=False).encode("utf-8")
            T = torch.frombuffer(bytearray(val), dtype=torch.uint8).to(device="cuda")
            S = torch.tensor([T.shape[0]], dtype=torch.int32).to(device="cuda")
            logger.info(f"hub rank {self.rank} send data: {data}")
        else:
            S = torch.zeros(1, dtype=torch.int32, device="cuda")

        dist.broadcast(S, src=src_rank)
        if self.rank != src_rank:
            T = torch.zeros(S.item(), dtype=torch.uint8, device="cuda")
        dist.broadcast(T, src=src_rank)

        if self.rank != src_rank:
            val = T.cpu().numpy().tobytes()
            data = json.loads(val.decode("utf-8"))
            logger.info(f"hub rank {self.rank} recv data: {data}")
        return data


class RunnerThread(threading.Thread):
    def __init__(self, loop, future, run_func, rank, *args, **kwargs):
        super().__init__(daemon=True)
        self.loop = loop
        self.future = future
        self.run_func = run_func
        self.args = args
        self.kwargs = kwargs
        self.rank = rank

    def run(self):
        try:
            # cuda device bind for each thread
            torch.cuda.set_device(self.rank)
            res = self.run_func(*self.args, **self.kwargs)
            status = True
        except:  # noqa
            logger.error(f"RunnerThread run failed: {traceback.format_exc()}")
            res = None
            status = False
        finally:

            async def set_future_result():
                self.future.set_result((status, res))

            # add the task of setting future to the loop queue
            asyncio.run_coroutine_threadsafe(set_future_result(), self.loop)

    def stop(self):
        if self.is_alive():
            try:
                logger.warning(f"Force terminate thread {self.ident} ...")
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(self.ident), ctypes.py_object(SystemExit))
            except Exception as e:
                logger.error(f"Force terminate thread failed: {e}")


def class_try_catch_async_with_thread(func):
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except asyncio.CancelledError:
            logger.warning(f"RunnerThread inside {func.__name__} cancelled")
            if hasattr(self, "thread"):
                # self.thread.stop()
                self.runner.stop_signal = True
                self.thread.join()
            raise asyncio.CancelledError
        except Exception:
            logger.error(f"Error in {self.__class__.__name__}.{func.__name__}:")
            traceback.print_exc()
            return None

    return wrapper


class PipelineWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        self.runner.init_modules()
        self.run_func = self.runner.run_pipeline

    @class_try_catch_async_with_thread
    async def run(self, inputs, outputs, params, data_manager):
        with tempfile.TemporaryDirectory() as tmp_dir:
            await self.prepare_input_image(params, inputs, tmp_dir, data_manager)
            await self.prepare_input_audio(params, inputs, tmp_dir, data_manager)
            await self.prepare_input_video(params, inputs, tmp_dir, data_manager)
            await self.prepare_input_last_frame(params, inputs, tmp_dir, data_manager)
            if self.is_image_task():
                tmp_image_path, output_image_path = self.prepare_output_image(params, outputs, tmp_dir, data_manager)
            else:
                tmp_video_path, output_video_path = self.prepare_output_video(params, outputs, tmp_dir, data_manager)
            logger.info(f"run params: {params}, {inputs}, {outputs}")

            self.set_inputs(params)
            self.runner.stop_signal = False

            future = asyncio.Future()
            self.thread = RunnerThread(asyncio.get_running_loop(), future, self.run_func, self.rank, input_info=self.input_info)
            self.thread.start()
            status, _ = await future
            if not status:
                return False
            if self.is_image_task():
                await self.save_output_image(tmp_image_path, output_image_path, data_manager)
            else:
                await self.save_output_video(tmp_video_path, output_video_path, data_manager)
            return True


class TextEncoderWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        self.runner.text_encoders = self.runner.load_text_encoder()

    @class_try_catch_async
    async def run(self, inputs, outputs, params, data_manager):
        logger.info(f"run params: {params}, {inputs}, {outputs}")
        input_image_path = inputs.get("input_image", "")

        self.set_inputs(params)
        prompt = self.runner.config["prompt"]
        img = None

        if self.runner.config["use_prompt_enhancer"]:
            prompt = self.runner.config["prompt_enhanced"]

        if self.runner.config["task"] == "i2v" and not self.is_audio_model():
            img = await data_manager.load_image(input_image_path)
            img = self.runner.read_image_input(img)
            if isinstance(img, tuple):
                img = img[0]

        out = self.runner.run_text_encoder(prompt, img)
        if self.rank == 0:
            await data_manager.save_object(out, outputs["text_encoder_output"])

        del out
        torch.cuda.empty_cache()
        gc.collect()
        return True


class ImageEncoderWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        self.runner.image_encoder = self.runner.load_image_encoder()

    @class_try_catch_async
    async def run(self, inputs, outputs, params, data_manager):
        logger.info(f"run params: {params}, {inputs}, {outputs}")
        self.set_inputs(params)

        img = await data_manager.load_image(inputs["input_image"])
        img = self.runner.read_image_input(img)
        if isinstance(img, tuple):
            img = img[0]
        out = self.runner.run_image_encoder(img)
        if self.rank == 0:
            await data_manager.save_object(out, outputs["clip_encoder_output"])

        del out
        torch.cuda.empty_cache()
        gc.collect()
        return True


class VaeEncoderWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        self.runner.vae_encoder, vae_decoder = self.runner.load_vae()
        del vae_decoder

    @class_try_catch_async
    async def run(self, inputs, outputs, params, data_manager):
        logger.info(f"run params: {params}, {inputs}, {outputs}")
        self.set_inputs(params)
        img = await data_manager.load_image(inputs["input_image"])
        # could change config.lat_h, lat_w, tgt_h, tgt_w
        img = self.runner.read_image_input(img)
        if isinstance(img, tuple):
            img = img[1] if self.runner.vae_encoder_need_img_original else img[0]
        # run vae encoder changed the config, we use kwargs pass changes
        vals = self.runner.run_vae_encoder(img)
        out = {"vals": vals, "kwargs": {}}

        for key in ["original_shape", "resized_shape", "latent_shape", "target_shape"]:
            if hasattr(self.input_info, key):
                out["kwargs"][key] = getattr(self.input_info, key)

        if self.rank == 0:
            await data_manager.save_object(out, outputs["vae_encoder_output"])

        del out, img, vals
        torch.cuda.empty_cache()
        gc.collect()
        return True


class DiTWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        self.runner.model = self.runner.load_transformer()

    @class_try_catch_async_with_thread
    async def run(self, inputs, outputs, params, data_manager):
        logger.info(f"run params: {params}, {inputs}, {outputs}")
        self.set_inputs(params)

        await self.prepare_dit_inputs(inputs, data_manager)
        self.runner.stop_signal = False
        future = asyncio.Future()
        self.thread = RunnerThread(asyncio.get_running_loop(), future, self.run_dit, self.rank)
        self.thread.start()
        status, out = await future
        if not status:
            return False

        if self.rank == 0:
            await data_manager.save_tensor(out, outputs["latents"])

        del out
        torch.cuda.empty_cache()
        gc.collect()
        return True

    def run_dit(self):
        self.runner.init_run()
        assert self.runner.video_segment_num == 1, "DiTWorker only support single segment"
        latents = self.runner.run_segment()
        self.runner.end_run()
        return latents


class VaeDecoderWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        vae_encoder, self.runner.vae_decoder = self.runner.load_vae()
        self.runner.vfi_model = self.runner.load_vfi_model() if "video_frame_interpolation" in self.runner.config else None
        del vae_encoder

    @class_try_catch_async
    async def run(self, inputs, outputs, params, data_manager):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_video_path, output_video_path = self.prepare_output_video(params, outputs, tmp_dir, data_manager)
            logger.info(f"run params: {params}, {inputs}, {outputs}")
            self.set_inputs(params)

            device = torch.device("cuda", self.rank)
            latents = await data_manager.load_tensor(inputs["latents"], device)
            self.runner.gen_video = self.runner.run_vae_decoder(latents)
            self.runner.process_images_after_vae_decoder()

            await self.save_output_video(tmp_video_path, output_video_path, data_manager)

            del latents
            torch.cuda.empty_cache()
            gc.collect()
            return True


class SegmentDiTWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        self.runner.model = self.runner.load_transformer()
        self.runner.vae_encoder, self.runner.vae_decoder = self.runner.load_vae()
        self.runner.vfi_model = self.runner.load_vfi_model() if "video_frame_interpolation" in self.runner.config else None
        if self.is_audio_model():
            self.runner.audio_encoder = self.runner.load_audio_encoder()
            self.runner.audio_adapter = self.runner.load_audio_adapter()
            self.runner.model.set_audio_adapter(self.runner.audio_adapter)

    @class_try_catch_async_with_thread
    async def run(self, inputs, outputs, params, data_manager):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_video_path, output_video_path = self.prepare_output_video(params, outputs, tmp_dir, data_manager)
            await self.prepare_input_audio(params, inputs, tmp_dir, data_manager)
            logger.info(f"run params: {params}, {inputs}, {outputs}")
            self.set_inputs(params)

            await self.prepare_dit_inputs(inputs, data_manager)
            self.runner.stop_signal = False
            future = asyncio.Future()
            self.thread = RunnerThread(asyncio.get_running_loop(), future, self.run_dit, self.rank)
            self.thread.start()
            status, _ = await future
            if not status:
                return False

            await self.save_output_video(tmp_video_path, output_video_path, data_manager)

            torch.cuda.empty_cache()
            gc.collect()
            return True

    def run_dit(self):
        self.runner.run_main()
        self.runner.process_images_after_vae_decoder()
