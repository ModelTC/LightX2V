import asyncio
import copy
import ctypes
import gc
import json
import os
import tempfile
import threading
import traceback

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.deploy.common.utils import class_try_catch_async
from lightx2v.infer import init_runner  # noqa
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import set_config, set_parallel_config
from lightx2v.utils.utils import seed_all


class BaseWorker:
    @ProfilingContext4DebugL1("Init Worker Worker Cost:")
    def __init__(self, args):
        args.save_video_path = ""
        config = set_config(args)
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        seed_all(config.seed)
        self.rank = 0
        self.world_size = 1
        if config.parallel:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            set_parallel_config(config)
            seed_all(config.seed)
        # same as va_recorder rank and worker main ping rank
        self.out_video_rank = self.world_size - 1
        torch.set_grad_enabled(False)
        self.runner = RUNNER_REGISTER[config.model_cls](config)
        # fixed config
        self.fixed_config = copy.deepcopy(self.runner.config)

    def update_config(self, kwargs):
        for k, v in kwargs.items():
            setattr(self.runner.config, k, v)

    def set_inputs(self, params):
        self.runner.config["prompt"] = params["prompt"]
        self.runner.config["negative_prompt"] = params.get("negative_prompt", "")
        self.runner.config["image_path"] = params.get("image_path", "")
        self.runner.config["save_video_path"] = params.get("save_video_path", "")
        self.runner.config["seed"] = params.get("seed", self.fixed_config.get("seed", 42))
        self.runner.config["audio_path"] = params.get("audio_path", "")

    async def prepare_input_image(self, params, inputs, tmp_dir, data_manager):
        input_image_path = inputs.get("input_image", "")
        tmp_image_path = os.path.join(tmp_dir, input_image_path)

        # prepare tmp image
        if self.runner.config.task == "i2v":
            img_data = await data_manager.load_bytes(input_image_path)
            with open(tmp_image_path, "wb") as fout:
                fout.write(img_data)

        params["image_path"] = tmp_image_path

    async def prepare_input_audio(self, params, inputs, tmp_dir, data_manager):
        input_audio_path = inputs.get("input_audio", "")
        tmp_audio_path = os.path.join(tmp_dir, input_audio_path)

        # for stream audio input, value is dict
        stream_audio_path = params.get("input_audio", None)
        if stream_audio_path is not None:
            tmp_audio_path = stream_audio_path

        if input_audio_path and self.is_audio_model() and isinstance(tmp_audio_path, str):
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

        params["save_video_path"] = tmp_video_path
        return tmp_video_path, output_video_path

    async def prepare_dit_inputs(self, inputs, data_manager):
        device = torch.device("cuda", self.rank)
        text_out = inputs["text_encoder_output"]
        text_encoder_output = await data_manager.load_object(text_out, device)
        image_encoder_output = None

        if self.runner.config.task == "i2v":
            clip_path = inputs["clip_encoder_output"]
            vae_path = inputs["vae_encoder_output"]
            clip_encoder_out = await data_manager.load_object(clip_path, device)
            vae_encoder_out = await data_manager.load_object(vae_path, device)
            image_encoder_output = {
                "clip_encoder_out": clip_encoder_out,
                "vae_encoder_out": vae_encoder_out["vals"],
            }
            # apploy the config changes by vae encoder
            self.update_config(vae_encoder_out["kwargs"])

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

    def is_audio_model(self):
        return "audio" in self.runner.config.model_cls or "seko_talk" in self.runner.config.model_cls


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
            tmp_video_path, output_video_path = self.prepare_output_video(params, outputs, tmp_dir, data_manager)
            logger.info(f"run params: {params}, {inputs}, {outputs}")

            self.set_inputs(params)
            self.runner.stop_signal = False

            future = asyncio.Future()
            self.thread = RunnerThread(asyncio.get_running_loop(), future, self.run_func, self.rank)
            self.thread.start()
            status, _ = await future
            if not status:
                return False
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

        if self.runner.config.task == "i2v" and not self.is_audio_model():
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

        for key in ["lat_h", "lat_w", "tgt_h", "tgt_w"]:
            if hasattr(self.runner.config, key):
                out["kwargs"][key] = int(getattr(self.runner.config, key))

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
        latents = self.runner.run_segment(total_steps=None)
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
            self.runner.process_images_after_vae_decoder(save_video=True)

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
        self.runner.process_images_after_vae_decoder(save_video=True)
