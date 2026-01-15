import gc
import json
import os

import torch
from loguru import logger

from lightx2v.deploy.common.va_controller import VAController
from lightx2v.infer import init_runner  # noqa
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import set_config, set_parallel_config
from lightx2v.utils.utils import seed_all
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


class ParallelModelRunner:
    def __init__(self, args):
        self.clip_runners = {}
        self.cur_name = None
        self.is_parallel = False
        self.prev_frame_length = None
        self.segment_durations = {}
        self.prev_durations = {}

        with open(args.config_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        clip_configs = cfg["clip_configs"]
        for clip_config in clip_configs:
            args.config_json = clip_config["path"]
            self.create_clip_runner(clip_config["name"], args)

        assert len(self.clip_runners) == 2, "Exactly 2 clip runners are required!"
        assert "s2v_clip" in self.clip_runners, "s2v_clip must be in the clip runners"
        assert "f2v_clip" in self.clip_runners, "f2v_clip must be in the clip runners"

    def create_clip_runner(self, name, args):
        config = set_config(args)
        logger.info(f"clip {name} config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")

        target_fps = config.get("target_fps", 16)
        max_num_frames = config.get("target_video_length", 81)
        prev_frame_length = config.get("prev_frame_length", 5)
        if self.prev_frame_length is None:
            self.prev_frame_length = prev_frame_length
        # check all clip runners have the same prev_frame_length
        assert self.prev_frame_length == prev_frame_length, "All clip runners must have the same prev_frame_length"
        self.segment_durations[name] = max_num_frames / target_fps
        self.prev_durations[name] = prev_frame_length / target_fps

        seed_all(args.seed)
        if config.get("parallel", False):
            if not self.is_parallel:
                platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
                platform_device.init_parallel_env()
                self.is_parallel = True
            set_parallel_config(config)

        torch.set_grad_enabled(False)
        self.clip_runners[name] = RUNNER_REGISTER[config["model_cls"]](config)
        if self.cur_name is None:
            self.cur_name = name
        self.clip_runners[name].stop_signal = False
        self.clip_runners[name].pause_signal = False
        logger.info(f"Clip {name} initialized successfully!")

    def init_modules(self):
        for name, runner in self.clip_runners.items():
            runner.init_modules()
            logger.info(f"Clip {name} modules initialized successfully!")

    def set_config(self, args, **kwargs):
        self.clip_runners[self.cur_name].set_config(args, **kwargs)

    def init_run(self):
        return self.clip_runners[self.cur_name].init_run()

    def end_run(self):
        return self.clip_runners[self.cur_name].end_run()

    def init_run_segment(self, segment_idx, audio_array):
        return self.clip_runners[self.cur_name].init_run_segment(segment_idx, audio_array)

    def run_segment(self, segment_idx):
        return self.clip_runners[self.cur_name].run_segment(segment_idx)

    def end_run_segment_stream(self, latents):
        return self.clip_runners[self.cur_name].end_run_segment_stream(latents)

    def check_stop(self):
        return self.clip_runners[self.cur_name].check_stop()

    @property
    def config(self):
        return self.clip_runners[self.cur_name].config

    @property
    def input_info(self):
        return self.clip_runners[self.cur_name].input_info

    @property
    def vfi_model(self):
        return self.clip_runners[self.cur_name].vfi_model

    @property
    def vsr_model(self):
        return self.clip_runners[self.cur_name].vsr_model

    @property
    def video_segment_num(self):
        return self.clip_runners[self.cur_name].video_segment_num

    @property
    def stop_signal(self):
        return self.clip_runners[self.cur_name].stop_signal

    @property
    def pause_signal(self):
        return self.clip_runners[self.cur_name].pause_signal

    @stop_signal.setter
    def stop_signal(self, value):
        self.clip_runners[self.cur_name].stop_signal = value

    @pause_signal.setter
    def pause_signal(self, value):
        self.clip_runners[self.cur_name].pause_signal = value

    def _switch_model(self, name):
        assert name in self.clip_runners, f"Clip {name} not found"
        self.cur_name = name

    def _run_input_encoder(self, input_info=None):
        for name, runner in self.clip_runners.items():
            with ProfilingContext4DebugL1(f"run_input_encoder_{name}"):
                if input_info is not None:
                    runner.input_info = input_info
                runner.inputs = runner.run_input_encoder()
                logger.info(f"Clip {name} input encoder run successfully!")

    def _change_image_path(self, image_path):
        for runner in self.clip_runners.values():
            runner.input_info.image_path = image_path

    def _change_prev_video(self, prev_video):
        for runner in self.clip_runners.values():
            runner.prev_video = prev_video

    def _reset_prev_video(self):
        for runner in self.clip_runners.values():
            if runner.config.get("f2v_process", False):
                runner.prev_video = runner.ref_img.unsqueeze(2)
            else:
                runner.prev_video = None

    def _broadcast_prev_video(self):
        runner = self.clip_runners[self.cur_name]
        self._change_prev_video(runner.prev_video)

    def _clear_va_controller(self):
        for r in self.clip_runners.values():
            r.va_controller = None

    def _set_va_controller(self, va_controller):
        for r in self.clip_runners.values():
            r.va_controller = None
        self.clip_runners[self.cur_name].va_controller = va_controller

    def _update_prompt(self, prompt):
        runner = self.clip_runners[self.cur_name]
        runner.input_info.prompt = prompt
        text_encoder_output = runner.run_text_encoder(runner.input_info)
        torch.cuda.empty_cache()
        gc.collect()
        runner.inputs["text_encoder_output"] = text_encoder_output

    @property
    def _segment_duration(self):
        return self.segment_durations[self.cur_name]

    @property
    def _prev_duration(self):
        return self.prev_durations[self.cur_name]

    def run_pipeline(self, input_info):
        try:
            va_controller = None
            va_controller = VAController(self)
            logger.info(f"init va_recorder: {va_controller.recorder} and va_reader: {va_controller.reader}")
            assert va_controller.reader is not None, "va_reader is required for parallel model runner"
            va_controller.start()

            self._run_input_encoder(input_info)
            self.init_run()

            # steam audio input, video segment num is unlimited
            self.video_segment_num = 10000000
            segment_idx = 0
            fail_count, max_fail_count = 0, 10
            va_controller.before_control()

            while True:
                with ProfilingContext4DebugL1(f"stream segment get audio segment {segment_idx}"):
                    self._switch_model("s2v_clip")
                    control = va_controller.next_control()

                    # blank -> voice
                    if control.action == "switch":
                        self._change_prev_video(control.data)

                    # base image changed
                    elif control.action == "image":
                        self._change_image_path(control.data)
                        self._run_input_encoder()
                        self._reset_prev_video()

                    # person perform some actions
                    elif control.action == "action":
                        self._switch_model("f2v_clip")
                        self._update_prompt(control.data)

                    # bufferd stream is enough, sleep for a while
                    elif control.action == "wait":
                        time.sleep(0.01)
                        continue

                    audio_array, valid_duration = va_controller.reader.get_audio_segment(
                        segment_duration=self._segment_duration,
                        prev_duration=self._prev_duration,
                    )
                    # for f2v clip, even audio is blank, should not truncate this frames
                    if self.cur_name == "f2v_clip":
                        valid_duration = self._segment_duration
                    if audio_array is None:
                        fail_count += 1
                        logger.warning(f"Failed to get audio chunk {fail_count} times")
                        if fail_count > max_fail_count:
                            raise Exception(f"Failed to get audio chunk {fail_count} times, stop reader")
                        continue

                with ProfilingContext4DebugL1(f"stream segment end2end {segment_idx}"):
                    try:
                        # reset pause signal
                        self.pause_signal = False
                        self._set_va_controller(va_controller)
                        self.init_run_segment(segment_idx, audio_array)
                        self.check_stop()
                        latents = self.run_segment(segment_idx)
                        self.check_stop()
                        self.end_run_segment_stream(latents, valid_duration=valid_duration)
                        self._broadcast_prev_video()
                        segment_idx += 1
                        fail_count = 0
                    except Exception as e:
                        if "pause_signal, pause running" in str(e):
                            logger.warning(f"model infer audio pause: {e}, should continue")
                        else:
                            raise
        finally:
            self._clear_va_controller()
            self.end_run()
            if va_controller is not None:
                va_controller.clear()
                va_controller = None
