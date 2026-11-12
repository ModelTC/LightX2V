"""H3 reader that owns its Bilibili adapter and background services."""

import threading
from pathlib import Path

from lightx2v.utils.va_reader_ws import MinimaxH3CausalWSReader

from .adapter import BiliLiveChatAdapter
from .config import BilibiliConfig


class BiliLiveOmniVAReader(MinimaxH3CausalWSReader):
    def __init__(self, *args, stream_config=None, va_recorder=None, **kwargs):
        self.started_event = threading.Event()
        super().__init__(*args, stream_config=dict(stream_config or {}), **kwargs)
        if self.rank == self.target_rank:
            image_path = self.model_runner.input_info.image_path
            work_dir = self.stream_config.get("bilibili_work_dir") or str(Path(image_path).parent)
            self.chat_adapter = BiliLiveChatAdapter(
                config=BilibiliConfig.from_env(self.stream_config),
                image_path=image_path,
                work_dir=work_dir,
                model_runner=self.model_runner,
            )
            self.chat_adapter.recorder = va_recorder
            # Reuse the base reader's binding logic without accepting a source
            # from the session or serializing runtime objects in input_info.
            self._preset_audio_source = self.chat_adapter

    def start(self) -> None:
        try:
            if self.rank == self.target_rank:
                self.chat_adapter.seg_duration = self.segment_duration
                self.chat_adapter.skip_immediate_switch = True
                self.chat_adapter.start()
            # H3 still owns audio chunk geometry, resampling and rank barriers.
            super().start()
            self.started_event.set()
        except Exception:
            self.stop()
            raise

    def on_clear_audio(self, note: str = "") -> None:
        self._control_adapter().on_clear_audio(note)

    def on_image_switch(self, image_path: str) -> None:
        self._control_adapter().on_image_switch(image_path)

    def on_image_switched(self, image_path: str, error: str = "") -> None:
        if self.rank == self.target_rank and self.chat_adapter is not None:
            self.chat_adapter.on_image_switched(image_path, error)

    def on_action_switch(self, prompt: str, duration: float = 0) -> None:
        self._control_adapter().on_action_switch(prompt, duration)

    def _control_adapter(self) -> BiliLiveChatAdapter:
        if not self.started_event.is_set() or self.chat_adapter is None:
            raise RuntimeError("Bilibili reader is not ready on this rank")
        return self.chat_adapter

    def stop(self) -> None:
        try:
            super().stop()
        finally:
            self._preset_audio_source = None
            self.started_event.clear()
