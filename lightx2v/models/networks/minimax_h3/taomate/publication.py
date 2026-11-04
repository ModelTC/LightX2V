"""Incremental publication with TaoMate's original endpoint frame selection."""

import torch


class TaoMateVideoPublication:
    """Apply the whole-video fps selection to independently decoded chunks."""

    def __init__(self, native_frames, published_frames, record):
        if native_frames < 1 or published_frames < 1:
            raise ValueError("TaoMate publication requires positive frame counts")
        self.native_frames = int(native_frames)
        self.published_frames = int(published_frames)
        self.indices = (torch.arange(1, published_frames + 1) * native_frames - 1) // published_frames
        self.record = record
        self.offset = 0
        self.written = 0

    def __call__(self, output, _index):
        for frames in output if isinstance(output, (tuple, list)) else (output,):
            if frames is None:
                continue
            stop = self.offset + frames.shape[2]
            if stop > self.native_frames:
                raise RuntimeError("TaoMate decoder produced more native frames than planned")
            selected = self.indices[(self.indices >= self.offset) & (self.indices < stop)] - self.offset
            if selected.numel():
                self.record(frames.index_select(2, selected.to(frames.device)), layout="BCTHW", value_range=(0, 1))
                self.written += selected.numel()
            self.offset = stop

    def finish(self):
        if self.offset != self.native_frames or self.written != self.published_frames:
            raise RuntimeError(f"TaoMate publication produced {self.offset}/{self.native_frames} native and {self.written}/{self.published_frames} output frames")
