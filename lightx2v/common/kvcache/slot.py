import torch


class SlotMetadata:
    """Upload ring slot plans while retaining metadata buffers for CUDA Graph."""

    def __init__(self, metadata_shape: tuple[int, ...], device: torch.device):
        shape = (*metadata_shape, 3)
        self._host_metadata = torch.empty(shape, dtype=torch.int32, device="cpu", pin_memory=True)
        self._gpu_metadata = torch.empty(shape, dtype=torch.int32, device=device)
        self._upload_complete = torch.cuda.Event()
        self._upload_pending = False

    @property
    def metadata(self) -> torch.Tensor:
        return self._gpu_metadata

    def prepare(self, index: int | tuple[int, ...], write_start: int, sink_tokens: int, recent_head: int) -> None:
        # The pinned source cannot change while its previous upload is in flight.
        if self._upload_pending:
            self._upload_complete.synchronize()
            self._upload_pending = False
        metadata = self._host_metadata[index]
        metadata[0] = write_start
        metadata[1] = sink_tokens
        metadata[2] = recent_head

    def upload(self) -> None:
        self._gpu_metadata.copy_(self._host_metadata, non_blocking=True)
        self._upload_complete.record(torch.cuda.current_stream(self._gpu_metadata.device))
        self._upload_pending = True
