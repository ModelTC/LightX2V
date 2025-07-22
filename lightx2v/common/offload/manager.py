import torch
import threading
import queue
import time
import gc
from loguru import logger
from collections import OrderedDict


class WeightAsyncStreamManager(object):
    def __init__(self, blocks_num, offload_ratio=1, phases_num=1):
        self.active_weights = [None for _ in range(3)]
        self.compute_stream = torch.cuda.Stream(priority=-1)
        self.cpu_load_stream = torch.cuda.Stream(priority=0)
        self.cuda_load_stream = torch.cuda.Stream(priority=0)
        self.offload_block_num = int(offload_ratio * blocks_num)
        self.phases_num = phases_num
        self.block_nums = blocks_num
        self.offload_phases_num = blocks_num * phases_num * offload_ratio

    def prefetch_weights(self, block_idx, blocks_weights):
        with torch.cuda.stream(self.cuda_load_stream):
            self.active_weights[2] = blocks_weights[block_idx]
            self.active_weights[2].to_cuda_async()
        with torch.cuda.stream(self.cpu_load_stream):
            if block_idx < self.offload_block_num:
                if self.active_weights[1] is not None:
                    self.active_weights[1].to_cpu_async()

    def swap_weights(self):
        self.compute_stream.synchronize()
        self.cpu_load_stream.synchronize()
        self.cuda_load_stream.synchronize()

        self.active_weights[0], self.active_weights[1] = (
            self.active_weights[2],
            self.active_weights[0],
        )

    def prefetch_phase(self, block_idx, phase_idx, blocks):
        with torch.cuda.stream(self.cuda_load_stream):
            new_phase = blocks[block_idx].compute_phases[phase_idx]
            new_phase.to_cuda_async()
            self.active_weights[2] = (phase_idx, blocks[block_idx].compute_phases[phase_idx])
        with torch.cuda.stream(self.cpu_load_stream):
            if block_idx * self.phases_num + phase_idx < self.offload_phases_num:
                if self.active_weights[1] is not None:
                    _, old_phase = self.active_weights[1]
                    old_phase.to_cpu_async()

    def swap_phases(self):
        self.compute_stream.synchronize()
        self.cpu_load_stream.synchronize()
        self.cuda_load_stream.synchronize()
        self.active_weights[0], self.active_weights[1] = self.active_weights[2], self.active_weights[0]
        self.active_weights[2] = None


class LazyWeightAsyncStreamManager(WeightAsyncStreamManager):
    def __init__(self, blocks_num, offload_ratio=1, phases_num=1, num_disk_workers=1, max_memory=2, offload_gra="phase"):
        super().__init__(blocks_num, offload_ratio, phases_num)
        self.offload_gra = offload_gra
        self.worker_stop_event = threading.Event()
        self.pin_memory_buffer = MemoryBuffer(max_memory * (1024**3))
        self.disk_task_queue = queue.PriorityQueue()
        self.disk_workers = []
        self.release_workers = []
        self._start_disk_workers(num_disk_workers)
        self.initial_prefetch_done = False
        self.pending_tasks = {}
        self.task_lock = threading.Lock()
        self.last_used_time = {}
        self.time_lock = threading.Lock()

    def _start_disk_workers(self, num_workers):
        for i in range(num_workers):
            if self.offload_gra == "phase":
                worker = threading.Thread(target=self._disk_worker_loop, daemon=True)
            else:
                worker = threading.Thread(target=self._disk_worker_loop_block, daemon=True)
            worker.start()
            self.disk_workers.append(worker)

    def _disk_worker_loop(self):
        while not self.worker_stop_event.is_set():
            try:
                _, task = self.disk_task_queue.get(timeout=0.5)
                if task is None:
                    break

                block_idx, phase_idx, phase = task

                phase.load_from_disk()
                self.pin_memory_buffer.push((block_idx, phase_idx), phase)

                with self.task_lock:
                    if (block_idx, phase_idx) in self.pending_tasks:
                        del self.pending_tasks[(block_idx, phase_idx)]
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Disk worker thread error: {e}")

    def _disk_worker_loop_block(self):
        while not self.worker_stop_event.is_set():
            try:
                _, task = self.disk_task_queue.get(timeout=0.5)
                if task is None:
                    break

                block_idx, block = task

                for phase in block.compute_phases:
                    phase.load_from_disk()
                self.pin_memory_buffer.push(block_idx, block)

                with self.task_lock:
                    if block_idx in self.pending_tasks:
                        del self.pending_tasks[block_idx]
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Disk worker thread error: {e}")

    def _async_prefetch_block(self, blocks, next_block_idx=None):
        if next_block_idx is None:
            next_block_idx = self.pin_memory_buffer.get_max_block_index()

        if next_block_idx < 0:
            next_block_idx = 0

        if next_block_idx == self.block_nums:
            return

        if self.offload_gra == "phase":
            for phase_idx in range(self.phases_num):
                obj_key = (next_block_idx, phase_idx)

                if self.pin_memory_buffer.exists(obj_key) or (obj_key in self.pending_tasks):
                    continue

                with self.task_lock:
                    self.pending_tasks[obj_key] = True

                phase = blocks[next_block_idx].compute_phases[phase_idx]

                priority_key = (next_block_idx, phase_idx)
                self.disk_task_queue.put((priority_key, (next_block_idx, phase_idx, phase)))
        else:
            obj_key = next_block_idx
            if self.pin_memory_buffer.exists(obj_key) or (obj_key in self.pending_tasks):
                return

            with self.task_lock:
                self.pending_tasks[obj_key] = True

            block = blocks[next_block_idx]
            self.disk_task_queue.put((obj_key, (next_block_idx, block)))

    def _sync_prefetch_block(self, blocks):
        block_idx = 0
        while not self.pin_memory_buffer.is_nearly_full():
            if self.offload_gra == "phase":
                for phase_idx in range(self.phases_num):
                    phase = blocks[block_idx].compute_phases[phase_idx]
                    logger.info(f"Synchronous loading: block={block_idx}, phase={phase_idx}")
                    phase.load_from_disk()
                    self.pin_memory_buffer.push((block_idx, phase_idx), phase)
            else:
                block = blocks[block_idx]
                logger.info(f"Synchronous loading: block={block_idx}")
                for phase in block.compute_phases:
                    phase.load_from_disk()
                self.pin_memory_buffer.push(block_idx, block)

            block_idx += 1
            if block_idx == self.block_nums:
                break

    def prefetch_weights_from_disk(self, blocks):
        if self.initial_prefetch_done:
            return

        self._sync_prefetch_block(blocks)
        self.initial_prefetch_done = True

    def prefetch_weights(self, block_idx, blocks):
        obj_key = block_idx

        if not self.pin_memory_buffer.exists(obj_key):
            is_loading = False
            with self.task_lock:
                if obj_key in self.pending_tasks:
                    is_loading = True

            if is_loading:
                start_time = time.time()
                while not self.pin_memory_buffer.exists(obj_key):
                    time.sleep(0.001)
                    if time.time() - start_time > 5:
                        raise TimeoutError(f"Load timeout: block={block_idx}")
            else:
                logger.info("Not find prefetch block={block_idx} task.")
                logger.info("Sync prefetch block={block_idx}.")
                self._async_prefetch_block(blocks, block_idx)
                start_time = time.time()
                for phase_idx in self.phases_num:
                    while not self.pin_memory_buffer.exists((block_idx, phase_idx)):
                        time.sleep(0.001)
                        if time.time() - start_time > 15:
                            raise TimeoutError(f"Load timeout: block={block_idx}, phase={phase_idx}")

        with torch.cuda.stream(self.cuda_load_stream):
            block = self.pin_memory_buffer.get(obj_key)
            block.to_cuda_async()
            self.active_weights[2] = (obj_key, block)

        with torch.cuda.stream(self.cpu_load_stream):
            if block_idx < self.offload_block_num:
                if self.active_weights[1] is not None:
                    old_key, old_block = self.active_weights[1]
                    if self.pin_memory_buffer.exists(old_key):
                        old_block.to_cpu_async()
                        self.pin_memory_buffer.pop(old_key)

    def prefetch_phase(self, block_idx, phase_idx, blocks):
        obj_key = (block_idx, phase_idx)

        if not self.pin_memory_buffer.exists(obj_key):
            is_loading = False
            with self.task_lock:
                if obj_key in self.pending_tasks:
                    is_loading = True

            if is_loading:
                start_time = time.time()
                while not self.pin_memory_buffer.exists(obj_key):
                    time.sleep(0.001)
                    if time.time() - start_time > 5:
                        raise TimeoutError(f"Load timeout: block={block_idx}, phase={phase_idx}")
            else:
                logger.info(f"Not find block={block_idx}, phase={phase_idx} task.")
                logger.info(f"Sync prefetch block={block_idx}, phase={phase_idx}.")
                self._async_prefetch_block(blocks, block_idx)
                start_time = time.time()
                while not self.pin_memory_buffer.exists((block_idx, phase_idx)):
                    time.sleep(0.001)
                    if time.time() - start_time > 5:
                        raise TimeoutError(f"Load timeout: block={block_idx}, phase={phase_idx}")

        with torch.cuda.stream(self.cuda_load_stream):
            phase = self.pin_memory_buffer.get(obj_key)
            phase.to_cuda_async()
            self.active_weights[2] = (obj_key, phase)

        with torch.cuda.stream(self.cpu_load_stream):
            if block_idx * self.phases_num + phase_idx < self.offload_phases_num:
                if self.active_weights[1] is not None:
                    old_key, old_phase = self.active_weights[1]
                    if self.pin_memory_buffer.exists(old_key):
                        old_phase.to_cpu_async()
                        self.pin_memory_buffer.pop(old_key)

    def shutdown(self):
        self.worker_stop_event.set()

        while not self.disk_task_queue.empty():
            try:
                self.disk_task_queue.get_nowait()
            except queue.Empty:
                continue

        for _ in self.disk_workers:
            self.disk_task_queue.put((0, None))

        for worker in self.disk_workers:
            worker.join(timeout=5)

        for worker in self.release_workers:
            worker.join(timeout=5)

        logger.info("All worker threads have been closed")

    def clear(self):
        self.pin_memory_buffer.clear()
        self.shutdown()


class MemoryBuffer:
    def __init__(self, max_memory_bytes=8 * (1024**3)):
        self.cache = OrderedDict()
        self.max_mem = max_memory_bytes
        self.used_mem = 0
        self.obj_size_map = {}
        self.lock = threading.Lock()
        self.insertion_order = []
        self.insertion_index = 0

    def push(self, key, obj):
        with self.lock:
            if key in self.cache:
                return
            if hasattr(obj, "compute_phases"):
                obj_idx = key
                if len(self.obj_size_map) == 0:
                    _size = 0
                    for phase in obj.compute_phases:
                        _size += phase.calculate_size()
                    self.obj_size_map[0] = _size
                size = self.obj_size_map[0]
            else:
                _, obj_idx = key
                if obj_idx not in self.obj_size_map:
                    self.obj_size_map[obj_idx] = obj.calculate_size()
                size = self.obj_size_map[obj_idx]

            self.cache[key] = (size, obj, self.insertion_index)
            self.insertion_order.append((key, self.insertion_index))
            self.insertion_index += 1
            self.used_mem += size

    def _remove_key(self, key):
        if key in self.cache:
            size, obj, idx = self.cache.pop(key)
            try:
                if hasattr(obj, "compute_phases"):
                    for phase in obj.compute_phases:
                        phase.clear()
                else:
                    obj.clear()
            except Exception as e:
                logger.info(f"Error clearing obj: {e}")
            self.used_mem -= size

            self.insertion_order = [(k, i) for (k, i) in self.insertion_order if k != key]

    def get(self, key, default=None):
        with self.lock:
            if key in self.cache:
                size, obj, idx = self.cache[key]
                return obj
        return default

    def exists(self, key):
        with self.lock:
            return key in self.cache

    def pop_front(self):
        with self.lock:
            if not self.insertion_order:
                return False
            front_key, _ = self.insertion_order[0]
            self._remove_key(front_key)
            return True

    def pop(self, key):
        with self.lock:
            if key in self.cache:
                self._remove_key(key)
                return True
        return False

    def is_nearly_full(self):
        with self.lock:
            return self.used_mem >= self.max_mem * 0.9

    def get_max_block_index(self):
        with self.lock:
            if not self.cache:
                return -1
            if isinstance(list(self.cache.keys())[-1], tuple):
                return (list(self.cache.keys())[-1][0] + 1) % 40
            else:
                return (list(self.cache.keys())[-1] + 1) % 40

    def clear(self):
        with self.lock:
            for key in list(self.cache.keys()):
                self._remove_key(key)

            self.insertion_order = []
            self.insertion_index = 0
            self.used_mem = 0
            torch.cuda.empty_cache()
            gc.collect()
