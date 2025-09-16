# LightX2V Server

## Overview

The LightX2V server is a distributed video generation service built with FastAPI that processes image-to-video tasks using a multi-process architecture with GPU support. It implements a sophisticated task queue system with distributed inference capabilities for high-throughput video generation workloads.

## Architecture

### System Architecture

```mermaid
flowchart TB
    Client[Client] -->|Send API Request| Router[FastAPI Router]

    subgraph API Layer
        Router --> TaskRoutes[Task APIs]
        Router --> FileRoutes[File APIs]
        Router --> ServiceRoutes[Service Status APIs]

        TaskRoutes --> CreateTask["POST /v1/tasks/ - Create Task"]
        TaskRoutes --> CreateTaskForm["POST /v1/tasks/form - Form Create"]
        TaskRoutes --> ListTasks["GET /v1/tasks/ - List Tasks"]
        TaskRoutes --> GetTaskStatus["GET /v1/tasks/id/status - Get Status"]
        TaskRoutes --> GetTaskResult["GET /v1/tasks/id/result - Get Result"]
        TaskRoutes --> StopTask["DELETE /v1/tasks/id - Stop Task"]

        FileRoutes --> DownloadFile["GET /v1/files/download/path - Download File"]

        ServiceRoutes --> GetServiceStatus["GET /v1/service/status - Service Status"]
        ServiceRoutes --> GetServiceMetadata["GET /v1/service/metadata - Metadata"]
    end

    subgraph Task Management
        TaskManager[Task Manager]
        TaskQueue[Task Queue]
        TaskStatus[Task Status]
        TaskResult[Task Result]

        CreateTask --> TaskManager
        CreateTaskForm --> TaskManager
        TaskManager --> TaskQueue
        TaskManager --> TaskStatus
        TaskManager --> TaskResult
    end

    subgraph File Service
        FileService[File Service]
        DownloadImage[Download Image]
        DownloadAudio[Download Audio]
        SaveFile[Save File]
        GetOutputPath[Get Output Path]

        FileService --> DownloadImage
        FileService --> DownloadAudio
        FileService --> SaveFile
        FileService --> GetOutputPath
    end

    subgraph Processing Thread
        ProcessingThread[Processing Thread]
        NextTask[Get Next Task]
        ProcessTask[Process Single Task]

        ProcessingThread --> NextTask
        ProcessingThread --> ProcessTask
    end

    subgraph Video Generation Service
        VideoService[Video Service]
        GenerateVideo[Generate Video]

        VideoService --> GenerateVideo
    end

    subgraph Distributed Inference Service
        InferenceService[Distributed Inference Service]
        SubmitTask[Submit Task]
        Worker[Inference Worker Node]
        ProcessRequest[Process Request]
        RunPipeline[Run Inference Pipeline]

        InferenceService --> SubmitTask
        SubmitTask --> Worker
        Worker --> ProcessRequest
        ProcessRequest --> RunPipeline
    end

    %% ====== Connect Modules ======
    TaskQueue --> ProcessingThread
    ProcessTask --> VideoService
    GenerateVideo --> InferenceService
    GetTaskResult --> FileService
    DownloadFile --> FileService
    VideoService --> FileService
    InferenceService --> TaskManager
    TaskManager --> TaskStatus
```

## Task Processing Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant API as API Server
    participant TM as TaskManager
    participant PT as Processing Thread
    participant VS as VideoService
    participant FS as FileService
    participant DIS as DistributedInferenceService
    participant TIW0 as TorchrunInferenceWorker<br/>(Rank 0)
    participant TIW1 as TorchrunInferenceWorker<br/>(Rank 1..N)

    C->>API: POST /v1/tasks<br/>(Create Task)
    API->>TM: create_task()
    TM->>TM: Generate task_id
    TM->>TM: Add to queue<br/>(status: PENDING)
    API->>PT: ensure_processing_thread()
    API-->>C: TaskResponse<br/>(task_id, status: pending)

    Note over PT: Processing Loop
    PT->>TM: get_next_pending_task()
    TM-->>PT: task_id

    PT->>TM: acquire_processing_lock()
    PT->>TM: start_task()<br/>(status: PROCESSING)

    PT->>VS: generate_video_with_stop_event()

    alt Image is URL
        VS->>FS: download_image()
        FS->>FS: HTTP download<br/>with retry
        FS-->>VS: image_path
    else Image is Base64
        VS->>FS: save_base64_image()
        FS-->>VS: image_path
    else Image is local path
        VS->>VS: use existing path
    end

    alt Audio is URL
        VS->>FS: download_audio()
        FS->>FS: HTTP download<br/>with retry
        FS-->>VS: audio_path
    else Audio is Base64
        VS->>FS: save_base64_audio()
        FS-->>VS: audio_path
    else Audio is local path
        VS->>VS: use existing path
    end

    VS->>DIS: submit_task_async(task_data)
    DIS->>TIW0: process_request(task_data)

    Note over TIW0,TIW1: Torchrun-based Distributed Processing
    TIW0->>TIW0: Check if processing
    TIW0->>TIW0: Set processing = True

    alt Multi-GPU Mode (world_size > 1)
        TIW0->>TIW1: broadcast_task_data()<br/>(via DistributedManager)
        Note over TIW1: worker_loop() listens for broadcasts
        TIW1->>TIW1: Receive task_data
    end

    par Parallel Inference across all ranks
        TIW0->>TIW0: runner.set_inputs(task_data)
        TIW0->>TIW0: runner.run_pipeline()
    and
        Note over TIW1: If world_size > 1
        TIW1->>TIW1: runner.set_inputs(task_data)
        TIW1->>TIW1: runner.run_pipeline()
    end

    Note over TIW0,TIW1: Synchronization
    alt Multi-GPU Mode
        TIW0->>TIW1: barrier() for sync
        TIW1->>TIW0: barrier() response
    end

    TIW0->>TIW0: Set processing = False
    TIW0->>DIS: Return result (only rank 0)
    TIW1->>TIW1: Return None (non-rank 0)

    DIS-->>VS: TaskResponse
    VS-->>PT: TaskResponse

    PT->>TM: complete_task()<br/>(status: COMPLETED)
    PT->>TM: release_processing_lock()

    Note over C: Client Polling
    C->>API: GET /v1/tasks/{task_id}/status
    API->>TM: get_task_status()
    TM-->>API: status info
    API-->>C: Task Status

    C->>API: GET /v1/tasks/{task_id}/result
    API->>TM: get_task_status()
    API->>FS: stream_file_response()
    FS-->>API: Video Stream
    API-->>C: Video File
```

## Task States

```mermaid
stateDiagram-v2
    [*] --> PENDING: create_task()
    PENDING --> PROCESSING: start_task()
    PROCESSING --> COMPLETED: complete_task()
    PROCESSING --> FAILED: fail_task()
    PENDING --> CANCELLED: cancel_task()
    PROCESSING --> CANCELLED: cancel_task()
    COMPLETED --> [*]
    FAILED --> [*]
    CANCELLED --> [*]
```

## Configuration

### Environment Variables

see `lightx2v/server/config.py`

### Command Line Arguments

```bash
python -m lightx2v.server.main \
    --model_path /path/to/model \
    --model_cls wan2.1_distill \
    --task i2v \
    --host 0.0.0.0 \
    --port 8000 \
    --config_json /path/to/xxx_config.json
```

```bash
python -m lightx2v.server.main \
    --model_path /path/to/model \
    --model_cls wan2.1_distill \
    --task i2v \
    --host 0.0.0.0 \
    --port 8000 \
    --config_json /path/to/xxx_dist_config.json \
    --nproc_per_node 2
```

## Key Features

### 1. Distributed Processing

- **Multi-process architecture** for GPU parallelization
- **Master-worker pattern** with rank 0 as coordinator
- **PyTorch distributed** backend (NCCL for GPU, Gloo for CPU)
- **Automatic GPU allocation** across processes
- **Task broadcasting** with chunked pickle serialization

### 2. Task Queue Management

- **Thread-safe** task queue with locks
- **Sequential processing** with single processing thread
- **Configurable queue limits** with overflow protection
- **Task prioritization** (FIFO)
- **Automatic cleanup** of old completed tasks
- **Cancellation support** for pending and running tasks

### 3. File Management

- **Multiple input formats**: URL, base64, file upload
- **HTTP downloads** with exponential backoff retry
- **Streaming responses** for large video files
- **Cache management** with automatic cleanup
- **File validation** and format detection

## Performance Considerations

1. **Single Task Processing**: Tasks are processed sequentially to manage GPU memory effectively
2. **Multi-GPU Support**: Distributes inference across available GPUs for parallelization
3. **Connection Pooling**: Reuses HTTP connections to reduce overhead
4. **Streaming Responses**: Large files are streamed to avoid memory issues
5. **Queue Management**: Automatic task cleanup prevents memory leaks
6. **Process Isolation**: Distributed workers run in separate processes for stability

## Monitoring and Debugging

### Logging

The server uses `loguru` for structured logging. Logs include:

- Request/response details
- Task lifecycle events
- Worker process status
- Error traces with context

### Health Checks

- `/v1/service/status` - Overall service health
- `/v1/tasks/queue/status` - Queue status and processing state
- Process monitoring via system tools (htop, nvidia-smi)

### Common Issues

1. **GPU Out of Memory**: Reduce `nproc_per_node` or adjust model batch size
2. **Task Timeout**: Increase `LIGHTX2V_TASK_TIMEOUT` for longer videos
3. **Queue Full**: Increase `LIGHTX2V_MAX_QUEUE_SIZE` or add rate limiting

## Security Considerations

1. **Input Validation**: All inputs validated with Pydantic schemas

## License

See the main project LICENSE file for licensing information.
