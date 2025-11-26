# LightX2V Server

## Overview

The LightX2V server is a distributed video/image generation service built with FastAPI that processes image-to-video and text-to-image tasks using a multi-process architecture with GPU support. It implements a sophisticated task queue system with distributed inference capabilities for high-throughput generation workloads.

## Directory Structure

```
server/
├── __init__.py
├── __main__.py              # Entry point
├── main.py                  # Server startup
├── config.py                # Configuration
├── task_manager.py          # Task management
├── schema.py                # Data models (VideoTaskRequest, ImageTaskRequest)
├── api/
│   ├── __init__.py
│   ├── router.py            # Main router aggregation
│   ├── deps.py              # Dependency injection container
│   ├── server.py            # ApiServer class
│   ├── files.py             # /v1/files/*
│   ├── service_routes.py    # /v1/service/*
│   └── tasks/
│       ├── __init__.py
│       ├── common.py        # Common task operations
│       ├── video.py         # POST /v1/tasks/video
│       └── image.py         # POST /v1/tasks/image
├── services/
│   ├── __init__.py
│   ├── file_service.py      # File service (unified download)
│   ├── distributed_utils.py # Distributed manager
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── worker.py        # TorchrunInferenceWorker
│   │   └── service.py       # DistributedInferenceService
│   └── generation/
│       ├── __init__.py
│       ├── base.py          # Base generation service
│       ├── video.py         # VideoGenerationService
│       └── image.py         # ImageGenerationService
├── media/
│   ├── __init__.py
│   ├── base.py              # MediaHandler base class
│   ├── image.py             # ImageHandler
│   └── audio.py             # AudioHandler
└── metrics/                 # Prometheus metrics
```

## Architecture

### System Architecture

```mermaid
flowchart TB
    Client[Client] -->|Send API Request| Router[FastAPI Router]

    subgraph API Layer
        Router --> TaskRoutes[Task APIs]
        Router --> FileRoutes[File APIs]
        Router --> ServiceRoutes[Service Status APIs]

        TaskRoutes --> CreateVideoTask["POST /v1/tasks/video - Create Video Task"]
        TaskRoutes --> CreateImageTask["POST /v1/tasks/image - Create Image Task"]
        TaskRoutes --> CreateVideoTaskForm["POST /v1/tasks/video/form - Form Create Video"]
        TaskRoutes --> CreateImageTaskForm["POST /v1/tasks/image/form - Form Create Image"]
        TaskRoutes --> ListTasks["GET /v1/tasks/ - List Tasks"]
        TaskRoutes --> GetTaskStatus["GET /v1/tasks/{id}/status - Get Status"]
        TaskRoutes --> GetTaskResult["GET /v1/tasks/{id}/result - Get Result"]
        TaskRoutes --> StopTask["DELETE /v1/tasks/{id} - Stop Task"]

        FileRoutes --> DownloadFile["GET /v1/files/download/{path} - Download File"]

        ServiceRoutes --> GetServiceStatus["GET /v1/service/status - Service Status"]
        ServiceRoutes --> GetServiceMetadata["GET /v1/service/metadata - Metadata"]
    end

    subgraph Task Management
        TaskManager[Task Manager]
        TaskQueue[Task Queue]
        TaskStatus[Task Status]
        TaskResult[Task Result]

        CreateVideoTask --> TaskManager
        CreateImageTask --> TaskManager
        TaskManager --> TaskQueue
        TaskManager --> TaskStatus
        TaskManager --> TaskResult
    end

    subgraph File Service
        FileService[File Service]
        DownloadMedia[Download Media]
        SaveFile[Save File]
        GetOutputPath[Get Output Path]

        FileService --> DownloadMedia
        FileService --> SaveFile
        FileService --> GetOutputPath
    end

    subgraph Media Handlers
        MediaHandler[MediaHandler Base]
        ImageHandler[ImageHandler]
        AudioHandler[AudioHandler]

        MediaHandler --> ImageHandler
        MediaHandler --> AudioHandler
    end

    subgraph Processing Thread
        ProcessingThread[Processing Thread]
        NextTask[Get Next Task]
        ProcessTask[Process Single Task]

        ProcessingThread --> NextTask
        ProcessingThread --> ProcessTask
    end

    subgraph Generation Services
        VideoService[VideoGenerationService]
        ImageService[ImageGenerationService]
        BaseService[BaseGenerationService]

        BaseService --> VideoService
        BaseService --> ImageService
    end

    subgraph Distributed Inference Service
        InferenceService[DistributedInferenceService]
        SubmitTask[Submit Task]
        Worker[TorchrunInferenceWorker]
        ProcessRequest[Process Request]
        RunPipeline[Run Inference Pipeline]

        InferenceService --> SubmitTask
        SubmitTask --> Worker
        Worker --> ProcessRequest
        ProcessRequest --> RunPipeline
    end

    TaskQueue --> ProcessingThread
    ProcessTask --> VideoService
    ProcessTask --> ImageService
    VideoService --> InferenceService
    ImageService --> InferenceService
    GetTaskResult --> FileService
    DownloadFile --> FileService
    VideoService --> FileService
    ImageService --> FileService
    FileService --> MediaHandler
```

## Task Processing Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant API as API Server
    participant TM as TaskManager
    participant PT as Processing Thread
    participant GS as GenerationService<br/>(Video/Image)
    participant FS as FileService
    participant DIS as DistributedInferenceService
    participant TIW0 as TorchrunInferenceWorker<br/>(Rank 0)
    participant TIW1 as TorchrunInferenceWorker<br/>(Rank 1..N)

    C->>API: POST /v1/tasks/video<br/>or /v1/tasks/image
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

    PT->>PT: Select service by task type
    PT->>GS: generate_with_stop_event()

    alt Image is URL
        GS->>FS: download_media(url, "image")
        FS->>FS: HTTP download<br/>with retry
        FS-->>GS: image_path
    else Image is Base64
        GS->>GS: save_base64_image()
        GS-->>GS: image_path
    else Image is local path
        GS->>GS: use existing path
    end

    alt Audio is URL (Video only)
        GS->>FS: download_media(url, "audio")
        FS->>FS: HTTP download<br/>with retry
        FS-->>GS: audio_path
    else Audio is Base64
        GS->>GS: save_base64_audio()
        GS-->>GS: audio_path
    end

    GS->>DIS: submit_task_async(task_data)
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

    DIS-->>GS: TaskResponse
    GS-->>PT: TaskResponse

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
    FS-->>API: Video/Image Stream
    API-->>C: Output File
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

## API Endpoints

### Task APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/tasks/video` | POST | Create video generation task |
| `/v1/tasks/video/form` | POST | Create video task with form data |
| `/v1/tasks/image` | POST | Create image generation task |
| `/v1/tasks/image/form` | POST | Create image task with form data |
| `/v1/tasks` | GET | List all tasks |
| `/v1/tasks/queue/status` | GET | Get queue status |
| `/v1/tasks/{task_id}/status` | GET | Get task status |
| `/v1/tasks/{task_id}/result` | GET | Get task result (stream) |
| `/v1/tasks/{task_id}` | DELETE | Cancel task |
| `/v1/tasks/all/running` | DELETE | Cancel all running tasks |

### File APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/files/download/{path}` | GET | Download output file |

### Service APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/service/status` | GET | Get service status |
| `/v1/service/metadata` | GET | Get service metadata |

## Request Models

### VideoTaskRequest

```python
class VideoTaskRequest(BaseTaskRequest):
    num_fragments: int = 1
    target_video_length: int = 81
    audio_path: str = ""
    video_duration: int = 5
    talk_objects: Optional[list[TalkObject]] = None
```

### ImageTaskRequest

```python
class ImageTaskRequest(BaseTaskRequest):
    aspect_ratio: str = "16:9"
```

### BaseTaskRequest (Common Fields)

```python
class BaseTaskRequest(BaseModel):
    task_id: str  # auto-generated
    prompt: str = ""
    use_prompt_enhancer: bool = False
    negative_prompt: str = ""
    image_path: str = ""  # URL, base64, or local path
    save_result_path: str = ""
    infer_steps: int = 5
    seed: int  # auto-generated
```

## Configuration

### Environment Variables

see `lightx2v/server/config.py`

### Command Line Arguments

```bash
# Single GPU
python -m lightx2v.server \
    --model_path /path/to/model \
    --model_cls wan2.1_distill \
    --task i2v \
    --host 0.0.0.0 \
    --port 8000 \
    --config_json /path/to/xxx_config.json
```

```bash
# Multi-GPU with torchrun
torchrun --nproc_per_node=2 -m lightx2v.server \
    --model_path /path/to/model \
    --model_cls wan2.1_distill \
    --task i2v \
    --host 0.0.0.0 \
    --port 8000 \
    --config_json /path/to/xxx_dist_config.json
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
- **Unified media handling** via MediaHandler pattern

### 4. Separate Video/Image Endpoints

- **Dedicated endpoints** for video and image generation
- **Type-specific request models** (VideoTaskRequest, ImageTaskRequest)
- **Automatic service routing** based on task type
- **Backward compatible** with legacy `/v1/tasks` endpoint

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
