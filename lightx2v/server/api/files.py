from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from .deps import get_services

router = APIRouter()


def _get_mime_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".txt": "text/plain; charset=utf-8",
        ".json": "application/json",
        ".npy": "application/octet-stream",
        ".glb": "model/gltf-binary",
    }.get(suffix, "application/octet-stream")


def _stream_file_response(file_path: Path, filename: str | None = None) -> StreamingResponse:
    services = get_services()
    assert services.file_service is not None, "File service is not initialized"

    try:
        resolved_path = file_path.resolve()

        output_root = services.file_service.output_video_dir.resolve()
        try:
            resolved_path.relative_to(output_root)
        except ValueError:
            raise HTTPException(status_code=403, detail="Access to this file is not allowed")

        if not resolved_path.exists() or not resolved_path.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        file_size = resolved_path.stat().st_size
        actual_filename = filename or resolved_path.name

        mime_type = _get_mime_type(actual_filename)

        headers = {
            "Content-Disposition": f'attachment; filename="{actual_filename}"',
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
        }

        def file_stream_generator(file_path: str, chunk_size: int = 1024 * 1024):
            with open(file_path, "rb") as file:
                while chunk := file.read(chunk_size):
                    yield chunk

        return StreamingResponse(
            file_stream_generator(str(resolved_path)),
            media_type=mime_type,
            headers=headers,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error occurred while processing file stream response: {e}")
        raise HTTPException(status_code=500, detail="File transfer failed")


@router.get("/download/{file_path:path}")
async def download_file(file_path: str):
    services = get_services()
    assert services.file_service is not None, "File service is not initialized"

    try:
        full_path = services.file_service.output_video_dir / file_path
        return _stream_file_response(full_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error occurred while processing file download request: {e}")
        raise HTTPException(status_code=500, detail="File download failed")
