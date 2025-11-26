from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from .deps import get_services

router = APIRouter()


def _stream_file_response(file_path: Path, filename: str | None = None) -> StreamingResponse:
    services = get_services()
    assert services.file_service is not None, "File service is not initialized"

    try:
        resolved_path = file_path.resolve()

        if not str(resolved_path).startswith(str(services.file_service.output_video_dir.resolve())):
            raise HTTPException(status_code=403, detail="Access to this file is not allowed")

        if not resolved_path.exists() or not resolved_path.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        file_size = resolved_path.stat().st_size
        actual_filename = filename or resolved_path.name

        mime_type = "application/octet-stream"
        if actual_filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            mime_type = "video/mp4"
        elif actual_filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            mime_type = "image/jpeg"

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
