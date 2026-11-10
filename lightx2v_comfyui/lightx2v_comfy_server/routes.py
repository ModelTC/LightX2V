"""ComfyUI endpoints for files, configuration and managed services."""

import asyncio
import re
from functools import wraps

from .files import ConfigConflict, config_path, directory_listing, directory_path, local_path, lora_path, read_config, save_config
from .gpus import gpu_status
from .models import model_catalog
from .nodes import manager
from .service import resolve_server


def register_routes():
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes

    def endpoint(method, path, *, runtime_status=409):
        def register(handler):
            @wraps(handler)
            async def respond(request):
                try:
                    data = await handler(request)
                    return web.json_response(data, headers={"Cache-Control": "no-store"})
                except PermissionError as exc:
                    status, error = 403, exc
                except ConfigConflict as exc:
                    status, error = 409, exc
                except (OSError, SyntaxError, ValueError) as exc:
                    status, error = 400, exc
                except RuntimeError as exc:
                    status, error = runtime_status, exc
                return web.json_response({"error": str(error)}, status=status, headers={"Cache-Control": "no-store"})

            routes.route(method, "/lightx2v" + path)(respond)
            return handler

        return register

    @endpoint("GET", "/gpus", runtime_status=503)
    async def list_gpus(request):
        return await asyncio.to_thread(gpu_status)

    @endpoint("GET", "/services")
    async def list_services(request):
        runtime = manager()
        return {"instances": await asyncio.to_thread(runtime.snapshots), "same_gpu_policy": runtime.same_gpu_policy}

    @endpoint("POST", "/services/release")
    async def release_service(request):
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Expected a JSON object")  # noqa: TRY004 - invalid input value
        key = body.get("service_key")
        if key is None and isinstance(body.get("server"), dict):
            settings = await asyncio.to_thread(resolve_server, body["server"])
            key = settings.service_key
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("A service_key or valid server descriptor is required")
        return await asyncio.to_thread(manager().release, key)

    @endpoint("GET", "/models")
    async def list_models(request):
        return await asyncio.to_thread(model_catalog, request.query.get("path", ""))

    @endpoint("GET", "/files")
    async def list_files(request):
        base = local_path(request.query["base"]) if request.query.get("base") else None
        return await asyncio.to_thread(
            directory_listing, request.query.get("path", ""), base, directories_only=request.query.get("directories_only") == "1", file_type=request.query.get("file_type", "config")
        )

    @endpoint("GET", "/directories/resolve")
    async def resolve_directory(request):
        path = await asyncio.to_thread(directory_path, request.query.get("path", ""))
        return {"directory": str(path)}

    @endpoint("GET", "/configs/resolve")
    async def resolve_config(request):
        path = await asyncio.to_thread(config_path, request.query.get("path", ""))
        return {"config_json": str(path)}

    @endpoint("GET", "/loras/resolve")
    async def resolve_lora(request):
        base = local_path(request.query["base"]) if request.query.get("base") else None
        path = await asyncio.to_thread(lora_path, request.query.get("path", ""), base)
        return {"lora_path": str(path)}

    @endpoint("GET", "/configs/content")
    async def config_content(request):
        data = await asyncio.to_thread(read_config, request.query.get("path", ""), request.query.get("base"))
        key = request.query.get("service_key")
        if key and not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Invalid service_key")
        data["loaded"] = await asyncio.to_thread(manager().config_snapshot, key) if key else None
        return data

    @endpoint("POST", "/configs/save")
    async def config_save(request):
        body = await request.json()
        if not isinstance(body, dict) or type(body.get("create_new", False)) is not bool:
            raise ValueError("Expected a JSON object with boolean create_new")
        return await asyncio.to_thread(save_config, body.get("path", ""), body.get("content"), body.get("expected_sha256"), create_new=body.get("create_new", False), base=body.get("base"))
