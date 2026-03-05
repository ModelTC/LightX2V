import asyncio
import base64
import hashlib
import mimetypes
import time
import traceback
from typing import Any

from loguru import logger

from lightx2v.deploy.data_manager import BaseDataManager
from lightx2v.deploy.task_manager import BaseTaskManager, FinishedStatus, TaskStatus

# 静态资源扩展名 -> MIME（供 canvas_fallback 返回正确类型，避免 text/html 导致 JS/CSS 报错）
CANVAS_MEDIA_TYPES = {
    ".html": "text/html",
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".json": "application/json",
    ".css": "text/css",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".webmanifest": "application/manifest+json",
    ".wasm": "application/wasm",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".eot": "application/vnd.ms-fontobject",
    ".otf": "font/otf",
}

# 后缀 -> mime_type，workflow 获取/存储 file 统一使用
WORKFLOW_FILE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".txt": "text/plain",
    ".json": "application/json",
}
# mime_type -> 后缀（用于按 mime_type 拼 storage path）
MIME_TO_EXT = {v: k for k, v in WORKFLOW_FILE_MIME.items()}
# 常见 MIME 别名（如 data URL 里写 image/jpg）
MIME_TO_EXT.setdefault("image/jpg", ".jpg")

# 按 mime_type 取不到文件时尝试的后缀（避免 metadata 里 mime_type 错成 text/plain 导致图片按 .txt 找不到）
WORKFLOW_LOAD_FALLBACK_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mp3", ".wav", ".ogg", ".txt", ".json", ".bin")

MAX_NODE_HISTORY = 20
MAX_CHAT_HISTORY = 50

# =========================
# Workflow Data Management Functions
# =========================


async def fmt_user_name(user_id: str, task_manager: BaseTaskManager) -> str:
    try:
        if not user_id:
            return ""
        user = await task_manager.query_user(user_id)
        return user.get("username", "")
    except Exception:
        logger.error(f"Failed to format user name for user {user_id}: {traceback.format_exc()}")
        return ""


def calc_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def fmt_workflow_file_path(file_entry: dict) -> str | None:
    workflow_id = file_entry["workflow_id"]
    file_id = file_entry["file_id"]
    ext = file_entry["ext"]
    return f"workflows/{workflow_id}/{file_id}{ext}"


def check_invalid_output_value(node):
    if "output_value" not in node:
        return False
    output_value = node["output_value"]
    if isinstance(output_value, str):
        return True
    if isinstance(output_value, dict):
        for value in output_value.values():
            if isinstance(value, str) or (isinstance(value, list) and any(isinstance(item, str) for item in value)):
                return True
            if isinstance(value, dict) and ("user_id" not in value or "workflow_id" not in value or "run_id" not in value):
                return True
    return False


def save_task_entry(
    user_id: str,
    workflow_id: str,
    node_id: str,
    port_id: str,
    run_id: str,
    data: dict,
    files_tasks: dict,
) -> dict:
    task_id = data["task_id"]
    task_entry = {
        "workflow_id": workflow_id,
        "node_id": node_id,
        "port_id": port_id,
        "user_id": user_id,
        "kind": "task",
        "task_id": task_id,
        "output_name": data["output_name"],
        "run_id": run_id,
    }
    if task_id in files_tasks:
        logger.warning(f"Task {task_id} already exists in files_tasks, skip save")
        return task_entry
    files_tasks[task_id] = task_entry
    return task_entry


# only existed file entry can be saved
def save_existed_file_entry(
    data: dict,
    files_tasks: dict,
) -> dict:
    file_id = data["file_id"]
    assert file_id in files_tasks, f"File {file_id} not found in files_tasks"
    file_entry = files_tasks[file_id]
    return file_entry


async def save_bytes_entry(
    user_id: str,
    workflow_id: str,
    node_id: str,
    port_id: str,
    run_id: str,
    output_bytes: bytes,
    mime_type: str,
    files_tasks: dict,
    data_manager: BaseDataManager,
):
    md5 = await asyncio.to_thread(calc_md5, output_bytes)
    # unique file id is related to workflow_id, user_id and md5 of output_bytes
    file_id = f"wf_{workflow_id}_{user_id}_{md5}"
    file_entry = {
        "workflow_id": workflow_id,
        "node_id": node_id,
        "port_id": port_id,
        "user_id": user_id,
        "kind": "file",
        "file_id": file_id,
        "run_id": run_id,
        "mime_type": mime_type,
        "ext": MIME_TO_EXT.get(mime_type, ".bin"),
    }
    if file_id in files_tasks:
        logger.warning(f"File {file_id} already exists in files_tasks, skip save")
        return file_entry

    files_tasks[file_id] = file_entry
    storage_path = fmt_workflow_file_path(file_entry)
    await data_manager.save_bytes(output_bytes, storage_path)
    logger.info(f"Saved file {file_id} to {storage_path}")
    return file_entry


async def save_b64_entry(
    user_id: str,
    workflow_id: str,
    node_id: str,
    port_id: str,
    run_id: str,
    data: Any,
    files_tasks: dict,
    data_manager: BaseDataManager,
) -> dict:
    encoded = data
    mime_type = "application/octet-stream"
    if data.startswith("data:"):
        header, encoded = data.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0].strip()
    output_bytes = await asyncio.to_thread(base64.b64decode, encoded)
    return await save_bytes_entry(user_id, workflow_id, node_id, port_id, run_id, output_bytes, mime_type, files_tasks, data_manager)


async def save_text_entry(
    user_id: str,
    workflow_id: str,
    node_id: str,
    port_id: str,
    run_id: str,
    data: str,
    files_tasks: dict,
    data_manager: BaseDataManager,
):
    output_bytes = data.encode("utf-8")
    mime_type = "text/plain"
    return await save_bytes_entry(user_id, workflow_id, node_id, port_id, run_id, output_bytes, mime_type, files_tasks, data_manager)


async def format_and_save_entry(
    output_data_raw: Any,
    user_id: str,
    run_id: str,
    workflow_id: str,
    node_id: str,
    port_id: str,
    files_tasks: dict,
    data_manager: BaseDataManager,
) -> dict:
    """解析用户上传的节点输出值，存储并返回 entry dict。"""
    item = output_data_raw
    if item["type"] == "task":
        return save_task_entry(user_id, workflow_id, node_id, port_id, run_id, item["data"], files_tasks)
    elif item["type"] == "base64":
        return await save_b64_entry(user_id, workflow_id, node_id, port_id, run_id, item["data"], files_tasks, data_manager)
    elif item["type"] == "text":
        return await save_text_entry(user_id, workflow_id, node_id, port_id, run_id, item["data"], files_tasks, data_manager)
    elif item["type"] == "file":
        return save_existed_file_entry(item["data"], files_tasks)
    else:
        raise ValueError(f"Unknown output data type: {item['type']}")


def update_workflow_node_output(
    workflow: dict,
    node_id: str,
    port_id: str,
    run_id: str,
    new_entry: dict,
) -> dict | list[dict]:
    timestamp_ms = int(time.time() * 1000)
    # 当 workflow 中没有 node_id 时，node 为空字典
    # 因为前端会先上传文件，再更新 node items，此时是找不到 node 记录的
    node = next((n for n in workflow["nodes"] if n["id"] == node_id), None)
    if node is None:
        node = {"id": node_id}
        workflow["nodes"].append(node)
    all_history = workflow.get("node_output_history", {})
    node_history = all_history.get(node_id, [])

    # 同步更新节点的 output_value[port_id]（确保 ext、run_id 等字段直接写入 DB）
    if "output_value" not in node or isinstance(node["output_value"], str):
        node["output_value"] = {}

    # 如果 port_id 已存在，则将 new_entry 添加到 old_entry 列表中
    # 目前仅支持多图的输入，其它 port_id 不支持合并
    if port_id in node["output_value"] and port_id == "out-image" and "file_id" in new_entry:
        old_entry = node["output_value"][port_id]
        old_entries = [old_entry] if not isinstance(old_entry, list) else old_entry
        old_ids = [old_entry["file_id"] for old_entry in old_entries]
        if new_entry["file_id"] in old_ids:
            logger.warning(f"new_entry has already been uploaded, skip: {new_entry}")
            new_entry = old_entry
        else:
            logger.debug(f"append new_entry to old_entries: {old_entries} + {new_entry}")
            old_entries.append(new_entry)
            new_entry = old_entries

    node["output_value"][port_id] = new_entry

    # 同一次 run_id 的多个 port 输出，会依次上传，需要合并到同一个 history item中
    item = next((e for e in node_history if e["id"] == run_id), None)
    if item:
        item["output_value"][port_id] = new_entry
        item["timestamp"] = timestamp_ms
    else:
        node_history.insert(
            0,
            {
                "id": run_id,
                "timestamp": timestamp_ms,
                "execution_time": 0,
                "output_value": {port_id: new_entry},
                "params": node.get("data", {}),
            },
        )

    # 按 timestamp 降序排列，截取前 MAX_NODE_HISTORY
    node_history.sort(key=lambda e: e["timestamp"], reverse=True)
    if len(node_history) > MAX_NODE_HISTORY:
        logger.warning(f"Node {node_id} clean history items: {node_history[MAX_NODE_HISTORY:]}")
    all_history[node_id] = node_history[:MAX_NODE_HISTORY]
    workflow["node_output_history"] = all_history
    return new_entry


def collect_refs(value: Any) -> set[str]:
    """从 value 中提取所有 file_id 和 task_id。"""
    ids = set()
    if not value:
        return ids
    if isinstance(value, dict):
        if "file_id" in value:
            ids.add(value["file_id"])
        elif "task_id" in value:
            ids.add(value["task_id"])
        else:
            for v in value.values():
                if isinstance(v, (dict, list)):
                    ids |= collect_refs(v)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                ids |= collect_refs(item)
    return ids


def collect_file_and_task_refs(nodes: list[dict], node_output_history: dict) -> set[str]:
    """收集 workflow 中所有仍被引用的 file_id 和 task_id（nodes.output_value + node_output_history）。"""
    all_ids = set()
    for node in nodes:
        all_ids |= collect_refs(node.get("output_value", {}))
    for entries in node_output_history.values():
        for entry in entries:
            all_ids |= collect_refs(entry.get("output_value", {}))
    return all_ids


async def delete_file_entry(entry: dict, data_manager: BaseDataManager) -> bool:
    try:
        storage_path = fmt_workflow_file_path(entry)
        await data_manager.delete_bytes(storage_path)
        return True
    except Exception:
        logger.error(f"Failed to delete workflow file {storage_path}: {traceback.format_exc()}")
        return False


async def delete_task_entry(entry: dict, task_manager: BaseTaskManager) -> bool:
    try:
        task_id = entry["task_id"]
        user_id = entry["user_id"]
        task = await task_manager.query_task(task_id, user_id, only_task=True)

        if task["status"] not in FinishedStatus:
            logger.warning(f"Task {task_id} is not finished, stop it and then delete")
            ret = await task_manager.cancel_task(task_id, user_id=user_id)
            if not ret:
                logger.error(f"Failed to cancel task {task_id}: {traceback.format_exc()}")
                return False

        ret = await task_manager.delete_task(task_id, user_id=user_id)
        if not ret:
            logger.error(f"Failed to delete task {task_id}: {traceback.format_exc()}")
            return False
        logger.info(f"Deleted workflow task: {task_id}")
        return True
    except Exception:
        logger.error(f"Failed to delete workflow task {task_id}: {traceback.format_exc()}")
        return False


async def clean_file_or_task_entry(
    user_id: str,
    workflow_id: str,
    entry: dict,
    data_manager: BaseDataManager,
    task_manager: BaseTaskManager,
) -> bool:
    """Clean a single file or task entry from workflow."""

    # for copied workflow, it cannot delete entry of original workflow
    entry_workflow_id = entry["workflow_id"]
    if entry_workflow_id != workflow_id:
        logger.warning(f"entry is not belong to cur workflow, {entry_workflow_id} != {workflow_id}, skip clean")
        return True

    # check the owner of the entry
    entry_user_id = entry["user_id"]
    if entry_user_id != user_id:
        logger.warning(f"entry is not belong to cur user, {entry_user_id} != {user_id}, skip clean")
        return True

    if entry["kind"] == "file":
        return await delete_file_entry(entry, data_manager)
    elif entry["kind"] == "task":
        return await delete_task_entry(entry, task_manager)
    raise ValueError(f"Unknown entry kind: {entry['kind']}")


async def tidy_workflow_files_and_tasks(
    user_id: str,
    workflow_id: str,
    nodes: list[dict],
    node_output_history: dict,
    files_tasks: dict,
    data_manager: BaseDataManager,
    task_manager: BaseTaskManager,
) -> dict:
    """Tidy workflow files and tasks after trimming history."""
    ref_ids = collect_file_and_task_refs(nodes, node_output_history)
    deleted = set()
    for id, entry in files_tasks.items():
        if id not in ref_ids:
            ret = await clean_file_or_task_entry(user_id, workflow_id, entry, data_manager, task_manager)
            if not ret:
                logger.error(f"Failed to clean file or task entry: {id}, {entry}")
                continue
            deleted.add(id)
            logger.warning(f"Deleted file or task entry: {entry}")
    for id in deleted:
        files_tasks.pop(id)


async def query_output_entries(
    user_id: str,
    workflow_id: str,
    node_id: str,
    port_id: str,
    file_id: str,
    task_id: str,
    task_manager: BaseTaskManager,
) -> dict | None:
    workflow = await task_manager.query_workflow(workflow_id, user_id=user_id)
    assert workflow, f"Workflow {workflow_id} not found"

    # 通过 file_id 在 files_tasks 中索引到 entry
    if file_id:
        entry = workflow.get("files_tasks", {}).get(file_id, None)
        if not entry:
            raise Exception(f"File {file_id} not found in files_tasks!")
    elif task_id:
        entry = workflow.get("files_tasks", {}).get(task_id, None)
    else:
        node = next((n for n in workflow["nodes"] if n["id"] == node_id), None)
        if not node or "output_value" not in node or port_id not in node["output_value"]:
            raise Exception(f"Node {node_id}, output {port_id} not found")
        entry = node["output_value"][port_id]
    entries = [entry] if not isinstance(entry, list) else entry

    # for copied workflows, entry_workflow_id != workflow_id, it can only access the public workflow's files
    for e in entries:
        e_worflow_id = e["workflow_id"]
        e_user_id = e["user_id"]
        if e_worflow_id != workflow_id:
            e_workflow = await task_manager.query_workflow(e_worflow_id, user_id=e_user_id)
            assert e_workflow, f"Entry workflow {e_worflow_id} not found"
            assert e_workflow["visibility"] == "public", f"Entry workflow {e_worflow_id} is not public"
    return entries


async def load_bytes_from_entry(
    entry: dict,
    task_manager: BaseTaskManager,
    data_manager: BaseDataManager,
) -> tuple[bytes, str, str] | None:
    if entry["kind"] == "file":
        storage_path = fmt_workflow_file_path(entry)
        assert storage_path, f"File entry {entry} not valid"
        data = await data_manager.load_bytes(storage_path)
        assert data is not None, f"File {storage_path} not valid"
        filename = storage_path.split("/")[-1]
        return data, filename, entry["mime_type"]

    elif entry["kind"] == "task":
        output_name = entry["output_name"]
        task_id = entry["task_id"]
        entry_user_id = entry["user_id"]
        task = await task_manager.query_task(task_id, user_id=entry_user_id)
        assert task is not None, f"Task {task_id} for entry {entry} not found"
        assert task["status"] == TaskStatus.SUCCEED, f"Task {task_id} for entry {entry} not succeed"
        data = await data_manager.load_bytes(task["outputs"][output_name])

        mime_type, _ = mimetypes.guess_type(output_name)
        if mime_type is None:
            mime_type = "application/octet-stream"
        logger.info(f"Load bytes from task entry: {entry}: {len(data)} bytes")
        return data, output_name, mime_type

    else:
        raise ValueError(f"Unknown entry kind: {entry['kind']}")


def to_persisted_message(msg: dict) -> dict:
    """Strip non-persisted fields from a chat message."""
    out = {
        "id": msg.get("id"),
        "role": msg.get("role"),
        "content": msg.get("content", ""),
        "timestamp": msg.get("timestamp", 0),
    }
    for k in ["image", "useSearch", "sources", "error"]:
        if k in msg and msg[k] is not None:
            out[k] = msg[k]
    return out


async def transfer_workflow_output(
    params: dict,
    raw_inputs: list[dict],
    user_id: str,
    task_manager: BaseTaskManager,
    data_manager: BaseDataManager,
) -> dict:
    # get output entry dict info like:
    # {"kind": "file", "file_id": "xxx", "mime_type": "xxx"}
    # {"kind": "task", "task_id": "xxx", "output_name": "xxx"}

    if isinstance(params.get("prompt", ""), list):
        logger.info(f"Transform workflow output: prompt: {params['prompt']}...")
        workflow_id, node_id, port_id = params["prompt"]
        entries = await query_output_entries(user_id, workflow_id, node_id, port_id, "", "", task_manager)
        assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"
        raw_bytes, _, _ = await load_bytes_from_entry(entries[0], task_manager, data_manager)
        params["prompt"] = raw_bytes.decode("utf-8")

    for inp in raw_inputs:
        assert inp in params, f"Input {inp} not found in params"
        if params[inp]["type"] != "workflow_output":
            continue
        old_data = params[inp]["data"]
        logger.info(f"Transform workflow output: {inp}: {old_data}...")
        assert old_data, f"workflow_output data for {inp} is empty"

        # align with multi images input
        # items: [[workflow_id, node_id, port_id], [workflow_id, node_id, port_id], ...]
        items = old_data if isinstance(old_data[0], list) else [old_data]
        bytes_data = []
        for item in items:
            workflow_id, node_id, port_id = item
            entries = await query_output_entries(user_id, workflow_id, node_id, port_id, "", "", task_manager)
            for entry in entries:
                raw_bytes, _, _ = await load_bytes_from_entry(entry, task_manager, data_manager)
                bytes_data.append(raw_bytes)

        new_data = bytes_data[0]
        if len(bytes_data) > 1:
            new_data = {}
            for idx, data in enumerate(bytes_data):
                new_data[f"{inp}_{idx + 1}"] = data
        params[inp]["data"] = new_data

        logger.info(f"Transform workflow output: {inp} done.")
