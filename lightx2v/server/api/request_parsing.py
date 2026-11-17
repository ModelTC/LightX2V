"""Decode form requests and document them from their JSON schemas."""

import asyncio
import json
from functools import lru_cache
from typing import Annotated, get_args, get_origin

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, TypeAdapter, ValidationError
from starlette.datastructures import UploadFile

from lightx2v.utils.request_spec import is_list_type


def _contains_json_value(annotation):
    if annotation in (list, dict, tuple) or get_origin(annotation) in (list, dict, tuple) or isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    return any(_contains_json_value(arg) for arg in get_args(annotation))


@lru_cache
def _form_fields(request_type, task=None):
    if get_origin(request_type) is Annotated:
        request_type = get_args(request_type)[0]
    models = get_args(request_type) or (request_type,)
    return {
        field.alias or name: field
        for model in models
        if task is None or "task" not in model.model_fields or task in get_args(model.model_fields["task"].annotation)
        for name, field in model.model_fields.items()
    }


def _json_form_fields(request_type, task=None):
    return {name for name, field in _form_fields(request_type, task).items() if _contains_json_value(field.annotation)}


def validate_request(request_type, request_data, *, strict=None):
    try:
        return TypeAdapter(request_type).validate_python(request_data, strict=strict)
    except ValidationError as exc:
        errors = exc.errors(include_input=False)
        for error in errors:
            error["loc"] = ("body", *error["loc"])
        raise RequestValidationError(errors) from exc


def parse_form_request(request_cls, request_data):
    """Decode JSON containers, then validate submitted fields with the request schema."""
    request_data = dict(request_data)
    json_fields = _json_form_fields(request_cls, request_data.get("task"))
    for name, value in request_data.items():
        if name in json_fields and isinstance(value, str) and value.lstrip().startswith(("[", "{")):
            try:
                request_data[name] = json.loads(value)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=f"{name} must contain valid JSON") from exc
    # Form scalar values arrive as strings, including strict numeric fields.
    return validate_request(request_cls, request_data, strict=False)


async def parse_upload_form(request_cls, form, file_service, uploads):
    """Save uploaded files into their request path fields, then validate the form."""
    data = dict(form)
    fields = _form_fields(request_cls, data.get("task"))
    for name, field in uploads.items():
        data.pop(name, None)
        submitted = form.getlist(name)
        if any(upload and not isinstance(upload, UploadFile) for upload in submitted):
            raise HTTPException(status_code=422, detail=f"{name} must be a file")
        files = [upload for upload in submitted if isinstance(upload, UploadFile) and upload.filename]
        if not files:
            continue
        if field not in fields:
            raise HTTPException(status_code=422, detail=f"{name} is not supported for this task")
        multiple = is_list_type(fields[field].annotation)
        if not multiple and len(files) > 1:
            raise HTTPException(status_code=422, detail=f"{name} accepts only one file")
        paths = []
        target_dir = file_service.input_audio_dir if field == "audio_path" else None
        for upload in files:
            content = await upload.read()
            path = await asyncio.to_thread(file_service.save_uploaded_file, content, upload.filename, target_dir)
            paths.append(str(path))
        data[field] = paths if multiple else paths[0]
    return parse_form_request(request_cls, data)


def _inline_schema_refs(value, definitions):
    if isinstance(value, list):
        return [_inline_schema_refs(item, definitions) for item in value]
    if isinstance(value, dict):
        if "$ref" in value:
            value = definitions[value["$ref"].rsplit("/", 1)[-1]] | {key: item for key, item in value.items() if key != "$ref"}
        return {key: _inline_schema_refs(item, definitions) for key, item in value.items()}
    return value


def form_request_openapi(request_cls, *, uploads=None, **extra_fields):
    """Describe raw form fields without duplicating scalar endpoint parameters."""
    schema = TypeAdapter(request_cls).json_schema()
    schema = _inline_schema_refs(schema, schema.pop("$defs", {}))
    # Form containers travel as JSON text; task variants keep their own fields.
    variants = schema.get("oneOf", [schema])
    for variant in variants:
        task = variant["properties"].get("task", {}).get("const")
        fields = _form_fields(request_cls, task)
        upload_fields = {}
        for name, field in (uploads or {}).items():
            if field in variant["properties"]:
                file_schema = {"type": "string", "format": "binary"}
                upload_fields[name] = {"type": "array", "items": file_schema} if is_list_type(fields[field].annotation) else file_schema
        for name in _json_form_fields(request_cls):
            if name not in variant["properties"]:
                continue
            field_schema = variant["properties"][name]
            variant["properties"][name] = {
                "type": "string",
                "title": field_schema.get("title", name),
                "description": (field_schema.get("description", "") + " Submit lists and objects as JSON text.").strip(),
            }
        variant["properties"].update(extra_fields)
        variant["properties"].update(upload_fields)
    schema.pop("discriminator", None)
    return {"requestBody": {"required": any(variant.get("required") for variant in variants), "content": {"multipart/form-data": {"schema": schema}}}}
