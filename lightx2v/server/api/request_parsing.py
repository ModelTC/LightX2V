"""Decode form requests and document them from their JSON schemas."""

import json
from functools import lru_cache
from typing import get_args, get_origin

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError


def _contains_json_value(annotation):
    if get_origin(annotation) in (list, dict, tuple) or isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    return any(_contains_json_value(arg) for arg in get_args(annotation))


@lru_cache
def _json_form_fields(request_cls):
    return {field.alias or name for name, field in request_cls.model_fields.items() if _contains_json_value(field.annotation)}


def parse_form_request(request_cls, request_data):
    """Decode JSON containers, then validate submitted fields with the request schema."""
    request_data = dict(request_data)
    json_fields = _json_form_fields(request_cls)
    for name, value in request_data.items():
        if name in json_fields and isinstance(value, str) and value.lstrip().startswith(("[", "{")):
            try:
                request_data[name] = json.loads(value)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=f"{name} must contain valid JSON") from exc
    try:
        return request_cls.model_validate(request_data)
    except ValidationError as exc:
        errors = exc.errors(include_input=False)
        for error in errors:
            error["loc"] = ("body", *error["loc"])
        raise RequestValidationError(errors) from exc


def _inline_schema_refs(value, definitions):
    if isinstance(value, list):
        return [_inline_schema_refs(item, definitions) for item in value]
    if isinstance(value, dict):
        if "$ref" in value:
            value = definitions[value["$ref"].rsplit("/", 1)[-1]] | {key: item for key, item in value.items() if key != "$ref"}
        return {key: _inline_schema_refs(item, definitions) for key, item in value.items()}
    return value


def form_request_openapi(request_cls, **upload_fields):
    """Describe raw form fields without duplicating scalar endpoint parameters."""
    schema = request_cls.model_json_schema()
    for name in _json_form_fields(request_cls):
        field_schema = schema["properties"][name]
        schema["properties"][name] = {
            "type": "string",
            "title": field_schema.get("title", name),
            "description": (field_schema.get("description", "") + " Submit lists and objects as JSON text.").strip(),
        }
    # Inline remaining references (e.g. enums): this schema is embedded in an
    # OpenAPI operation, where Pydantic's root-relative #/$defs paths do not apply.
    schema = _inline_schema_refs(schema, schema.pop("$defs", {}))
    schema["properties"].update(upload_fields)
    return {"requestBody": {"required": bool(schema.get("required")), "content": {"multipart/form-data": {"schema": schema}}}}
