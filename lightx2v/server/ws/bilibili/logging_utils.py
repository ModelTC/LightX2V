"""Small, single-line previews for live logs; redact credentials before truncating."""

import json


def log_preview(value, config, limit: int = 160) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list, tuple)) else str(value)
    secrets = set()
    for name in ("id_code", "access_key", "access_secret", "llm_api_key", "tts_api_key", "image_api_key"):
        secret = getattr(config, name, "")
        if isinstance(secret, str) and secret:
            secrets.update((secret, json.dumps(secret, ensure_ascii=False)[1:-1]))
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "<redacted>")
    text = "".join(char if char.isprintable() else repr(char)[1:-1] for char in text)
    if limit <= 0:
        return ""
    return text if len(text) <= limit else text[: limit - 1] + "…"
