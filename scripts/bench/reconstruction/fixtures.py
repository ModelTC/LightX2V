import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_requests(path):
    requests = []
    for entry in json.loads(Path(path).read_text()):
        if "repeats" not in entry:
            requests.append(entry)
            continue
        request = {key: value for key, value in entry.items() if key != "repeats"}
        for repeat in range(entry["repeats"]):
            requests.append({**request, "id": f"{entry['id']}_r{repeat}"})
    return requests


def unique_images(requests):
    return {image["path"]: image for request in requests for image in (request["images"] if "images" in request else [request["image"]])}
