import json
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from PIL import Image
from loguru import logger

from lightx2v.models.networks.bagel.sensenova_tasks import (
    OMNI_VISION_SUBTASK_ALIASES,
    OMNI_VISION_TASK_SPECS,
    TEXT_OUTPUT_MODES,
    OmniVisionTaskSpec,
    normalize_omni_vision_subtask,
)
from lightx2v.models.runners.bagel.sensenova_postprocess import (
    load_official_example_constant,
    load_official_visualizers,
)

from ...schema import (
    SenseNovaArtifact,
    SenseNovaVisionGenerationResponse,
    SenseNovaVisionTaskRequest,
    SenseNovaVisionTaskResult,
)
from ..file_service import FileService
from ..inference import DistributedInferenceService
from .base import BaseGenerationService

SenseNovaTaskSpec = OmniVisionTaskSpec
SENSENOVA_TASK_SPECS = OMNI_VISION_TASK_SPECS
SENSENOVA_TASK_ALIASES = OMNI_VISION_SUBTASK_ALIASES

DEFAULT_UNDERSTANDING_PROMPT = "What are the main objects in this scene and their relationships?"
VGD_REFERENCE_PATTERN = re.compile(
    r"<p>\s*(?P<label>[^<>]+?)\s*</p>\s*"
    r"<bbox>\s*(?P<bbox>\[[^\[\]]+\])\s*</bbox>",
    flags=re.IGNORECASE | re.DOTALL,
)
VGD_COLOR_LABEL_PATTERN = re.compile(
    r"<p>\s*[^<>]+?\s*<color>\s*"
    r"\(\s*(?P<red>\d+)\s*,\s*(?P<green>\d+)\s*,\s*(?P<blue>\d+)\s*\)"
    r"\s*</color>\s*</p>",
    flags=re.IGNORECASE | re.DOTALL,
)


def build_official_vgd_prompt(prompt: str) -> str:
    """Convert one visual grounding reference into the official task-10 prompt."""
    match = VGD_REFERENCE_PATTERN.search(str(prompt or "").strip())
    if match is None:
        raise ValueError("SenseNova-Vision task='vgd_segmentation' requires one visual reference in the exact form <p>label</p><bbox>[x1, y1, x2, y2]</bbox>.")

    label = " ".join(match.group("label").split())
    bbox = match.group("bbox")
    try:
        coordinates = [float(value.strip()) for value in bbox[1:-1].split(",")]
    except ValueError as exc:
        raise ValueError("SenseNova-Vision VGD bbox coordinates must be numeric.") from exc
    if len(coordinates) != 4:
        raise ValueError("SenseNova-Vision VGD bbox must contain exactly four coordinates.")
    if any(value < 0.0 or value > 1.0 for value in coordinates):
        raise ValueError("SenseNova-Vision VGD bbox coordinates must be normalized to [0, 1].")
    if coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]:
        raise ValueError("SenseNova-Vision VGD bbox must satisfy x1 < x2 and y1 < y2.")

    label_tag = f"<p>{label}</p>"
    reference = f"{label_tag}<bbox>{bbox}</bbox>"
    colored_label = f"<p>{label}<color>(R,G,B)</color></p>"
    return (
        "<image> Identify all objects belonging to the same classes as the visually provided "
        f"{reference}. Generate an instance segmentation visualization and each identified category "
        f"{label_tag} is colored different. First, enumerate each visible {label_tag} instance "
        f"mentioned in the request and assign each {label_tag} a different color. Reformat them in "
        f"the EXACT format: {colored_label}. Then respond with interleaved instance segmentation "
        "masks using those instance labels and colors."
    )


def validate_vgd_output_text(text: str) -> None:
    """Reject VGD outputs that cannot condition or visualize an instance-color mask."""
    matches = list(VGD_COLOR_LABEL_PATTERN.finditer(str(text or "")))
    if not matches:
        raise RuntimeError(
            "SenseNova-Vision vgd_segmentation AR output did not follow the official colored-instance format <p>label<color>(R,G,B)</color></p>; refusing to report an invalid mask as completed."
        )
    for match in matches:
        rgb = tuple(int(match.group(channel)) for channel in ("red", "green", "blue"))
        if any(value > 255 for value in rgb):
            raise RuntimeError(f"SenseNova-Vision vgd_segmentation AR output contains invalid RGB color {rgb}.")


def normalize_sensenova_task(task: str) -> str:
    return normalize_omni_vision_subtask(task)


def validate_sensenova_request(message: SenseNovaVisionTaskRequest) -> tuple[str, SenseNovaTaskSpec, str]:
    task = normalize_sensenova_task(message.task)
    spec = SENSENOVA_TASK_SPECS[task]
    image_count = len(message.images)
    if image_count < spec.min_images or image_count > spec.max_images:
        expected = str(spec.min_images) if spec.min_images == spec.max_images else f"{spec.min_images}-{spec.max_images}"
        raise ValueError(f"SenseNova-Vision task={task!r} requires {expected} input image(s), got {image_count}.")
    if spec.requires_prompt and not str(message.prompt or "").strip():
        raise ValueError(f"SenseNova-Vision task={task!r} requires a non-empty prompt.")
    if message.target_shape:
        if len(message.target_shape) != 2 or any(int(value) <= 0 for value in message.target_shape):
            raise ValueError("SenseNova-Vision target_shape must be [height, width] with two positive integers.")
    if message.postprocess_3d and task != "recon3d":
        raise ValueError("postprocess_3d is only valid for task='recon3d'.")
    mode = spec.mode
    return task, spec, mode


class SenseNovaVisionGenerationService(BaseGenerationService):
    def __init__(self, file_service: FileService, inference_service: DistributedInferenceService):
        super().__init__(file_service, inference_service)

    def get_output_extension(self) -> str:
        return ".json"

    def get_task_type(self) -> str:
        return "sensenova_vision"

    def ensure_compatible_runner(self) -> None:
        worker = self.inference_service.worker
        runner = worker.runner if worker else None
        model_cls = runner.config.get("model_cls") if runner is not None else None
        task = runner.config.get("task") if runner is not None else None
        if model_cls != "sensenova_vision":
            raise RuntimeError("The SenseNova-Vision API requires a server started with --model_cls sensenova_vision.")

        if task != "omni_vision_task":
            raise RuntimeError("The SenseNova-Vision API requires a server started with --task omni_vision_task.")

    async def _resolve_images(self, image_sources: list[str]) -> list[str]:
        resolved = []
        for index, image_source in enumerate(image_sources):
            if not str(image_source or "").strip():
                raise ValueError(f"SenseNova-Vision image at index {index} is empty.")
            resolved.append(await self._resolve_image_path(image_source))
        return resolved

    def _artifact(self, path: Path, kind: str, media_type: str) -> SenseNovaArtifact:
        resolved_path = path.resolve()
        output_root = self.file_service.output_video_dir.resolve()
        try:
            relative = resolved_path.relative_to(output_root)
        except ValueError as exc:
            raise RuntimeError(f"SenseNova artifact is outside the server output directory: {resolved_path}") from exc
        if not resolved_path.is_file():
            raise FileNotFoundError(f"SenseNova artifact was not created: {resolved_path}")
        relative_url = quote(relative.as_posix(), safe="/")
        return SenseNovaArtifact(
            kind=kind,
            media_type=media_type,
            filename=relative.as_posix(),
            url=f"/v1/files/download/{relative_url}",
            size_bytes=resolved_path.stat().st_size,
        )

    @staticmethod
    def _segmentation_label(prompt: str) -> str:
        match = re.search(r"<p>(.*?)</p>", str(prompt or ""), flags=re.DOTALL)
        if match:
            return " ".join(match.group(1).split())
        return "segmentation"

    def _create_visualization(
        self,
        task: str,
        spec: SenseNovaTaskSpec,
        image_paths: list[str],
        raw_image_path: Optional[Path],
        text: str,
        prompt: str,
        output_path: Path,
    ) -> Optional[Path]:
        if spec.visualizer is None:
            return None

        source_path = self.inference_service.worker.runner.config.get(
            "sensenova_source_path",
            "/data/nvme0/lhd_codes/sensenova-vision-v2",
        )
        visualizers = load_official_visualizers(source_path)
        config = visualizers.VisualizationConfig()
        with Image.open(image_paths[0]) as source_image:
            source = source_image.convert("RGB").copy()

        if spec.visualizer == "detection":
            task_name = {
                "object_detection": "common_object_detection",
                "point_detection": "point_detection",
                "keypoint": "keypoint",
                "ocr": "ocr",
            }[task]
            pred = visualizers.visualize_detection(
                source,
                text,
                task_name=task_name,
                prompt=prompt,
                config=config,
            )
            visualization = visualizers.visualize_concat_col(source, pred, concat_col=2)
        else:
            if raw_image_path is None:
                raise RuntimeError(f"SenseNova task={task!r} did not produce the raw image required for visualization.")
            with Image.open(raw_image_path) as raw_image:
                prediction = raw_image.convert("RGB").copy()
            if spec.visualizer == "binary":
                pred = visualizers.visualize_binary_segmentation(
                    source,
                    prediction,
                    label=self._segmentation_label(prompt),
                    config=config,
                )
                visualization = visualizers.visualize_concat_col(source, pred, concat_col=2)
            elif spec.visualizer == "interactive":
                with Image.open(image_paths[1]) as prompt_image:
                    prompt_panel = visualizers.draw_visual_prompt(
                        source,
                        prompt_image.convert("L"),
                        prompt_style="boundary",
                    )
                pred = visualizers.visualize_binary_segmentation(
                    source,
                    prediction,
                    label="box prompt",
                    config=config,
                )
                visualization = visualizers.visualize_concat_col(
                    source,
                    pred,
                    concat_col=3,
                    prompt=prompt_panel,
                )
            elif spec.visualizer == "gcg":
                pred = visualizers.visualize_gcg_segmentation(source, prediction, text, config=config)
                visualization = visualizers.visualize_concat_col(source, pred, concat_col=2)
            elif spec.visualizer == "panoptic":
                pred = visualizers.visualize_panoptic_segmentation(
                    source,
                    prediction,
                    text,
                    question=prompt,
                    config=config,
                )
                visualization = visualizers.visualize_concat_col(source, pred, concat_col=2)
            else:
                raise ValueError(f"Unsupported SenseNova visualizer: {spec.visualizer}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        visualization.save(output_path)
        return output_path

    async def generate_with_stop_event(self, message: Any, stop_event) -> Optional[Any]:
        if not isinstance(message, SenseNovaVisionTaskRequest):
            raise TypeError(f"Expected SenseNovaVisionTaskRequest, got {type(message)!r}")
        self.ensure_compatible_runner()
        task, spec, mode = validate_sensenova_request(message)
        if stop_event.is_set():
            logger.info(f"SenseNova task {message.task_id} cancelled before processing")
            return None

        image_paths = await self._resolve_images(message.images)
        output_dir = self.file_service.output_video_dir / message.task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        prompt = str(message.prompt or "").strip()
        if task == "understanding" and not prompt:
            prompt = DEFAULT_UNDERSTANDING_PROMPT
        elif task == "panoptic_segmentation" and not prompt:
            source_path = self.inference_service.worker.runner.config.get(
                "sensenova_source_path",
                "/data/nvme0/lhd_codes/sensenova-vision-v2",
            )
            prompt = load_official_example_constant(source_path, "EXAMPLE_08_PANOPTIC_QUESTION")
        elif task == "vgd_segmentation":
            prompt = build_official_vgd_prompt(prompt)

        is_recon3d = spec.runner_task == "recon3d"
        is_text_only = mode in TEXT_OUTPUT_MODES
        if is_recon3d:
            save_result_path = output_dir / "pointmaps.npy"
        elif is_text_only:
            save_result_path = output_dir / "raw.txt"
        else:
            save_result_path = output_dir / "raw.png"

        raw_output_path = output_dir / "pointmaps.npy" if is_recon3d else None
        glb_output_path = output_dir / "scene.glb" if is_recon3d and message.postprocess_3d else None
        task_data = {
            "task_id": message.task_id,
            "prompt": prompt,
            "image_path": ",".join(image_paths),
            "seed": int(message.seed),
            "target_shape": list(message.target_shape),
            "save_result_path": str(save_result_path),
            "omni_vision_subtask": task,
            "raw_output_path": str(raw_output_path) if raw_output_path else "",
            "glb_output_path": str(glb_output_path) if glb_output_path else "",
            "postprocess_predictions": bool(message.postprocess_3d),
            "return_result_tensor": False,
            "_return_pipeline_result": True,
        }

        inference_result = await self.inference_service.submit_task_async(task_data)
        if inference_result is None:
            if stop_event.is_set():
                return None
            raise RuntimeError("SenseNova-Vision inference returned no result.")
        if inference_result.get("status") != "success":
            error = inference_result.get("error", "SenseNova-Vision inference failed")
            error_type = inference_result.get("error_type", "")
            exc = RuntimeError(error)
            exc.original_error_type = error_type
            raise exc

        pipeline_result = inference_result.get("pipeline_return")
        if not isinstance(pipeline_result, dict):
            raise RuntimeError("SenseNova-Vision runner did not return its structured pipeline result.")
        text = str(pipeline_result.get("text") or "")
        if task == "vgd_segmentation":
            validate_vgd_output_text(text)
        artifacts: list[SenseNovaArtifact] = []
        warnings: list[str] = []

        raw_image_paths = []
        direct_raw_image = output_dir / "raw.png"
        if direct_raw_image.is_file():
            raw_image_paths.append(direct_raw_image)
        else:
            raw_image_paths.extend(sorted(output_dir.glob("raw_[0-9][0-9][0-9][0-9][0-9].png")))
        for index, path in enumerate(raw_image_paths):
            kind = "raw_image" if len(raw_image_paths) == 1 else f"raw_image_{index}"
            artifacts.append(self._artifact(path, kind, "image/png"))

        if text:
            text_path = output_dir / ("raw.txt" if is_text_only else "raw_text.txt")
            if not text_path.is_file():
                text_path.write_text(text, encoding="utf-8")
            artifacts.append(self._artifact(text_path, "text", "text/plain; charset=utf-8"))

        pose_path = output_dir / "raw_pose.json"
        if pose_path.is_file():
            artifacts.append(self._artifact(pose_path, "pose_json", "application/json"))

        if raw_output_path is not None and raw_output_path.is_file():
            artifacts.append(self._artifact(raw_output_path, "pointmaps", "application/octet-stream"))
        if glb_output_path is not None and glb_output_path.is_file():
            artifacts.append(self._artifact(glb_output_path, "scene_glb", "model/gltf-binary"))

        if message.visualize and spec.visualizer is not None:
            try:
                visualization_path = self._create_visualization(
                    task=task,
                    spec=spec,
                    image_paths=image_paths,
                    raw_image_path=raw_image_paths[0] if raw_image_paths else None,
                    text=text,
                    prompt=prompt,
                    output_path=output_dir / "visualization.png",
                )
                if visualization_path is not None:
                    artifacts.append(self._artifact(visualization_path, "visualization", "image/png"))
            except Exception as exc:
                warning = f"Official visualization failed; raw inference artifacts are still available: {exc}"
                logger.exception(warning)
                warnings.append(warning)

        result_model = SenseNovaVisionTaskResult(
            task_id=message.task_id,
            status="completed",
            task=task,
            runner_task=spec.runner_task,
            mode=mode,
            text=text or None,
            artifacts=artifacts,
            warnings=warnings,
        )
        result_data = result_model.model_dump()
        manifest_path = output_dir / "result.json"
        manifest_path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_relative = manifest_path.relative_to(self.file_service.output_video_dir).as_posix()
        return SenseNovaVisionGenerationResponse(
            task_id=message.task_id,
            task_status="completed",
            save_result_path=manifest_relative,
            result_data=result_data,
        )
