import io
import os
import traceback
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image, ImageDraw
from loguru import logger
from ultralytics import YOLO

# Try to import transformers for Grounding DINO
try:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available, Grounding DINO method will not work")


class FaceDetector:
    """
    Face detection using multiple methods

    Supports three detection methods:

    1. YOLO World (method='yolo'):
       - Open-vocabulary detection
       - Supports various face types: human, animal, anime, sketch
       - More flexible but slower
       - Can detect custom classes via text description

    2. Grounding DINO (method='grounding'):
       - Open-vocabulary object detection
       - Supports various face types via text prompts
       - Requires transformers library
       - Good balance between accuracy and flexibility
    """

    def __init__(
        self,
        method: str = "yolo",
        model_path: str = None,
        conf_threshold: float = None,
        device: str = None,
        custom_classes: List[str] = None,
        cascade_path: str = None,
    ):
        """
        Initialize face detector

        Args:
            method: Detection method. Options:
                - "yolo": Use YOLO World (supports various face types)
                - "grounding": Use Grounding DINO (requires transformers)
                Default: "yolo"
            model_path: YOLO World model path (only used when method="yolo")
                If None, uses default YOLO World model
            conf_threshold: Confidence threshold (only used when method="yolo")
                If None, uses adaptive threshold based on classes
            device: Device for YOLO ('cpu', 'cuda', '0', '1', etc.), None for auto
            custom_classes: List of custom class names for YOLO World. Default: ["face"]
                Examples: ["face"], ["animal face"], ["human face", "animal face"]
        """

        self.method = method.lower()
        self.device = device

        if self.method == "yolo":
            # Initialize YOLO World detector
            # Set custom classes (default to "face")
            if custom_classes is None:
                custom_classes = ["human face", "animal face", "anime face", "sketch face"]
            self.custom_classes = custom_classes

            # Adaptive confidence threshold based on class specificity
            if conf_threshold is None:
                if len(custom_classes) > 1:
                    # Multiple classes: use lower threshold to catch all detections
                    conf_threshold = 0.1
                elif len(custom_classes) == 1:
                    class_name = custom_classes[0].lower()
                    if "face" in class_name and class_name.strip() == "face":
                        # Generic "face" class: needs higher threshold but not too high
                        conf_threshold = 0.15
                    else:
                        # Specific class like "animal face": can use moderate threshold
                        conf_threshold = 0.15
                else:
                    conf_threshold = 0.25
            self.conf_threshold = conf_threshold

            if model_path is None:
                # Use YOLO World model for open-vocabulary detection
                logger.info("Loading YOLO World model for face detection")
                try:
                    # Try to load YOLO World small model first (lighter and faster)
                    self.model = YOLO("yolov8s-world.pt")
                except Exception as e:
                    logger.warning(f"Failed to load yolov8s-world.pt, trying yolov8m-world.pt: {e}")
                    try:
                        self.model = YOLO("yolov8m-world.pt")
                    except Exception as e2:
                        logger.warning(f"Failed to load yolov8m-world.pt, trying yolov8l-world.pt: {e2}")
                        self.model = YOLO("yolov8l-world.pt")
                # Set custom classes for YOLO World
                # YOLO World can detect any object described in natural language
                self.model.set_classes(self.custom_classes)
            else:
                logger.info(f"Loading YOLO World model from {model_path}")
                self.model = YOLO(model_path)

            logger.info(f"Face detector initialized with YOLO World, custom classes: {self.custom_classes}, confidence threshold: {self.conf_threshold}")
            self.face_cascade = None

        elif self.method == "grounding":
            # Initialize Grounding DINO detector
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers library is required for Grounding DINO. Install it with: pip install transformers torch")

            # Set up proxy for HuggingFace model download
            # Check if proxy is already set, if not try to use common proxy settings
            if not os.environ.get("HTTP_PROXY") and not os.environ.get("http_proxy"):
                # Try to use HTTPS_PROXY for HTTP requests as well if available
                https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
                if https_proxy:
                    os.environ["HTTP_PROXY"] = https_proxy
                    os.environ["http_proxy"] = https_proxy
                    logger.info(f"Using proxy from HTTPS_PROXY: {https_proxy}")

            # Log proxy settings
            http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
            https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
            if http_proxy or https_proxy:
                logger.info(f"Using proxy - HTTP: {http_proxy}, HTTPS: {https_proxy}")

            # Set custom classes (default to "face")
            if custom_classes is None:
                custom_classes = ["human face", "animal face", "anime face", "sketch face"]
            self.custom_classes = custom_classes

            # Adaptive confidence threshold
            if conf_threshold is None:
                if len(custom_classes) > 1:
                    conf_threshold = 0.1
                else:
                    conf_threshold = 0.3  # Grounding DINO typically needs higher threshold
            self.conf_threshold = conf_threshold

            # Load Grounding DINO model
            model_id = "IDEA-Research/grounding-dino-base"  # or "grounding-dino-tiny" for faster inference
            if model_path is not None:
                model_id = model_path
            logger.info(f"Loading Grounding DINO model: {model_id}")
            try:
                # Grounding DINO requires trust_remote_code=True
                self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, trust_remote_code=True)
                if device:
                    self.model = self.model.to(device)
                logger.info(f"Face detector initialized with Grounding DINO, custom classes: {self.custom_classes}, confidence threshold: {self.conf_threshold}")
            except Exception as e:
                error_msg = str(e)
                if "connection" in error_msg.lower() or "proxy" in error_msg.lower() or "network" in error_msg.lower():
                    logger.error(f"Failed to download model. Please check your network connection and proxy settings.")
                    logger.error(f"Current proxy settings - HTTP_PROXY: {http_proxy}, HTTPS_PROXY: {https_proxy}")
                    logger.error("You can set proxy with: export http_proxy=... && export https_proxy=...")
                raise
            self.face_cascade = None

        else:
            raise ValueError(f"Unknown method: {method}. Must be 'yolo', or 'grounding'")

    def detect_faces(
        self,
        image: Union[str, Image.Image, bytes, np.ndarray],
        return_image: bool = False,
    ) -> Dict:
        """
        Detect faces in image

        Args:
            image: Input image, can be path, PIL Image, bytes or numpy array
            return_image: Whether to return annotated image with detection boxes

        Returns:
            Dict containing:
                - faces: List of face detection results, each containing:
                    - bbox: [x1, y1, x2, y2] bounding box coordinates (absolute pixel coordinates)
                    - confidence: Confidence score (0.0-1.0)
                    - class_id: Class ID
                    - class_name: Class name
                    - face_type: Type of face detected
                - image (optional): PIL Image with detection boxes drawn (if return_image=True)
        """
        try:
            if self.method == "grounding":
                return self._detect_faces_grounding(image, return_image)
            elif self.method == "yolo":
                return self._detect_faces_yolo(image, return_image)
        except Exception as e:
            logger.error(f"Face detection failed: {traceback.format_exc()}")
            raise RuntimeError(f"Face detection error: {e}")

    def _detect_faces_yolo(
        self,
        image: Union[str, Image.Image, bytes, np.ndarray],
        return_image: bool = False,
    ) -> Dict:
        """Detect faces using YOLO World"""
        # Load image
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Use YOLO World for open-vocabulary detection
        # YOLO World detects objects based on the custom classes set via set_classes()
        results = self.model.predict(
            source=img,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        faces = []
        annotated_img = img.copy() if return_image else None

        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    # Get bounding box coordinates (xyxy format)
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())

                    # Get class name from custom classes list
                    # YOLO World returns class_id that corresponds to index in custom_classes
                    if class_id < len(self.custom_classes):
                        class_name = self.custom_classes[class_id]
                    else:
                        class_name = result.names.get(class_id, "unknown")

                    # Determine face type based on class name
                    # For "face" class, it can detect all types of faces
                    if class_name.lower() == "face":
                        face_type = "face"  # Generic face type (can be human, animal, anime, etc.)
                    elif any(keyword in class_name.lower() for keyword in ["human", "person"]):
                        face_type = "human"
                    elif any(keyword in class_name.lower() for keyword in ["animal", "cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]):
                        face_type = "animal"
                    elif any(keyword in class_name.lower() for keyword in ["anime", "cartoon", "manga"]):
                        face_type = "anime"
                    elif any(keyword in class_name.lower() for keyword in ["sketch", "line", "drawing"]):
                        face_type = "sketch"
                    else:
                        logger.debug(f"Dropped unused detected result: {class_name}")
                        face_type = None

                    face_info = {
                        "bbox": bbox,  # [x1, y1, x2, y2] - absolute pixel coordinates
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name,
                        "face_type": face_type,
                    }
                    if face_type is not None:
                        faces.append(face_info)

                    # Draw annotations on image if needed
                    if return_image and annotated_img is not None:
                        draw = ImageDraw.Draw(annotated_img)
                        x1, y1, x2, y2 = bbox
                        # Draw bounding box
                        draw.rectangle(
                            [x1, y1, x2, y2],
                            outline="red",
                            width=2,
                        )
                        # Draw label
                        label = f"{class_name} {confidence:.2f}"
                        draw.text((x1, y1 - 15), label, fill="red")

        result_dict = {"faces": faces}

        if return_image and annotated_img is not None:
            result_dict["image"] = annotated_img

        logger.info(f"Detected {len(faces)} faces using YOLO World")
        return result_dict

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes

        Args:
            bbox1: [x1, y1, x2, y2] format
            bbox2: [x1, y1, x2, y2] format

        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection area
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _calculate_bbox_area(self, bbox: List[float]) -> float:
        """Calculate the area of a bounding box"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def _calculate_containment(self, bbox_small: List[float], bbox_large: List[float]) -> float:
        """
        Calculate how much of bbox_small is contained in bbox_large
        Returns the ratio of intersection area to bbox_small area
        """
        x1_s, y1_s, x2_s, y2_s = bbox_small
        x1_l, y1_l, x2_l, y2_l = bbox_large

        # Calculate intersection
        inter_x1 = max(x1_s, x1_l)
        inter_y1 = max(y1_s, y1_l)
        inter_x2 = min(x2_s, x2_l)
        inter_y2 = min(y2_s, y2_l)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        small_area = (x2_s - x1_s) * (y2_s - y1_s)

        if small_area == 0:
            return 0.0

        return inter_area / small_area

    def _apply_nms(self, faces: List[Dict], iou_threshold: float = 0.4, containment_threshold: float = 0.6) -> List[Dict]:
        """
        Apply Non-Maximum Suppression (NMS) to remove duplicate detections.
        When detections overlap, keeps the one with larger area (preferring whole objects over parts).

        Args:
            faces: List of face detection dictionaries
            iou_threshold: IoU threshold for considering detections as duplicates
            containment_threshold: If a smaller box is contained in a larger box by this ratio, suppress it

        Returns:
            Filtered list of faces with duplicates removed
        """
        if len(faces) == 0:
            return faces

        # Sort by area (largest first), then by confidence as tie-breaker
        # This ensures we keep the larger detection (whole object) over smaller ones (parts)
        for face in faces:
            face["_area"] = self._calculate_bbox_area(face["bbox"])

        sorted_faces = sorted(faces, key=lambda x: (x["_area"], x["confidence"]), reverse=True)

        keep = []
        suppressed = set()

        for i, face in enumerate(sorted_faces):
            if i in suppressed:
                continue

            keep.append(face)
            bbox_i = face["bbox"]
            area_i = face["_area"]

            # Suppress overlapping detections (prefer larger area)
            for j in range(i + 1, len(sorted_faces)):
                if j in suppressed:
                    continue

                bbox_j = sorted_faces[j]["bbox"]
                area_j = sorted_faces[j]["_area"]

                # Check IoU overlap
                iou = self._calculate_iou(bbox_i, bbox_j)
                if iou > iou_threshold:
                    # If overlapping, suppress the smaller one
                    suppressed.add(j)
                    continue

                # Check if smaller box is mostly contained in larger box
                # (e.g., face is contained in whole animal body)
                # Since we sorted by area, area_i >= area_j for j > i
                if area_j < area_i:
                    containment = self._calculate_containment(bbox_j, bbox_i)
                    if containment > containment_threshold:
                        suppressed.add(j)

        # Clean up temporary area field
        for face in keep:
            face.pop("_area", None)

        logger.info(f"NMS filtered {len(faces)} detections to {len(keep)} (IoU threshold: {iou_threshold}, containment threshold: {containment_threshold}, prefer larger area)")
        return keep

    def _detect_faces_grounding(
        self,
        image: Union[str, Image.Image, bytes, np.ndarray],
        return_image: bool = False,
    ) -> Dict:
        """Detect faces using Grounding DINO"""
        # Load image
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Prepare text prompt - join custom classes with ". " separator
        text_prompt = ". ".join(self.custom_classes)
        if not text_prompt.endswith("."):
            text_prompt += "."

        # Process image and text
        inputs = self.processor(images=img, text=text_prompt, return_tensors="pt")
        if self.device:
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        # Note: Grounding DINO uses 'threshold' instead of 'box_threshold'
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=self.conf_threshold,
            text_threshold=self.conf_threshold,
            target_sizes=[img.size[::-1]],  # [height, width]
        )

        faces = []
        annotated_img = img.copy() if return_image else None

        if len(results) > 0:
            result = results[0]

            # Get detections
            # Use text_labels instead of labels to avoid FutureWarning
            boxes = result.get("boxes", [])
            text_labels = result.get("text_labels", [])
            # Fallback to labels if text_labels not available
            if not text_labels:
                labels = result.get("labels", [])
                # Convert label IDs to class names if needed
                text_labels = [self.custom_classes[label] if isinstance(label, int) and label < len(self.custom_classes) else str(label) for label in labels]
            scores = result.get("scores", [])

            for i, (box, label, score) in enumerate(zip(boxes, text_labels, scores)):
                # Grounding DINO returns boxes as [x1, y1, x2, y2]
                if isinstance(box, torch.Tensor):
                    bbox = box.tolist()
                else:
                    bbox = list(box)
                # Ensure it's [x1, y1, x2, y2] format
                if len(bbox) == 4:
                    bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                else:
                    # If it's in center format, convert
                    x_center, y_center, width, height = bbox
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    bbox = [float(x1), float(y1), float(x2), float(y2)]

                # Get class name from label
                # Grounding DINO may return multiple class names concatenated
                class_name_raw = label if isinstance(label, str) else self.custom_classes[0]

                # If class_name contains multiple classes, try to extract the most specific one
                # Priority: specific classes (animal, anime, sketch) > human > generic face
                class_name = class_name_raw
                if isinstance(class_name_raw, str) and len(self.custom_classes) > 1:
                    class_name_lower = class_name_raw.lower()
                    # Check for specific classes first
                    if any(keyword in class_name_lower for keyword in ["animal"]):
                        for c in self.custom_classes:
                            if "animal" in c.lower():
                                class_name = c
                                break
                    elif any(keyword in class_name_lower for keyword in ["anime", "cartoon"]):
                        for c in self.custom_classes:
                            if any(k in c.lower() for k in ["anime", "cartoon"]):
                                class_name = c
                                break
                    elif any(keyword in class_name_lower for keyword in ["sketch", "line", "drawing"]):
                        for c in self.custom_classes:
                            if any(k in c.lower() for k in ["sketch", "line", "drawing"]):
                                class_name = c
                                break
                    elif any(keyword in class_name_lower for keyword in ["human", "person"]):
                        for c in self.custom_classes:
                            if any(k in c.lower() for k in ["human", "person"]):
                                class_name = c
                                break

                # Determine face type based on class name
                if class_name.lower() == "face":
                    face_type = "face"
                elif any(keyword in class_name.lower() for keyword in ["human", "person"]):
                    face_type = "human"
                elif any(keyword in class_name.lower() for keyword in ["animal", "cat", "dog", "bird"]):
                    face_type = "animal"
                elif any(keyword in class_name.lower() for keyword in ["anime", "cartoon", "manga"]):
                    face_type = "anime"
                elif any(keyword in class_name.lower() for keyword in ["sketch", "line", "drawing"]):
                    face_type = "sketch"
                else:
                    face_type = class_name.lower()

                face_info = {
                    "bbox": bbox,
                    "confidence": float(score),
                    "class_id": i,
                    "class_name": class_name,
                    "face_type": face_type,
                }
                faces.append(face_info)

                # Draw annotations if needed
                if return_image and annotated_img is not None:
                    draw = ImageDraw.Draw(annotated_img)
                    x1, y1, x2, y2 = bbox
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    label = f"{class_name} {score:.2f}"
                    draw.text((x1, y1 - 15), label, fill="red")

        # Apply NMS to remove duplicate detections (only when multiple classes are specified)
        if len(self.custom_classes) > 1:
            faces = self._apply_nms(faces, iou_threshold=0.4, containment_threshold=0.6)
            # Re-draw annotations after NMS if needed
            if return_image and annotated_img is not None and len(faces) > 0:
                annotated_img = img.copy()
                draw = ImageDraw.Draw(annotated_img)
                for face in faces:
                    x1, y1, x2, y2 = face["bbox"]
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    label = f"{face['class_name']} {face['confidence']:.2f}"
                    draw.text((x1, y1 - 15), label, fill="red")

        result_dict = {"faces": faces}
        if return_image and annotated_img is not None:
            result_dict["image"] = annotated_img

        logger.info(f"Detected {len(faces)} faces using Grounding DINO (after NMS)")
        return result_dict

    def detect_faces_from_bytes(self, image_bytes: bytes, **kwargs) -> Dict:
        """
        Detect faces from byte data

        Args:
            image_bytes: Image byte data
            **kwargs: Additional parameters passed to detect_faces

        Returns:
            Detection result dictionary
        """
        return self.detect_faces(image_bytes, **kwargs)

    def extract_face_regions(self, image: Union[str, Image.Image, bytes], expand_ratio: float = 0.1) -> List[Image.Image]:
        """
        Extract detected face regions

        Args:
            image: Input image
            expand_ratio: Bounding box expansion ratio to include more context

        Returns:
            List of extracted face region images
        """
        result = self.detect_faces(image)
        faces = result["faces"]

        # Load original image
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        face_regions = []
        img_width, img_height = img.size

        for face in faces:
            x1, y1, x2, y2 = face["bbox"]

            # Expand bounding box
            width = x2 - x1
            height = y2 - y1
            expand_x = width * expand_ratio
            expand_y = height * expand_ratio

            x1 = max(0, int(x1 - expand_x))
            y1 = max(0, int(y1 - expand_y))
            x2 = min(img_width, int(x2 + expand_x))
            y2 = min(img_height, int(y2 + expand_y))

            # Crop region
            face_region = img.crop((x1, y1, x2, y2))
            face_regions.append(face_region)

        return face_regions

    def count_faces(self, image: Union[str, Image.Image, bytes]) -> int:
        """
        Count number of faces in image

        Args:
            image: Input image

        Returns:
            Number of detected faces
        """
        result = self.detect_faces(image, return_image=False)
        return len(result["faces"])


def detect_faces_in_image(
    image_path: str,
    method: str = "grounding",
    model_path: str = None,
    conf_threshold: float = None,
    return_image: bool = False,
    custom_classes: List[str] = None,
) -> Dict:
    """
    Convenience function: detect faces in image

    Args:
        image_path: Image path
        method: Detection method ("yolo", or "grounding"), default "yolo"
        model_path: YOLO World model path (only for method="yolo")
        conf_threshold: Confidence threshold (None for adaptive, only for method="yolo")
        return_image: Whether to return annotated image
        custom_classes: List of custom class names for YOLO (default: ["face"])

    Returns:
        Detection result dictionary containing:
            - faces: List of face detection results with bbox coordinates [x1, y1, x2, y2]
              Each face contains: bbox, confidence, class_id, class_name, face_type
            - image (optional): Annotated image with detection boxes

    Examples:
        # Detect faces using YOLO World with default "face" class
        result = detect_faces_in_image("image.jpg", method="yolo")

        # Detect with YOLO World and custom classes
        result = detect_faces_in_image("image.jpg", method="yolo",
                                      custom_classes=["human face", "animal face"])

        # Detect with Grounding DINO
        result = detect_faces_in_image("image.jpg", method="grounding",
                                      custom_classes=["animal face"])
    """
    detector = FaceDetector(
        method=method,
        model_path=model_path,
        conf_threshold=conf_threshold,
        custom_classes=custom_classes,
    )
    return detector.detect_faces(image_path, return_image=return_image)


if __name__ == "__main__":
    # Test code
    import sys

    if len(sys.argv) < 2:
        print("Usage: python face_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    detector = FaceDetector()
    result = detector.detect_faces(image_path, return_image=True)

    print(f"Detected {len(result['faces'])} faces:")
    for i, face in enumerate(result["faces"]):
        print(f"  Face {i + 1}: {face}")

    output_path = "detected_faces.png"
    result["image"].save(output_path)
    print(f"Annotated image saved to: {output_path}")
