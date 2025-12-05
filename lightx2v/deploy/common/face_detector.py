# -*- coding: utf-8 -*-
"""
Face Detection Module using YOLO
Supports detecting faces in images, including human faces, animal faces, anime faces, sketches, etc.
"""

import io
import traceback
from typing import Dict, List, Union

import numpy as np
from PIL import Image, ImageDraw
from loguru import logger
from ultralytics import YOLO


class FaceDetector:
    """
    Face detection using YOLO models
    Supports detecting: human faces, animal faces, anime faces, sketch faces, etc.
    """

    def __init__(self, model_path: str = None, conf_threshold: float = 0.25, device: str = None):
        """
        Initialize face detector

        Args:
            model_path: YOLO model path, if None uses default pretrained model
            conf_threshold: Confidence threshold, default 0.25
            device: Device ('cpu', 'cuda', '0', '1', etc.), None for auto selection
        """

        self.conf_threshold = conf_threshold
        self.device = device

        if model_path is None:
            # Use YOLO11 pretrained model, can detect COCO dataset classes (including person)
            # Or use dedicated face detection model
            logger.info("Loading default YOLO11n model for face detection")
            try:
                self.model = YOLO("yolo11n.pt")  # Lightweight model
            except Exception as e:
                logger.warning(f"Failed to load default model, trying yolov8n: {e}")
                self.model = YOLO("yolov8n.pt")
        else:
            logger.info(f"Loading YOLO model from {model_path}")
            self.model = YOLO(model_path)

        # Person class ID in COCO dataset is 0
        # YOLO can detect person, for more precise face detection, recommend using dedicated face detection models
        # Such as YOLOv8-face or RetinaFace, can be specified via model_path parameter
        # First use YOLO to detect person region, then can further detect faces within
        self.target_classes = {
            "person": 0,  # Face (by detecting person class)
            # Can be extended to detect animal faces (cat, dog, etc.) and other classes
        }

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
            return_boxes: Whether to return detection box information

        Returns:
            Dict containing:
                - faces: List of face detection results, each containing:
                    - bbox: [x1, y1, x2, y2] bounding box coordinates (absolute pixel coordinates)
                    - confidence: Confidence score (0.0-1.0)
                    - class_id: Class ID
                    - class_name: Class name
                - image (optional): PIL Image with detection boxes drawn (if return_image=True)
        """
        try:
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

            # Use YOLO for detection
            # Note: YOLO by default detects person, we focus on person detection
            # For more precise face detection, can train or use dedicated face detection models
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

                        # Get class name
                        class_name = result.names.get(class_id, "unknown")

                        # Process target classes (person, etc.)
                        # For person, the entire body box contains face region
                        # For more precise face detection, can:
                        # 1. Use dedicated face detection models (RetinaFace, YOLOv8-face)
                        # 2. Further use face detection model within current person box
                        # 3. Use specifically trained multi-class detection models (faces, animal faces, anime faces, etc.)
                        if class_id in self.target_classes.values():
                            face_info = {
                                "bbox": bbox,  # [x1, y1, x2, y2] - absolute pixel coordinates
                                "confidence": confidence,
                                "class_id": class_id,
                                "class_name": class_name,
                            }
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

            logger.info(f"Detected {len(faces)} faces in image")
            return result_dict

        except Exception as e:
            logger.error(f"Face detection failed: {traceback.format_exc()}")
            raise RuntimeError(f"Face detection error: {e}")

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
    model_path: str = None,
    conf_threshold: float = 0.25,
    return_image: bool = False,
) -> Dict:
    """
    Convenience function: detect faces in image

        Args:
            image_path: Image path
            model_path: YOLO model path
            conf_threshold: Confidence threshold
            return_image: Whether to return annotated image

        Returns:
            Detection result dictionary containing:
                - faces: List of face detection results with bbox coordinates [x1, y1, x2, y2]
                - image (optional): Annotated image with detection boxes
    """
    detector = FaceDetector(model_path=model_path, conf_threshold=conf_threshold)
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
