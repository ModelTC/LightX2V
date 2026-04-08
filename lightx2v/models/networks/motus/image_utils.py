import cv2
import numpy as np


def resize_with_padding(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_size
    original_height, original_width = frame.shape[:2]

    scale = min(target_height / original_height, target_width / original_width)
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)

    resized_frame = cv2.resize(frame, (new_width, new_height))
    padded_frame = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)

    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    padded_frame[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized_frame
    return padded_frame
