from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class FaceDetector:
    """Lightweight OpenCV face detector and cropper.

    This uses OpenCV's bundled Haar cascade, so no extra model download is
    required. For production deployments, replace it with a stronger detector
    such as RetinaFace or SCRFD.
    """

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Could not load face detector: {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, image: np.ndarray):
        if image is None or image.size == 0:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(40, 40),
        )

    def largest_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        faces = self.detect(image)
        if len(faces) == 0:
            return None
        return max(faces, key=lambda box: int(box[2]) * int(box[3]))

    def crop_largest(self, image: np.ndarray, margin: float = 0.20) -> np.ndarray:
        box = self.largest_face(image)
        if box is None:
            raise ValueError("No face detected in image")

        x, y, w, h = [int(value) for value in box]
        pad_x = int(w * margin)
        pad_y = int(h * margin)
        height, width = image.shape[:2]
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)
        return image[y1:y2, x1:x2]


def read_face_image(path: str, detector: Optional[FaceDetector] = None):
    image = cv2.imread(str(Path(path)))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    detector = detector or FaceDetector()
    return detector.crop_largest(image)
