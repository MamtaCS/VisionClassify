from typing import Any

import cv2
import numpy as np

IMAGE_SIZE = (128, 128)
HOG = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)


def remove_capture_overlay(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    region_height = max(40, int(height * 0.10))
    region_width = max(1, int(width * 0.50))
    region = image[:region_height, :region_width]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    green_text = cv2.inRange(
        hsv,
        np.array((35, 100, 150), dtype=np.uint8),
        np.array((85, 255, 255), dtype=np.uint8),
    )
    if cv2.countNonZero(green_text) < 100:
        return image

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:region_height, :region_width] = green_text
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def extract_features(
    image: np.ndarray,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Cannot extract features from an empty image.")

    cleaned = remove_capture_overlay(image)
    resized = cv2.resize(cleaned, image_size)
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    features = HOG.compute(grayscale)
    if features is None:
        raise ValueError("HOG feature extraction failed.")
    return features.flatten().astype(np.float32, copy=False)


def classify_image(
    model: Any,
    image: np.ndarray,
    score_threshold: float,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[str | None, str, float | None, np.ndarray | None]:
    features = extract_features(image, image_size).reshape(1, -1)
    candidate = str(model.predict(features)[0])
    probabilities = None
    score = None

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features)[0], dtype=np.float64)
        classes = np.asarray(model.classes_)
        matching_indices = np.flatnonzero(classes == candidate)
        if probabilities.size != len(classes) or matching_indices.size != 1:
            raise ValueError("Model prediction and class probability outputs do not match.")
        score = float(probabilities[matching_indices[0]])

    accepted = candidate if score is None or score >= score_threshold else None
    return accepted, candidate, score, probabilities
