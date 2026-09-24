from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2

try:
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
    from mediapipe.tasks.python.vision.core import image as image_lib
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    from mediapipe.tasks.python.core import base_options as base_options_lib
    MEDIAPIPE_TASKS_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    MEDIAPIPE_TASKS_AVAILABLE = False


class ASLPreprocessor:
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        fail_on_missing: bool = False,
        model_asset_path: Optional[str] = None
    ):
        """Initialize hand landmark extractor.

        Supports two backends:
        1. MediaPipe Tasks HandLandmarker (requires TFLite model file)
        2. Fallback: returns zero vectors with warning

        Args:
            static_image_mode: Whether to use static image mode (unused in fallback).
            max_num_hands: Maximum number of hands to detect (unused in fallback).
            fail_on_missing: If True, raise ImportError if dependencies unavailable.
            model_asset_path: Path to hand_landmarker.tflite model file for Tasks API.
                             If None and MediaPipe Tasks available, will attempt fallback.

        Raises:
            ImportError: If dependencies are missing and fail_on_missing=True.
            FileNotFoundError: If model_asset_path is specified but not found.
            ValueError: If MediaPipe Tasks API is unavailable and fail_on_missing=True.
        """
        if cv2 is None:
            if fail_on_missing:
                raise ImportError("opencv (cv2) is required for preprocessing but not installed")
            print("Warning: opencv not available — preprocessing will return zero vectors for all images.")
            self.landmarker = None
            self.backend = "none"
            return

        self.landmarker = None
        self.backend = "none"

        # Try to initialize MediaPipe Tasks backend if model path provided
        if model_asset_path is not None:
            if not MEDIAPIPE_TASKS_AVAILABLE:
                if fail_on_missing:
                    raise ImportError("mediapipe.tasks not available but model path provided")
                print("Warning: mediapipe.tasks not available, falling back to zero vectors")
                return

            try:
                if not Path(model_asset_path).exists():
                    raise FileNotFoundError(f"Model asset not found: {model_asset_path}")

                base_options = base_options_lib.BaseOptions(model_asset_path=model_asset_path)
                options = HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=VisionTaskRunningMode.IMAGE,
                    num_hands=max_num_hands,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.landmarker = HandLandmarker.create_from_options(options)
                self.backend = "mediapipe_tasks"
                self.static_image_mode = static_image_mode
                self.max_num_hands = max_num_hands
                print(f"Using MediaPipe Tasks backend with model: {model_asset_path}")
            except Exception as e:
                if fail_on_missing:
                    raise
                print(f"Warning: Failed to initialize MediaPipe Tasks: {e}")
                print("Preprocessing will return zero vectors for all images.")
                return
        else:
            # No model provided
            if not MEDIAPIPE_TASKS_AVAILABLE:
                if fail_on_missing:
                    raise ValueError(
                        "MediaPipe Tasks not available and no model_asset_path provided. "
                        "Please provide path to hand_landmarker.tflite or install mediapipe.solutions"
                    )
                print("Warning: MediaPipe not available — preprocessing will return zero vectors for all images.")
            else:
                print("Warning: No model_asset_path provided for MediaPipe Tasks HandLandmarker.")
                print("Preprocessing will return zero vectors for all images.")

    def extract_landmarks(self, image_path: str) -> np.ndarray:
        """Extract hand landmarks from an image.

        Returns:
            Flattened array of 21 landmarks × 3 coordinates (x, y, z) = 63 floats.
            If no hand detected or if error occurs, returns zero vector.
        """
        if self.backend == "none":
            return np.zeros(63, dtype=np.float32)

        if self.backend == "mediapipe_tasks":
            try:
                # Load image using MediaPipe Image API
                image = image_lib.Image.create_from_file(image_path)

                # Detect hand landmarks
                result = self.landmarker.detect(image)

                # Extract landmarks from first detected hand
                if not result.hand_landmarks or len(result.hand_landmarks) == 0:
                    return np.zeros(63, dtype=np.float32)

                landmarks = result.hand_landmarks[0]  # Get first hand
                coords = []
                for landmark in landmarks:
                    coords.extend([landmark.x, landmark.y, landmark.z])

                return np.array(coords, dtype=np.float32)
            except Exception as e:
                # Return zero vector on any error
                print(f"Warning: Failed to extract landmarks from {image_path}: {e}")
                return np.zeros(63, dtype=np.float32)

        # Fallback for unknown backend
        return np.zeros(63, dtype=np.float32)

    def extract_landmarks_from_array(self, bgr_image: np.ndarray) -> np.ndarray:
        """Extract hand landmarks directly from an in-memory BGR frame.

        Used by the live webcam path so frames do not have to be written to a
        temporary file first.

        Returns:
            Flattened 63-float vector, or zeros when no hand is detected.
        """
        if self.backend != "mediapipe_tasks" or bgr_image is None:
            return np.zeros(63, dtype=np.float32)

        try:
            import mediapipe as mp

            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(image)

            if not result.hand_landmarks:
                return np.zeros(63, dtype=np.float32)

            coords: List[float] = []
            for landmark in result.hand_landmarks[0]:
                coords.extend([landmark.x, landmark.y, landmark.z])
            return np.array(coords, dtype=np.float32)
        except Exception as exc:
            print(f"Warning: Failed to extract landmarks from frame: {exc}")
            return np.zeros(63, dtype=np.float32)

    def close(self) -> None:
        """Release the MediaPipe landmarker.

        Closing explicitly avoids the noisy ``TypeError`` MediaPipe raises when
        its landmarker is finalized during interpreter shutdown.
        """
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    def __enter__(self) -> "ASLPreprocessor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def load_dataset(
        self,
        root_dir: str,
        per_class_limit: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Load images from root_dir and extract hand landmarks.

        Args:
            root_dir: Directory containing class subdirectories with .jpg/.png images.
            per_class_limit: If set, only process the first N images per class.
                Useful for quickly building a small outline dataset instead of
                running MediaPipe over the full ~200k-image corpus.

        Returns:
            Tuple of (features, labels) where features is a list of landmark vectors
            and labels is a list of class names.
        """
        features: List[np.ndarray] = []
        labels: List[str] = []

        for class_dir in sorted(Path(root_dir).iterdir()):
            if not class_dir.is_dir():
                continue

            label = class_dir.name
            image_paths = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
            if per_class_limit is not None:
                image_paths = image_paths[:per_class_limit]

            for image_path in image_paths:
                features.append(self.extract_landmarks(str(image_path)))
                labels.append(label)

            print(f"  {label}: processed {len(image_paths)} images")

        return features, labels


def preprocess_dataset(
    root_dir: str,
    fail_on_missing: bool = False,
    model_asset_path: Optional[str] = None,
    per_class_limit: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[str]]:
    """Preprocess ASL dataset using hand landmark extraction.

    Args:
        root_dir: Path to directory containing class subdirectories.
        fail_on_missing: If True, raise error if dependencies missing.
        model_asset_path: Path to hand_landmarker.task model file.
        per_class_limit: If set, only process the first N images per class.

    Returns:
        Tuple of (features, labels).
    """
    preprocessor = ASLPreprocessor(
        fail_on_missing=fail_on_missing,
        model_asset_path=model_asset_path
    )
    return preprocessor.load_dataset(root_dir, per_class_limit=per_class_limit)
