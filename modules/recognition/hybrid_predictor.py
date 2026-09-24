"""Inference wrapper for the hybrid CNN + BiLSTM + geometry recogniser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from modules.recognition.features import normalize_landmark_features
from modules.recognition.hand_crop import DEFAULT_CROP_SIZE, crop_hand
from modules.recognition.hybrid_trainer import METADATA_FILENAME, MODEL_FILENAME

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class HybridPredictor:
    """Predicts signs from a hand crop plus the hand's landmark geometry.

    Both streams come from one MediaPipe detection that the caller already
    performs, so the extra accuracy costs no additional preprocessing.
    """

    backend_name = "hybrid"

    def __init__(
        self,
        model_path: str = f"models/recognition/{MODEL_FILENAME}",
        metadata_path: Optional[str] = f"models/recognition/{METADATA_FILENAME}",
        input_size: Optional[int] = None,
        backend_name: Optional[str] = None,
    ):
        if backend_name:
            self.backend_name = backend_name
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Hybrid model not found: {model_file}")

        from tensorflow import keras

        self.model = keras.models.load_model(model_file)

        self.class_names: List[str] = []
        self.input_size = input_size or DEFAULT_CROP_SIZE
        if metadata_path and Path(metadata_path).exists():
            meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            self.class_names = list(meta.get("class_names", []))
            if input_size is None:
                self.input_size = int(meta.get("input_size", self.input_size))

        if not self.class_names:
            self.class_names = [str(i) for i in range(int(self.model.output_shape[-1]))]

    def _no_hand_result(self) -> dict:
        label = "nothing"
        index = self.class_names.index(label) if label in self.class_names else -1
        return {
            "pred_index": index,
            "label": label,
            "confidence": 0.0,
            "hand_detected": False,
            "backend": self.backend_name,
        }

    def predict_frame(
        self,
        image_path: Optional[str] = None,
        bgr_image: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> dict:
        if cv2 is None:
            raise ImportError("opencv (cv2) is required for hybrid inference")
        if landmarks is None or not np.any(landmarks):
            return self._no_hand_result()

        image = bgr_image
        if image is None:
            if image_path is None:
                raise ValueError("Either image_path or bgr_image must be provided")
            image = cv2.imread(str(image_path))
            if image is None:
                return self._no_hand_result()

        crop = crop_hand(image, landmarks, size=self.input_size)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Identical normalization to training.
        geom = normalize_landmark_features(
            np.asarray(landmarks, dtype=np.float32)
        ).reshape(1, -1).astype(np.float32)

        probs = self.model.predict(
            [np.expand_dims(rgb, 0), geom], verbose=0
        )[0]
        index = int(np.argmax(probs))

        return {
            "pred_index": index,
            "label": self.class_names[index] if index < len(self.class_names) else str(index),
            "confidence": float(probs[index]),
            "hand_detected": True,
            "backend": self.backend_name,
        }
