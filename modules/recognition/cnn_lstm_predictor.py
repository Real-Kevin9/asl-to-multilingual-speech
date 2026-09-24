"""Inference wrapper for the MobileNetV2 + BiLSTM recogniser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from modules.recognition.cnn_lstm_trainer import METADATA_FILENAME, MODEL_FILENAME
from modules.recognition.hand_crop import DEFAULT_CROP_SIZE, crop_hand

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class CNNLSTMPredictor:
    """Predicts sign labels from images using the trained CNN+LSTM model.

    Inference mirrors training exactly: the frame is cropped to the MediaPipe
    hand bounding box, resized, and passed through MobileNetV2 -> BiLSTM.

    When no hand is detected the model is **not** run. The cached training set
    excluded hand-less frames, so the network has no meaningful response to
    them; returning ``nothing`` directly avoids random predictions when the hand
    leaves the webcam view.
    """

    backend_name = "cnn_lstm"

    def __init__(
        self,
        model_path: str = f"models/recognition/{MODEL_FILENAME}",
        metadata_path: Optional[str] = f"models/recognition/{METADATA_FILENAME}",
        input_size: Optional[int] = None,
    ):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"CNN+LSTM model not found: {model_file}")

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
            n = int(self.model.output_shape[-1])
            self.class_names = [str(i) for i in range(n)]

    def _label(self, index: int) -> str:
        if 0 <= index < len(self.class_names):
            return self.class_names[index]
        return str(index)

    def _no_hand_result(self) -> dict:
        """Result used when MediaPipe finds no hand.

        ``nothing`` is reported directly rather than as a model output: the crop
        cache has no ``nothing`` samples (a hand-less frame yields no crop), so
        the CNN never learns that class. The gloss assembler already treats
        ``nothing`` as a no-op token.
        """
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
        """Predict the sign for one frame.

        Args:
            image_path: Path to the frame (used when ``bgr_image`` is None).
            bgr_image: In-memory BGR frame (preferred for webcam use).
            landmarks: 63-dim MediaPipe landmark vector for the hand crop.
        """
        if cv2 is None:
            raise ImportError("opencv (cv2) is required for CNN+LSTM inference")

        if landmarks is not None and not np.any(landmarks):
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
        batch = np.expand_dims(rgb, axis=0)

        probs = self.model.predict(batch, verbose=0)[0]
        index = int(np.argmax(probs))

        return {
            "pred_index": index,
            "label": self._label(index),
            "confidence": float(probs[index]),
            "hand_detected": landmarks is None or bool(np.any(landmarks)),
            "backend": self.backend_name,
        }

    # Kept so the CNN predictor can stand in for the landmark predictor's API.
    def predict(self, landmarks: np.ndarray) -> dict:  # pragma: no cover
        raise NotImplementedError(
            "CNNLSTMPredictor needs pixels; call predict_frame(image_path=...) instead."
        )
