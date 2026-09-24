from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from modules.recognition.features import normalize_landmark_features


class RecognitionPredictor:
    """Loads a trained sign-recognition model and predicts labels from landmarks.

    Wraps the classifier, the feature scaler, and the label encoder produced by
    :class:`RecognitionTrainer` so that inference applies the exact same
    normalization pipeline used during training.

    This is the landmark (MLP) backend. It is kept alongside the CNN+LSTM
    backend as a lightweight, CPU-friendly fallback.
    """

    backend_name = "landmark_mlp"

    def __init__(
        self,
        model_path: str = "models/recognition/recognition_model.joblib",
        scaler_path: Optional[str] = "models/recognition/scaler.joblib",
        encoder_path: Optional[str] = "models/recognition/label_encoder.joblib",
    ):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Recognition model not found: {model_file}")
        self.model = joblib.load(model_file)

        self.scaler = None
        if scaler_path and Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)

        self.encoder = None
        if encoder_path and Path(encoder_path).exists():
            self.encoder = joblib.load(encoder_path)

    def _prepare(self, landmarks: np.ndarray) -> np.ndarray:
        features = normalize_landmark_features(np.asarray(landmarks, dtype=np.float32))
        features = features.reshape(1, -1).astype(np.float32)
        if self.scaler is not None:
            features = self.scaler.transform(features)
        return features

    def predict(self, landmarks: np.ndarray) -> dict:
        """Predict the sign label for a single 63-dim landmark vector."""
        features = self._prepare(landmarks)
        pred = int(self.model.predict(features)[0])

        confidence = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            confidence = float(np.max(proba))

        if self.encoder is not None:
            label = str(self.encoder.inverse_transform([pred])[0])
        else:
            label = str(pred)

        return {"pred_index": pred, "label": label, "confidence": confidence}

    def predict_frame(
        self,
        image_path: Optional[str] = None,
        bgr_image: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> dict:
        """Uniform per-frame interface shared with the CNN+LSTM backend.

        This backend only needs the landmark vector; ``image_path`` and
        ``bgr_image`` are accepted so callers can treat both backends alike.
        """
        if landmarks is None:
            raise ValueError("RecognitionPredictor requires landmarks")

        result = self.predict(landmarks)
        result["hand_detected"] = bool(np.any(landmarks))
        result["backend"] = self.backend_name
        return result


def load_recognition_model(
    model_path: str = "models/recognition/recognition_model.joblib",
) -> RecognitionPredictor:
    """Convenience loader returning a ready-to-use :class:`RecognitionPredictor`."""
    return RecognitionPredictor(model_path=model_path)
