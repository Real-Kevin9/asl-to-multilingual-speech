from __future__ import annotations

from pathlib import Path
from typing import Optional

from modules.recognition.model import RecognitionPredictor
from modules.recognition.preprocess import ASLPreprocessor

_DEFAULT_HAND_MODEL = "models/recognition/hand_landmarker.task"


def predict_image_label(
    image_path: str,
    model_path: str,
    encoder_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    hand_model_path: Optional[str] = None,
) -> dict:
    """Predict the ASL label for a single image.

    Extracts hand landmarks with MediaPipe, then runs the trained recognizer.
    The same wrist-centering normalization and feature scaler used in training
    are applied here (via :class:`RecognitionPredictor`).
    """
    if scaler_path is None:
        default_scaler = Path(model_path).parent / "scaler.joblib"
        scaler_path = str(default_scaler) if default_scaler.exists() else None

    predictor = RecognitionPredictor(
        model_path=model_path,
        scaler_path=scaler_path,
        encoder_path=encoder_path,
    )

    if hand_model_path is None and Path(_DEFAULT_HAND_MODEL).exists():
        hand_model_path = _DEFAULT_HAND_MODEL

    preprocessor = ASLPreprocessor(model_asset_path=hand_model_path)
    landmarks = preprocessor.extract_landmarks(image_path)

    return predictor.predict(landmarks)
