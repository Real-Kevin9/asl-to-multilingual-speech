from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# Emotion classes targeted by the project (proposal objective 4).
EMOTIONS = ["happy", "sad", "angry", "neutral"]

DEFAULT_MODEL_PATH = "models/emotion/emotion_model.keras"
DEFAULT_METADATA_PATH = "models/emotion/emotion_metadata.json"

# Maps each emotion to text-to-speech prosody multipliers. These are consumed
# by the TTS module to modulate pitch and speaking rate.
#   rate:  1.0 = normal speed   (>1 faster, <1 slower)
#   pitch: 0   = normal pitch   (semitone-ish offset, used by backends that support it)
EMOTION_PROSODY: Dict[str, Dict[str, float]] = {
    "happy":   {"rate": 1.15, "pitch": 3.0},
    "sad":     {"rate": 0.85, "pitch": -3.0},
    "angry":   {"rate": 1.10, "pitch": 1.0},
    "neutral": {"rate": 1.00, "pitch": 0.0},
}

ImageLike = Union[str, "np.ndarray"]


class EmotionDetector:
    """Facial-expression classifier for TTS prosody modulation.

    Uses OpenCV Haar cascades for face detection, then a trained Keras CNN
    (48×48 grayscale → softmax over ``EMOTIONS``) when
    ``models/emotion/emotion_model.keras`` is present.

    If the model is missing or no face is found, returns ``neutral`` so the
    end-to-end pipeline always runs.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        auto_load: bool = True,
    ):
        self.model = None
        self.backend = "fallback_neutral"
        self.class_names: List[str] = list(EMOTIONS)
        self._face_cascade = None

        if cv2 is not None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if cascade_path.exists():
                self._face_cascade = cv2.CascadeClassifier(str(cascade_path))

        path = model_path
        if path is None and auto_load and Path(DEFAULT_MODEL_PATH).exists():
            path = DEFAULT_MODEL_PATH
        meta_path = metadata_path or DEFAULT_METADATA_PATH
        if path and Path(path).exists():
            self._try_load_model(path, meta_path)

    def _try_load_model(self, model_path: str, metadata_path: str) -> None:
        try:
            from tensorflow import keras  # type: ignore

            self.model = keras.models.load_model(model_path)
            self.backend = "keras"
            meta_file = Path(metadata_path)
            if meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                names = meta.get("emotions")
                if isinstance(names, list) and names:
                    self.class_names = list(names)
        except Exception as exc:  # pragma: no cover - dependency dependent
            print(f"[emotion] Could not load model {model_path} ({exc}); using fallback.")
            self.model = None
            self.backend = "fallback_neutral"

    def _read_image(self, image: ImageLike) -> Optional["np.ndarray"]:
        if cv2 is None:
            return None
        if isinstance(image, str):
            return cv2.imread(image)
        return image

    def _detect_face(self, bgr: "np.ndarray") -> Optional["np.ndarray"]:
        if self._face_cascade is None or bgr is None:
            return None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        # Largest detected face.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return gray[y:y + h, x:x + w]

    def _classify_face(self, face_gray: "np.ndarray") -> tuple[str, float]:
        inp = cv2.resize(face_gray, (48, 48)).astype("float32") / 255.0
        inp = inp.reshape(1, 48, 48, 1)
        proba = self.model.predict(inp, verbose=0)[0]
        idx = int(np.argmax(proba))
        if idx < len(self.class_names):
            emotion = self.class_names[idx]
        else:
            emotion = "neutral"
        return emotion, float(np.max(proba))

    def detect(self, image: ImageLike) -> Dict[str, object]:
        """Return the detected emotion for a face image.

        Returns a dict with ``emotion``, ``confidence``, ``face_found``,
        ``backend`` and the matching ``prosody`` parameters.
        """
        bgr = self._read_image(image)
        face = self._detect_face(bgr) if bgr is not None else None
        face_found = face is not None

        emotion = "neutral"
        confidence = 0.0

        if self.model is not None and face_found:
            try:
                emotion, confidence = self._classify_face(face)
            except Exception as exc:  # pragma: no cover
                print(f"[emotion] inference failed ({exc}); defaulting to neutral.")
                emotion = "neutral"

        return {
            "emotion": emotion,
            "confidence": confidence,
            "face_found": face_found,
            "backend": self.backend,
            "prosody": EMOTION_PROSODY.get(emotion, EMOTION_PROSODY["neutral"]),
        }


def detect_emotion(image: ImageLike, model_path: Optional[str] = None) -> Dict[str, object]:
    """Module-level convenience wrapper around :class:`EmotionDetector`."""
    return EmotionDetector(model_path=model_path).detect(image)
