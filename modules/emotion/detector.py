from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# Emotion classes targeted by the project (proposal objective 4).
EMOTIONS = ["happy", "sad", "angry", "neutral"]

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

    This is the outline/skeleton implementation. It performs real face
    detection with an OpenCV Haar cascade, and exposes a classifier hook:

    * If a trained emotion model is provided via ``model_path`` and can be
      loaded (Keras ``.h5``/``.keras``), it is used for classification.
    * Otherwise it falls back to returning ``neutral`` (still reporting whether
      a face was found), so the end-to-end pipeline always runs.

    The interface (:meth:`detect`) is stable, so the fallback can later be
    swapped for a trained CNN without touching the rest of the pipeline.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.backend = "fallback_neutral"
        self._face_cascade = None

        if cv2 is not None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if cascade_path.exists():
                self._face_cascade = cv2.CascadeClassifier(str(cascade_path))

        if model_path and Path(model_path).exists():
            self._try_load_model(model_path)

    def _try_load_model(self, model_path: str) -> None:
        try:
            from tensorflow import keras  # type: ignore

            self.model = keras.models.load_model(model_path)
            self.backend = "keras"
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

    def detect(self, image: ImageLike) -> Dict[str, object]:
        """Return the detected emotion for a face image.

        Returns a dict with ``emotion``, ``confidence``, ``face_found`` and the
        matching ``prosody`` parameters.
        """
        bgr = self._read_image(image)
        face = self._detect_face(bgr) if bgr is not None else None
        face_found = face is not None

        emotion = "neutral"
        confidence = 0.0

        if self.model is not None and face_found:
            try:
                inp = cv2.resize(face, (48, 48)).astype("float32") / 255.0
                inp = inp.reshape(1, 48, 48, 1)
                proba = self.model.predict(inp, verbose=0)[0]
                idx = int(np.argmax(proba))
                emotion = EMOTIONS[idx] if idx < len(EMOTIONS) else "neutral"
                confidence = float(np.max(proba))
            except Exception as exc:  # pragma: no cover
                print(f"[emotion] inference failed ({exc}); defaulting to neutral.")
                emotion = "neutral"

        return {
            "emotion": emotion,
            "confidence": confidence,
            "face_found": face_found,
            "prosody": EMOTION_PROSODY.get(emotion, EMOTION_PROSODY["neutral"]),
        }


def detect_emotion(image: ImageLike, model_path: Optional[str] = None) -> Dict[str, object]:
    """Module-level convenience wrapper around :class:`EmotionDetector`."""
    return EmotionDetector(model_path=model_path).detect(image)
