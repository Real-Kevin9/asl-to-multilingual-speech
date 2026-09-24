"""Recogniser backend selection.

Four recognition backends exist and share the ``predict_frame`` interface:

``hybrid_user``
    The hybrid model after fine-tuning on samples captured from this machine's
    own webcam. Preferred when present: the corpus was filmed by other people in
    other rooms, and closing that domain gap is worth far more than any
    architecture change (measured 65.9 % to 94.6 % on held-out user samples).

``hybrid``
    MobileNetV2 + 2-layer BiLSTM over a MediaPipe hand crop, fused with the
    hand's landmark geometry. The pixel and geometry streams fail in opposite
    directions, so combining them is more robust across camera framings than
    either alone.

``cnn_lstm``
    The pixel-only MobileNetV2 + BiLSTM model. Strong on close-up images, weak on
    webcam-style framing.

``landmark_mlp``
    The original MLP over 63-dim MediaPipe landmarks. Retained (never deleted) as
    a fast, dependency-light fallback and as the baseline for the evaluation
    chapter.

``wlasl_video``
    Word-level MobileNetV2 + BiLSTM trained on a top-K WLASL subset. Not used by
    ``auto`` for the alphabet webcam path — call it explicitly for clip input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

HYBRID_USER_MODEL = "models/recognition/hybrid_user_model.keras"
HYBRID_MODEL = "models/recognition/hybrid_model.keras"
CNN_LSTM_MODEL = "models/recognition/cnn_lstm_model.keras"
MLP_MODEL = "models/recognition/recognition_model.joblib"
WLASL_MODEL = "models/recognition/wlasl_video_model.keras"

BACKENDS = ("auto", "hybrid_user", "hybrid", "cnn_lstm", "landmark_mlp", "wlasl_video")

# Order used by ``auto``: most robust first, always ending at the MLP fallback.
# WLASL is excluded: it is a word-clip backend, not a per-frame alphabet model.
_AUTO_ORDER = ("hybrid_user", "hybrid", "cnn_lstm", "landmark_mlp")

_MODEL_FILES = {
    "hybrid_user": "hybrid_user_model.keras",
    "hybrid": "hybrid_model.keras",
    "cnn_lstm": "cnn_lstm_model.keras",
    "landmark_mlp": "recognition_model.joblib",
    "wlasl_video": "wlasl_video_model.keras",
}


def available_backends(model_dir: str = "models/recognition") -> list[str]:
    """Return the backends whose model artefacts exist on disk."""
    base = Path(model_dir)
    return [name for name in _AUTO_ORDER if (base / _MODEL_FILES[name]).exists()]


def backend_model_path(backend: str, model_dir: str = "models/recognition") -> Path:
    """Path of the artefact a named backend loads (may not exist yet)."""
    if backend not in _MODEL_FILES:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {tuple(_MODEL_FILES)}")
    return Path(model_dir) / _MODEL_FILES[backend]


def create_recognizer(
    backend: str = "auto",
    model_dir: str = "models/recognition",
    cnn_model_path: Optional[str] = None,
    mlp_model_path: Optional[str] = None,
):
    """Instantiate a recogniser.

    Args:
        backend: ``auto`` (prefer CNN+LSTM, fall back to MLP), ``cnn_lstm``, or
            ``landmark_mlp``.
        model_dir: Directory holding the model artefacts.

    Returns:
        An object exposing ``predict_frame(image_path=, bgr_image=, landmarks=)``
        and a ``backend_name`` attribute.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}")

    base = Path(model_dir)
    cnn_path = Path(cnn_model_path) if cnn_model_path else base / "cnn_lstm_model.keras"
    mlp_path = Path(mlp_model_path) if mlp_model_path else base / "recognition_model.joblib"
    hybrid_path = base / "hybrid_model.keras"
    hybrid_user_path = base / "hybrid_user_model.keras"
    wlasl_path = base / "wlasl_video_model.keras"

    def _load_hybrid():
        from modules.recognition.hybrid_predictor import HybridPredictor

        return HybridPredictor(
            model_path=str(hybrid_path),
            metadata_path=str(base / "hybrid_metadata.json"),
        )

    def _load_hybrid_user():
        from modules.recognition.hybrid_predictor import HybridPredictor

        return HybridPredictor(
            model_path=str(hybrid_user_path),
            metadata_path=str(base / "hybrid_user_metadata.json"),
            backend_name="hybrid_user",
        )

    def _load_cnn():
        from modules.recognition.cnn_lstm_predictor import CNNLSTMPredictor

        return CNNLSTMPredictor(
            model_path=str(cnn_path),
            metadata_path=str(base / "cnn_lstm_metadata.json"),
        )

    def _load_mlp():
        from modules.recognition.model import RecognitionPredictor

        return RecognitionPredictor(
            model_path=str(mlp_path),
            scaler_path=str(base / "scaler.joblib"),
            encoder_path=str(base / "label_encoder.joblib"),
        )

    def _load_wlasl():
        from modules.recognition.wlasl_predictor import WLASLPredictor

        return WLASLPredictor(
            model_path=str(wlasl_path),
            metadata_path=str(base / "wlasl_video_metadata.json"),
        )

    loaders = {
        "hybrid_user": _load_hybrid_user,
        "hybrid": _load_hybrid,
        "cnn_lstm": _load_cnn,
        "landmark_mlp": _load_mlp,
        "wlasl_video": _load_wlasl,
    }
    paths = {
        "hybrid_user": hybrid_user_path,
        "hybrid": hybrid_path,
        "cnn_lstm": cnn_path,
        "landmark_mlp": mlp_path,
        "wlasl_video": wlasl_path,
    }

    if backend != "auto":
        return loaders[backend]()

    for name in _AUTO_ORDER:
        if not paths[name].exists():
            continue
        try:
            return loaders[name]()
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[recognition] {name} unavailable ({exc}); trying next backend.")

    raise FileNotFoundError(
        f"No recogniser model found in {model_dir}. Train one with "
        "scripts/train_hybrid.py, scripts/train_cnn_lstm.py or scripts/train_recognition.py."
    )
