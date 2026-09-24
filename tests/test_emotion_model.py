"""Tests for emotion CNN data helpers and detector wiring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from modules.emotion.data import (
    EMOTIONS,
    FER7_TO_PROJECT,
    image_to_array,
    remap_fer_label,
)
from modules.emotion.detector import EMOTION_PROSODY, EmotionDetector
from modules.emotion.model import ARCHITECTURES, build_emotion_cnn, build_emotion_cnn_v2


def test_remap_keeps_only_four_classes():
    assert remap_fer_label(3) == EMOTIONS.index("happy")
    assert remap_fer_label(4) == EMOTIONS.index("sad")
    assert remap_fer_label(0) == EMOTIONS.index("angry")
    assert remap_fer_label(6) == EMOTIONS.index("neutral")
    assert remap_fer_label(1) is None  # disgust dropped
    assert remap_fer_label(2) is None  # fear dropped
    assert remap_fer_label(5) is None  # surprise dropped


def test_fer7_map_covers_project_emotions():
    mapped = {FER7_TO_PROJECT[k] for k in FER7_TO_PROJECT}
    assert mapped == set(range(len(EMOTIONS)))


def test_image_to_array_from_flat_list():
    flat = list(range(48 * 48))
    arr = image_to_array(flat)
    assert arr.shape == (48, 48, 1)
    assert arr.dtype == np.float32
    assert 0.0 <= float(arr.min()) and float(arr.max()) <= 1.0


def test_image_to_array_from_pixel_string():
    pixels = " ".join(str(i % 256) for i in range(48 * 48))
    arr = image_to_array(pixels)
    assert arr.shape == (48, 48, 1)


def test_build_emotion_cnn_output_shape():
    model = build_emotion_cnn()
    out = model.predict(np.zeros((2, 48, 48, 1), dtype=np.float32), verbose=0)
    assert out.shape == (2, 4)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)


def test_build_emotion_cnn_v2_output_shape():
    model = build_emotion_cnn_v2(width=8)
    out = model.predict(np.zeros((2, 48, 48, 1), dtype=np.float32), verbose=0)
    assert out.shape == (2, 4)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)


def test_v2_augmentation_is_inert_at_inference():
    # Augmentation lives inside the model, so the detector would silently jitter
    # live frames if the layers ever ran outside training.
    model = build_emotion_cnn_v2(width=8)
    rng = np.random.default_rng(0)
    face = rng.random((1, 48, 48, 1)).astype(np.float32)

    first = model.predict(face, verbose=0)
    second = model.predict(face, verbose=0)

    assert np.allclose(first, second)


def test_v2_actually_augments_during_training():
    # The mirror of the previous test: if augmentation were a no-op in training
    # too, the whole change would be inert and the run would prove nothing.
    model = build_emotion_cnn_v2(width=8)
    rng = np.random.default_rng(0)
    face = rng.random((8, 48, 48, 1)).astype(np.float32)

    outputs = [model(face, training=True).numpy() for _ in range(2)]

    assert not np.allclose(outputs[0], outputs[1])


def test_architecture_registry_exposes_both_versions():
    assert set(ARCHITECTURES) == {"v1", "v2"}
    assert ARCHITECTURES["v1"] is build_emotion_cnn
    assert ARCHITECTURES["v2"] is build_emotion_cnn_v2


def test_detector_fallback_without_model(tmp_path: Path):
    detector = EmotionDetector(model_path=str(tmp_path / "missing.keras"), auto_load=False)
    assert detector.backend == "fallback_neutral"
    blank = np.zeros((120, 120, 3), dtype=np.uint8)
    result = detector.detect(blank)
    assert result["emotion"] == "neutral"
    assert result["face_found"] is False
    assert result["prosody"] == EMOTION_PROSODY["neutral"]
    assert "backend" in result


def test_prosody_differs_across_emotions():
    rates = {e: EMOTION_PROSODY[e]["rate"] for e in EMOTIONS}
    assert rates["sad"] < rates["neutral"] < rates["happy"]
