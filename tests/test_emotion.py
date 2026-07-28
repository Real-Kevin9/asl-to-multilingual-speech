import numpy as np

from modules.emotion.detector import EMOTION_PROSODY, EmotionDetector


def test_emotion_detector_fallback_neutral():
    detector = EmotionDetector()
    # A blank frame has no face; detector should default to neutral and not crash.
    blank = np.zeros((120, 120, 3), dtype=np.uint8)
    result = detector.detect(blank)
    assert result["emotion"] in EMOTION_PROSODY
    assert result["emotion"] == "neutral"
    assert result["face_found"] is False
    assert result["prosody"] == EMOTION_PROSODY["neutral"]
