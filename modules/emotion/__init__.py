"""Emotion detection: facial expression → TTS prosody."""

from modules.emotion.detector import EMOTION_PROSODY, EMOTIONS, EmotionDetector, detect_emotion

__all__ = ["EMOTIONS", "EMOTION_PROSODY", "EmotionDetector", "detect_emotion"]
