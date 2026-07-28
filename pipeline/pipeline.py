from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Control tokens produced by the ASL-alphabet recognizer.
_SPACE_TOKENS = {"space"}
_DELETE_TOKENS = {"del"}
_NOTHING_TOKENS = {"nothing"}


def assemble_text(labels: Sequence[str], debounce: bool = True) -> str:
    """Assemble a sequence of per-frame sign labels into gloss text.

    Handles the ASL-alphabet control signs:
      * ``space``   -> word break
      * ``del``     -> delete the previous character
      * ``nothing`` -> ignored

    Args:
        labels: Ordered per-frame predicted labels (e.g. ["H", "H", "E", "L", ...]).
        debounce: If True, collapse runs of the same label so a sign held across
            several frames is only emitted once (crude but effective for a webcam).

    Returns:
        The assembled string, e.g. "HELLO WORLD".
    """
    chars: List[str] = []
    prev: Optional[str] = None

    for label in labels:
        if debounce and label == prev:
            continue
        prev = label

        if label in _NOTHING_TOKENS:
            continue
        if label in _DELETE_TOKENS:
            if chars:
                chars.pop()
            continue
        if label in _SPACE_TOKENS:
            chars.append(" ")
            continue
        chars.append(label)

    return "".join(chars).strip()


class ASLPipeline:
    """End-to-end Sign -> Multilingual Speech pipeline (outline implementation).

    Stages, each independently swappable:
        1. Recognition   : image frames -> per-frame sign labels (MediaPipe + MLP)
        2. Gloss assembly : labels -> gloss text
        3. NLP correction : gloss text -> grammatical English sentence (T5/rules)
        4. Emotion        : face image -> emotion + TTS prosody
        5. TTS            : English sentence -> English + Nepali audio

    Components are loaded lazily so importing the pipeline is cheap and missing
    artifacts (e.g. an untrained recognizer) don't prevent the other stages from
    being exercised.
    """

    def __init__(
        self,
        recognition_model_path: str = "models/recognition/recognition_model.joblib",
        scaler_path: str = "models/recognition/scaler.joblib",
        encoder_path: str = "models/recognition/label_encoder.joblib",
        hand_model_path: str = "models/recognition/hand_landmarker.task",
        emotion_model_path: Optional[str] = None,
        use_nlp_model: bool = False,
        tts_output_dir: str = "logs/tts",
    ):
        self.recognition_model_path = recognition_model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        self.hand_model_path = hand_model_path if Path(hand_model_path).exists() else None
        self.emotion_model_path = emotion_model_path
        self.use_nlp_model = use_nlp_model
        self.tts_output_dir = tts_output_dir

        self._predictor = None
        self._preprocessor = None
        self._corrector = None
        self._emotion = None
        self._synth = None

    # -- lazy component accessors -------------------------------------------------
    @property
    def preprocessor(self):
        if self._preprocessor is None:
            from modules.recognition.preprocess import ASLPreprocessor

            self._preprocessor = ASLPreprocessor(model_asset_path=self.hand_model_path)
        return self._preprocessor

    @property
    def predictor(self):
        if self._predictor is None:
            from modules.recognition.model import RecognitionPredictor

            self._predictor = RecognitionPredictor(
                model_path=self.recognition_model_path,
                scaler_path=self.scaler_path,
                encoder_path=self.encoder_path,
            )
        return self._predictor

    @property
    def corrector(self):
        if self._corrector is None:
            from modules.nlp.correction import GrammarCorrector

            self._corrector = GrammarCorrector(use_model=self.use_nlp_model)
        return self._corrector

    @property
    def emotion(self):
        if self._emotion is None:
            from modules.emotion.detector import EmotionDetector

            self._emotion = EmotionDetector(model_path=self.emotion_model_path)
        return self._emotion

    @property
    def synthesizer(self):
        if self._synth is None:
            from modules.tts.synthesizer import MultilingualSynthesizer

            self._synth = MultilingualSynthesizer(output_dir=self.tts_output_dir)
        return self._synth

    def close(self) -> None:
        """Release held resources (currently the MediaPipe landmarker)."""
        if self._preprocessor is not None:
            self._preprocessor.close()

    # -- stages -------------------------------------------------------------------
    def recognize(self, image_paths: Sequence[str]) -> Dict[str, Any]:
        labels: List[str] = []
        confidences: List[Optional[float]] = []
        for path in image_paths:
            landmarks = self.preprocessor.extract_landmarks(str(path))
            pred = self.predictor.predict(landmarks)
            labels.append(pred["label"])
            confidences.append(pred["confidence"])
        return {"labels": labels, "confidences": confidences}

    def process(
        self,
        image_paths: Sequence[str],
        face_image: Optional[str] = None,
        languages: Sequence[str] = ("en", "ne"),
    ) -> Dict[str, Any]:
        """Run the full pipeline over a sequence of sign frames.

        Args:
            image_paths: Ordered image paths, each showing one static sign.
            face_image: Optional image used for emotion detection. Defaults to
                the first frame if not provided.
            languages: Output languages for TTS.

        Returns:
            A structured result dict with per-stage outputs and timings (ms).
        """
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        rec = self.recognize(image_paths)
        timings["recognition_ms"] = (time.perf_counter() - t0) * 1000

        t = time.perf_counter()
        gloss = assemble_text(rec["labels"])
        timings["gloss_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        english = self.corrector.correct(gloss) if gloss else ""
        timings["nlp_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        emo = self.emotion.detect(face_image or (image_paths[0] if image_paths else None))
        timings["emotion_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        speech = (
            self.synthesizer.synthesize(english, prosody=emo["prosody"], languages=languages)
            if english
            else {}
        )
        timings["tts_ms"] = (time.perf_counter() - t) * 1000

        timings["total_ms"] = (time.perf_counter() - t0) * 1000

        return {
            "recognized_labels": rec["labels"],
            "confidences": rec["confidences"],
            "gloss": gloss,
            "english": english,
            "emotion": emo["emotion"],
            "emotion_confidence": emo["confidence"],
            "speech": speech,
            "timings": timings,
        }


def run_demo(input_text: str) -> Dict[str, Any]:
    """Lightweight text-only demo of the NLP -> TTS tail of the pipeline.

    Kept for backwards compatibility and quick smoke checks that don't require a
    trained recognizer or the MediaPipe model.
    """
    from modules.emotion.detector import EMOTION_PROSODY
    from modules.nlp.correction import correct_text
    from modules.tts.synthesizer import synthesize

    english = correct_text(input_text)
    speech = synthesize(english, prosody=EMOTION_PROSODY["neutral"])
    return {
        "input_text": input_text,
        "status": "ready",
        "steps": ["recognition", "nlp", "tts"],
        "english": english,
        "speech": speech,
        "message": "Pipeline scaffolding is working.",
    }
