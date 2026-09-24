from __future__ import annotations

import base64
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from modules.emotion.detector import EmotionDetector
from modules.nlp.correction import GrammarCorrector
from modules.recognition.hand_crop import landmark_bbox
from modules.recognition.preprocess import ASLPreprocessor
from modules.recognition.registry import create_recognizer
from modules.recognition.stabilizer import SignStabilizer
from modules.recognition.wlasl_live import ClipRecorder, assemble_word_gloss
from modules.tts.synthesizer import MultilingualSynthesizer
from pipeline.model_selection import resolve_models, resolve_nlp_use_model
from pipeline.pipeline import assemble_text

REPO_ROOT = Path(__file__).resolve().parents[2]
HAND_MODEL = REPO_ROOT / "models" / "recognition" / "hand_landmarker.task"
WLASL_MODEL = REPO_ROOT / "models" / "recognition" / "wlasl_video_model.keras"
WLASL_LANDMARK_PT = REPO_ROOT / "models" / "recognition" / "wlasl_landmark_transformer.pt"


def _decode_frame(image_data: str) -> Optional[np.ndarray]:
    if not image_data:
        return None
    payload = image_data.split(",", 1)[-1]
    try:
        raw = base64.b64decode(payload)
    except (ValueError, TypeError):
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def _looks_nepali(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in text)


@dataclass
class RealtimeSession:
    """One browser connection: webcam frames in, recognition state out."""

    backend: str = "auto"
    recognition_mode: str = "alphabet"  # alphabet | wlasl
    use_nlp_model: bool = False
    hold_time: float = 0.9
    min_confidence: float = 0.6
    agreement: float = 0.65
    cooldown: float = 0.5
    predict_interval: float = 0.12
    languages: Tuple[str, ...] = ("en", "ne")

    labels: List[str] = field(default_factory=list)
    last_label: str = ""
    last_conf: float = 0.0
    last_box: Optional[Tuple[int, int, int, int]] = None
    last_frame: Optional[np.ndarray] = None
    next_predict_at: float = 0.0
    recording: bool = False
    record_frames: int = 0
    record_seconds: float = 0.0
    notice: str = ""

    _recognizer: Any = field(default=None, repr=False)
    _wlasl: Any = field(default=None, repr=False)
    _preprocessor: Any = field(default=None, repr=False)
    _emotion: Any = field(default=None, repr=False)
    _corrector: Any = field(default=None, repr=False)
    _synth: Any = field(default=None, repr=False)
    _stabilizer: Any = field(default=None, repr=False)
    _recorder: Any = field(default=None, repr=False)
    _loaded: bool = field(default=False, repr=False)
    _sign_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def load(self) -> Dict[str, Any]:
        if self._loaded:
            return self.model_info()

        mode = (self.recognition_mode or "alphabet").lower()
        if mode not in ("alphabet", "wlasl"):
            raise ValueError(f"Unknown recognition_mode {mode!r}")
        self.recognition_mode = mode

        if mode == "wlasl":
            if not (WLASL_MODEL.exists() or WLASL_LANDMARK_PT.exists()):
                raise FileNotFoundError(
                    "WLASL model missing. Run scripts/extract_wlasl_landmarks.py and "
                    "scripts/train_wlasl_v3.py (or scripts/train_wlasl.py) first."
                )
            from modules.recognition.wlasl_predictor import WLASLPredictor

            self._wlasl = WLASLPredictor(min_confidence=0.12)
            self._recognizer = self._wlasl
            self._recorder = ClipRecorder(target_fps=12.0, max_seconds=4.0)
        else:
            self._recognizer = create_recognizer(backend=self.backend)
            self._preprocessor = ASLPreprocessor(
                model_asset_path=str(HAND_MODEL) if HAND_MODEL.exists() else None
            )
            self._stabilizer = SignStabilizer(
                hold_time=self.hold_time,
                min_confidence=self.min_confidence,
                agreement=self.agreement,
                cooldown=self.cooldown,
            )

        self._emotion = EmotionDetector()
        self._corrector = GrammarCorrector(use_model=self.use_nlp_model)
        self._synth = MultilingualSynthesizer()
        self._loaded = True
        return self.model_info()

    def model_info(self) -> Dict[str, Any]:
        backend_name = getattr(self._recognizer, "backend_name", self.backend)
        selection = resolve_models(
            recognition_backend="wlasl_video" if self.recognition_mode == "wlasl" else self.backend,
            nlp_mode="on" if self.use_nlp_model else "off",
            languages=self.languages,
        )
        return {
            "recognition_mode": self.recognition_mode,
            "recognition_backend": backend_name,
            "nlp_backend": getattr(self._corrector, "backend", "rule_based"),
            "emotion_backend": getattr(self._emotion, "backend", "fallback_neutral"),
            "tts_backend": "gtts_cached",
            "selection": selection.as_dict(),
        }

    def clear(self) -> None:
        self.labels = []
        self.notice = ""
        if self._stabilizer is not None:
            self._stabilizer.reset()
        if self._recorder is not None:
            self._recorder.cancel()
        self.recording = False
        self.record_frames = 0
        self.record_seconds = 0.0

    def close(self) -> None:
        if self._preprocessor is not None:
            self._preprocessor.close()

    def _gloss(self) -> str:
        if self.recognition_mode == "wlasl":
            return assemble_word_gloss(self.labels)
        return assemble_text(self.labels, debounce=False)

    def start_sign(self) -> Dict[str, Any]:
        if not self._loaded:
            self.load()
        if self.recognition_mode != "wlasl":
            return self.snapshot(error="start_sign is only used in WLASL word mode.")
        self._recorder.start()
        self.recording = True
        self.record_frames = 0
        self.record_seconds = 0.0
        self.notice = "Recording… perform one word sign, then press Stop."
        return self.snapshot()

    def cancel_sign(self) -> Dict[str, Any]:
        if self._recorder is not None:
            self._recorder.cancel()
        self.recording = False
        self.record_frames = 0
        self.record_seconds = 0.0
        self.notice = "Recording cancelled."
        return self.snapshot()

    def commit_sign(self) -> Dict[str, Any]:
        with self._sign_lock:
            if not self._loaded:
                self.load()
            if self.recognition_mode != "wlasl":
                return self.snapshot(error="commit_sign is only used in WLASL word mode.")
            frames = self._recorder.stop() if self._recorder is not None else []
            self.recording = False
            self.record_frames = 0
            self.record_seconds = 0.0
            if len(frames) < 4:
                self.notice = "Recording too short — hold Record longer while signing."
                return self.snapshot()
            try:
                result = self._wlasl.predict_clip(frames_bgr=frames)
            except Exception as exc:
                self.notice = str(exc)
                return self.snapshot(error=str(exc))
            label = result["label"]
            conf = float(result.get("confidence") or 0.0)
            if result.get("rejected") or label == "unknown":
                self.notice = (
                    f"Low confidence ({conf:.0%}"
                    f"{', raw=' + str(result.get('raw_label')) if result.get('raw_label') else ''}"
                    "). Try recording again with the signer centred."
                )
                self.last_label = str(result.get("raw_label") or "unknown")
                self.last_conf = conf
                return self.snapshot()
            self.labels.append(label)
            self.last_label = label
            self.last_conf = conf
            top = result.get("top5") or []
            top_txt = ", ".join(f"{t['label']}:{t['confidence']:.0%}" for t in top[:3])
            self.notice = f"Committed “{label}” ({conf:.0%}). Top: {top_txt}"
            return self.snapshot()

    def process_frame(self, image_data: str) -> Dict[str, Any]:
        if not self._loaded:
            self.load()

        frame = _decode_frame(image_data)
        if frame is None:
            return self.snapshot(error="Could not decode camera frame.")

        self.last_frame = frame
        now = time.monotonic()
        h, w = frame.shape[:2]

        if self.recognition_mode == "wlasl":
            if self._recorder is not None and self._recorder.recording:
                still = self._recorder.add_frame(frame, now=now)
                self.recording = True
                self.record_frames = self._recorder.frame_count
                self.record_seconds = round(self._recorder.duration(now), 2)
                if not still:
                    # Auto-commit at max duration.
                    return self.commit_sign()
            return self.snapshot()

        if now >= self.next_predict_at:
            self.next_predict_at = now + self.predict_interval
            landmarks = self._preprocessor.extract_landmarks_from_array(frame)
            pred = self._recognizer.predict_frame(bgr_image=frame, landmarks=landmarks)
            self.last_label = pred["label"]
            self.last_conf = float(pred.get("confidence") or 0.0)
            self.last_box = landmark_bbox(landmarks, w, h)

            committed = self._stabilizer.update(self.last_label, self.last_conf, now=now)
            if committed is not None:
                self.labels.append(committed)

        candidate, progress = self._stabilizer.status(now=time.monotonic())
        gloss = self._gloss()

        bbox_pct = None
        if self.last_box:
            x1, y1, x2, y2 = self.last_box
            bbox_pct = [
                round(100 * x1 / w, 2),
                round(100 * y1 / h, 2),
                round(100 * x2 / w, 2),
                round(100 * y2 / h, 2),
            ]

        return {
            "recognition_mode": self.recognition_mode,
            "current_label": self.last_label,
            "confidence": round(self.last_conf, 3),
            "backend": getattr(self._recognizer, "backend_name", self.backend),
            "assembled_gloss": gloss,
            "holding_label": candidate or "",
            "hold_progress": round(progress, 3),
            "bbox_pct": bbox_pct,
            "committed_labels": list(self.labels),
            "recording": False,
            "record_frames": 0,
            "record_seconds": 0.0,
            "notice": self.notice,
        }

    def snapshot(self, error: Optional[str] = None) -> Dict[str, Any]:
        gloss = self._gloss()
        if self.recognition_mode == "wlasl":
            candidate, progress = None, 0.0
        else:
            candidate, progress = (
                self._stabilizer.status(now=time.monotonic())
                if self._stabilizer is not None
                else (None, 0.0)
            )
        payload = {
            "recognition_mode": self.recognition_mode,
            "current_label": self.last_label,
            "confidence": round(self.last_conf, 3),
            "backend": getattr(self._recognizer, "backend_name", self.backend),
            "assembled_gloss": gloss,
            "holding_label": candidate or "",
            "hold_progress": round(progress, 3),
            "bbox_pct": None,
            "committed_labels": list(self.labels),
            "recording": self.recording,
            "record_frames": self.record_frames,
            "record_seconds": self.record_seconds,
            "notice": self.notice,
        }
        if error:
            payload["error"] = error
        return payload

    def finish(self) -> Dict[str, Any]:
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()
        gloss = self._gloss()
        english = self._corrector.correct(gloss) if gloss else ""

        t_nlp = time.perf_counter()
        emo = self._emotion.detect(self.last_frame)
        t_emo = time.perf_counter()

        speech: Dict[str, Any] = {}
        tts_timings: Dict[str, Any] = {}
        if english:
            speech, tts_timings = self._synth.synthesize(
                english,
                prosody=emo["prosody"],
                languages=self.languages,
                return_timings=True,
            )

        total_ms = (time.perf_counter() - t0) * 1000
        self.clear()

        audio_urls = {}
        notices: List[str] = []
        for lang, info in speech.items():
            path = info.get("audio_path")
            if path:
                audio_urls[lang] = _audio_url(path)

        nepali_info = speech.get("ne") if isinstance(speech.get("ne"), dict) else None
        if nepali_info:
            nepali_text = str(nepali_info.get("text") or "")
            if nepali_text.strip().lower() == english.strip().lower() and not _looks_nepali(nepali_text):
                audio_urls.pop("ne", None)
                notices.append("Nepali translation was unavailable, so Nepali audio was not generated.")
            elif str(nepali_info.get("audio_format") or "").lower() == "wav":
                notices.append("Nepali audio is using the local fallback voice, so pronunciation may sound rough.")

        english_info = speech.get("en") if isinstance(speech.get("en"), dict) else None
        if english_info and str(english_info.get("audio_format") or "").lower() == "wav":
            notices.append("English audio is using the local fallback voice because Google TTS was unavailable.")

        return {
            "gloss": gloss,
            "english": english,
            "emotion": emo["emotion"],
            "emotion_confidence": emo["confidence"],
            "speech": speech,
            "audio_urls": audio_urls,
            "notices": notices,
            "timings": {
                "nlp_ms": round((t_nlp - t0) * 1000, 1),
                "emotion_ms": round((t_emo - t_nlp) * 1000, 1),
                "tts_ms": float(tts_timings.get("total_ms") or 0.0),
                "total_ms": round(total_ms, 1),
            },
        }


def _audio_url(path: str) -> str:
    p = Path(path)
    try:
        rel = p.resolve().relative_to((REPO_ROOT / "logs" / "tts").resolve())
    except ValueError:
        return ""
    return f"/api/audio/{rel.as_posix()}"


def parse_nlp_mode(raw: Optional[str], use_nlp: Optional[bool]) -> bool:
    if use_nlp is not None:
        return bool(use_nlp)
    if raw in (None, "", "auto", "on"):
        return bool(resolve_nlp_use_model("auto"))
    return raw == "on"


_SAFE_AUDIO = re.compile(r"^[a-zA-Z0-9_./-]+$")


def safe_audio_path(relative: str) -> Optional[Path]:
    if not relative or ".." in relative:
        return None
    if not _SAFE_AUDIO.match(relative):
        return None
    root = (REPO_ROOT / "logs" / "tts").resolve()
    candidate = (root / relative).resolve()
    if not str(candidate).startswith(str(root)):
        return None
    if candidate.suffix.lower() not in {".mp3", ".wav"}:
        return None
    if not candidate.is_file():
        return None
    return candidate
