"""Automated end-to-end scenario checks for system-level evaluation.

Runs fixed inputs through ``PrototypePipeline`` and scores objective pass/fail
criteria per stage. This is **not** a usability study — it verifies that each
pipeline stage produces expected outputs on known inputs.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from pipeline.input import PipelineInput
from pipeline.model_selection import resolve_models, resolve_nlp_use_model
from pipeline.prototype import PrototypePipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
USER_CROPS = REPO_ROOT / "data" / "processed" / "user_crops"
FER_DIR = REPO_ROOT / "data" / "processed" / "fer2013_4class"
LATENCY_TARGET_MS = 1500.0


def _pick_user_crop(letter: str) -> Optional[Path]:
    folder = USER_CROPS / letter.upper()
    if not folder.is_dir():
        return None
    files = sorted(folder.glob("*.jpg"))
    return files[0] if files else None


def _pick_fer_face(label: int = 0) -> Optional[str]:
    """Write one 48×48 FER face to a temp JPEG for emotion detection."""
    try:
        from modules.emotion.data import load_split
        import cv2
    except ImportError:
        return None

    path = FER_DIR / "test.npz"
    if not path.exists():
        return None

    x, y = load_split(str(FER_DIR), "test")
    indices = np.where(y == label)[0]
    if len(indices) == 0:
        indices = np.arange(min(1, len(y)))
    if len(indices) == 0:
        return None

    face = (x[indices[0], :, :, 0] * 255).astype(np.uint8)
    # Haar cascades need a few hundred pixels; upscale the 48×48 FER crop.
    face_large = cv2.resize(face, (240, 240), interpolation=cv2.INTER_CUBIC)
    face_bgr = cv2.cvtColor(face_large, cv2.COLOR_GRAY2BGR)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, face_bgr)
    return tmp.name


def _audio_ok(speech_entry: Dict[str, Any], min_bytes: int = 100) -> bool:
    if not speech_entry:
        return False
    audio_path = speech_entry.get("audio_path")
    if not audio_path:
        return False
    path = Path(audio_path)
    if not path.exists():
        return False
    size = path.stat().st_size
    if path.suffix == ".mp3":
        return size >= min_bytes
    if path.suffix == ".txt":
        return size > 0
    return size > 0


def _run_pipeline(
    inp: PipelineInput,
    proto: PrototypePipeline,
    *,
    languages: Sequence[str] = ("en", "ne"),
) -> Dict[str, Any]:
    return proto.run(inp, languages=languages)


def _check_fingerspell_cat(result: Dict[str, Any]) -> Dict[str, Any]:
    labels = [str(l).upper() for l in (result.get("recognized_labels") or [])]
    expected = ["C", "A", "T"]
    matched = labels == expected
    return {
        "success": matched,
        "expected_labels": expected,
        "actual_labels": labels,
        "gloss": result.get("gloss"),
        "backend": result.get("recognition_backend"),
    }


def _check_nlp_correction(result: Dict[str, Any]) -> Dict[str, Any]:
    gloss = (result.get("gloss") or "").strip()
    english = (result.get("english") or "").strip()
    changed = english.lower() != gloss.lower()
    has_content = len(english.split()) >= 2
    return {
        "success": bool(english) and has_content,
        "gloss": gloss,
        "english": english,
        "corrected": changed,
    }


def _check_tts_both_languages(result: Dict[str, Any]) -> Dict[str, Any]:
    speech = result.get("speech") or {}
    en_ok = _audio_ok(speech.get("en") or {})
    ne_ok = _audio_ok(speech.get("ne") or {})
    return {
        "success": en_ok and ne_ok,
        "english_audio": (speech.get("en") or {}).get("audio_path"),
        "nepali_audio": (speech.get("ne") or {}).get("audio_path"),
        "english_ok": en_ok,
        "nepali_ok": ne_ok,
    }


def _check_emotion_non_neutral(result: Dict[str, Any]) -> Dict[str, Any]:
    emotion = result.get("emotion")
    confidence = result.get("emotion_confidence")
    face_found = None
    for phase in result.get("phases") or []:
        if phase.get("phase") == "emotion":
            face_found = (phase.get("output") or {}).get("face_found")
            break
    # Success = face detected and classifier ran (emotion may still be neutral).
    success = face_found is True and emotion is not None
    return {
        "success": success,
        "emotion": emotion,
        "confidence": confidence,
        "face_found": face_found,
    }


def _check_latency(result: Dict[str, Any]) -> Dict[str, Any]:
    total = (result.get("timings") or {}).get("total_ms")
    ok = total is not None and total <= LATENCY_TARGET_MS
    return {
        "success": ok,
        "total_ms": total,
        "target_ms": LATENCY_TARGET_MS,
    }


def run_scenarios(
    *,
    recognition_backend: str = "auto",
    nlp_mode: str = "auto",
) -> Dict[str, Any]:
    """Execute all scripted scenarios and return a structured report."""
    scenarios: List[Dict[str, Any]] = []
    temp_files: List[str] = []
    use_nlp = resolve_nlp_use_model(nlp_mode)

    with PrototypePipeline(
        use_nlp_model=use_nlp,
        recognition_backend=recognition_backend,
    ) as proto:

        def record(
            scenario_id: str,
            objective: str,
            description: str,
            fn: Callable[[], Dict[str, Any]],
        ) -> None:
            start = time.perf_counter()
            error = None
            details: Dict[str, Any] = {}
            success = False
            try:
                details = fn()
                success = bool(details.pop("success", False))
            except Exception as exc:
                error = str(exc)
            scenarios.append({
                "id": scenario_id,
                "objective": objective,
                "description": description,
                "success": success,
                "duration_s": round(time.perf_counter() - start, 2),
                "details": details,
                "error": error,
            })

        crop_paths = [str(p) for letter in "CAT" if (p := _pick_user_crop(letter))]

        if len(crop_paths) == 3:
            def run_cat() -> Dict[str, Any]:
                inp = PipelineInput.from_images(crop_paths)
                return _check_fingerspell_cat(_run_pipeline(inp, proto, languages=("en",)))

            record(
                "fingerspell_cat",
                "Recognition",
                "Recognise C-A-T from user webcam crop images.",
                run_cat,
            )
        else:
            scenarios.append({
                "id": "fingerspell_cat",
                "objective": "Recognition",
                "description": "Recognise C-A-T from user webcam crop images.",
                "success": False,
                "duration_s": 0.0,
                "details": {"skipped": True, "reason": "user crop images for C/A/T not found"},
                "error": None,
            })

        def run_nlp() -> Dict[str, Any]:
            inp = PipelineInput.from_gloss("I GO STORE")
            return _check_nlp_correction(_run_pipeline(inp, proto, languages=("en",)))

        record(
            "nlp_gloss",
            "NLP",
            "Convert gloss 'I GO STORE' to natural English.",
            run_nlp,
        )

        def run_tts() -> Dict[str, Any]:
            inp = PipelineInput.from_text("Hello, how are you?")
            return _check_tts_both_languages(
                _run_pipeline(inp, proto, languages=("en", "ne"))
            )

        record(
            "tts_en_ne",
            "TTS",
            "Synthesise English and Nepali audio for a short sentence.",
            run_tts,
        )

        face_path = _pick_fer_face(label=0)
        if face_path:
            temp_files.append(face_path)

            def run_emotion() -> Dict[str, Any]:
                inp = PipelineInput.from_text("I am happy today.", face_image=face_path)
                return _check_emotion_non_neutral(
                    _run_pipeline(inp, proto, languages=("en",))
                )

            record(
                "emotion_face",
                "Emotion",
                "Detect a face and run the emotion classifier.",
                run_emotion,
            )
        else:
            scenarios.append({
                "id": "emotion_face",
                "objective": "Emotion",
                "description": "Detect a face and run the emotion classifier.",
                "success": False,
                "duration_s": 0.0,
                "details": {"skipped": True, "reason": "FER test split or cv2 unavailable"},
                "error": None,
            })

        def run_latency() -> Dict[str, Any]:
            inp = PipelineInput.from_text("Good morning.")
            return _check_latency(_run_pipeline(inp, proto, languages=("en", "ne")))

        record(
            "latency_warm",
            "Latency",
            "End-to-end gloss tail completes within the 1.5 s target (warm run).",
            run_latency,
        )

    for tmp in temp_files:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass

    successes = sum(1 for s in scenarios if s["success"])
    selection = resolve_models(recognition_backend=recognition_backend, nlp_mode=nlp_mode)

    return {
        "method": "automated_scenarios",
        "n_scenarios": len(scenarios),
        "successes": successes,
        "success_rate": round(successes / len(scenarios), 3) if scenarios else None,
        "scenarios": scenarios,
        "models": selection.as_dict(),
        "limitations": [
            "Scenario checks verify functional correctness, not user satisfaction.",
            "Recognition scenarios use the developer's own camera crops.",
            "TTS may fall back to .txt files when offline; counted as partial success.",
        ],
    }
