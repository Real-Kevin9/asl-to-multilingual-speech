from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from modules.evaluation.runner import EvaluationRunner
from pipeline.input import PipelineInput
from pipeline.pipeline import ASLPipeline, assemble_text


class PrototypePipeline:
    """Full project prototype — every phase wired with explicit I/O.

    Phases (matching the interim report Gantt):
      1. Input capture
      2. Preprocessing (MediaPipe landmarks)
      3. Recognition (sign labels)
      4. Gloss assembly + NLP grammar correction
      5. Emotion detection → prosody
      6. Multilingual TTS (English + Nepali)
      7. Evaluation & logging (stub metrics)
    """

    PHASES = (
        "input",
        "preprocessing",
        "recognition",
        "nlp",
        "emotion",
        "tts",
        "evaluation",
    )

    def __init__(
        self,
        use_nlp_model: bool = False,
        tts_output_dir: str = "logs/tts",
        eval_log_dir: str = "logs/evaluation",
        recognition_backend: str = "auto",
    ):
        self._core = ASLPipeline(
            use_nlp_model=use_nlp_model,
            tts_output_dir=tts_output_dir,
            recognition_backend=recognition_backend,
        )
        self._evaluator = EvaluationRunner(log_dir=eval_log_dir)

    def close(self) -> None:
        self._core.close()

    def __enter__(self) -> "PrototypePipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _phase(self, name: str, status: str, input_data: Any, output_data: Any, ms: float) -> Dict[str, Any]:
        return {
            "phase": name,
            "status": status,
            "input": input_data,
            "output": output_data,
            "duration_ms": round(ms, 2),
        }

    def run(
        self,
        inp: PipelineInput,
        languages: Sequence[str] = ("en", "ne"),
    ) -> Dict[str, Any]:
        """Execute all phases for the given input and return structured results."""
        phases: List[Dict[str, Any]] = []
        timings: Dict[str, float] = {}
        t_total = time.perf_counter()

        # --- Phase 1: Input capture ---
        t = time.perf_counter()
        input_summary = {
            "mode": inp.mode,
            "image_count": len(inp.image_paths),
            "image_paths": inp.image_paths[:5] + (["..."] if len(inp.image_paths) > 5 else []),
            "gloss": inp.gloss,
            "text": inp.text,
            "face_image": inp.face_image,
        }
        phases.append(self._phase("input", "ok", None, input_summary, (time.perf_counter() - t) * 1000))
        timings["input_ms"] = phases[-1]["duration_ms"]

        labels: List[str] = []
        confidences: List[Optional[float]] = []
        landmarks_preview: Optional[List[float]] = None
        gloss = inp.gloss or inp.text or ""

        # --- Phase 2 & 3: Preprocessing + Recognition (skipped for gloss/text modes) ---
        if inp.mode in ("images", "spell") and inp.image_paths:
            t = time.perf_counter()
            landmarks_list = []
            hands_found = 0
            for path in inp.image_paths:
                lm = self._core.preprocessor.extract_landmarks(path)
                landmarks_list.append(lm)
                if lm is not None and lm.any():
                    hands_found += 1
            if landmarks_list:
                landmarks_preview = [round(float(v), 4) for v in landmarks_list[0][:6]]
            phases.append(self._phase(
                "preprocessing", "ok",
                {"frames": len(inp.image_paths)},
                {
                    "landmark_dim": 63,
                    "hands_detected": f"{hands_found}/{len(inp.image_paths)}",
                    "sample_preview": landmarks_preview,
                },
                (time.perf_counter() - t) * 1000,
            ))
            timings["preprocessing_ms"] = phases[-1]["duration_ms"]

            t = time.perf_counter()
            # Reuse the landmarks extracted above instead of running MediaPipe twice.
            rec = self._core.recognize(inp.image_paths, landmarks_list=landmarks_list)
            labels = rec["labels"]
            confidences = rec["confidences"]
            phases.append(self._phase(
                "recognition", "ok",
                {"frames": len(inp.image_paths), "backend": rec.get("backend")},
                {"labels": labels, "confidences": confidences, "backend": rec.get("backend")},
                (time.perf_counter() - t) * 1000,
            ))
            timings["recognition_ms"] = phases[-1]["duration_ms"]

            t = time.perf_counter()
            gloss = assemble_text(labels)
            phases[-1]["output"]["assembled_gloss"] = gloss  # attach to recognition for traceability
            timings["gloss_ms"] = (time.perf_counter() - t) * 1000

        elif inp.mode == "gloss":
            phases.append(self._phase("preprocessing", "skipped", None, {"reason": "gloss input"}, 0))
            phases.append(self._phase("recognition", "skipped", None, {"reason": "gloss input"}, 0))
            timings["preprocessing_ms"] = 0
            timings["recognition_ms"] = 0
            timings["gloss_ms"] = 0

        elif inp.mode == "text":
            phases.append(self._phase("preprocessing", "skipped", None, {"reason": "text input"}, 0))
            phases.append(self._phase("recognition", "skipped", None, {"reason": "text input"}, 0))
            gloss = inp.text or ""
            timings["preprocessing_ms"] = 0
            timings["recognition_ms"] = 0
            timings["gloss_ms"] = 0

        # --- Phase 4: NLP correction ---
        t = time.perf_counter()
        english = self._core.corrector.correct(gloss) if gloss else ""
        phases.append(self._phase(
            "nlp", "ok" if gloss else "empty",
            {"gloss": gloss},
            {"english": english, "backend": self._core.corrector.backend},
            (time.perf_counter() - t) * 1000,
        ))
        timings["nlp_ms"] = phases[-1]["duration_ms"]

        # --- Phase 5: Emotion ---
        t = time.perf_counter()
        face = inp.face_image or (inp.image_paths[0] if inp.image_paths else None)
        emo = self._core.emotion.detect(face)
        phases.append(self._phase(
            "emotion", "ok",
            {"face_image": face},
            emo,
            (time.perf_counter() - t) * 1000,
        ))
        timings["emotion_ms"] = phases[-1]["duration_ms"]

        # --- Phase 6: TTS ---
        t = time.perf_counter()
        if english:
            speech, tts_timings = self._core.synthesizer.synthesize(
                english,
                prosody=emo["prosody"],
                languages=languages,
                return_timings=True,
            )
            timings["tts_translation_ms"] = tts_timings.get("translation_ms")
            timings["tts_en_ms"] = tts_timings.get("en_ms")
            timings["tts_ne_ms"] = tts_timings.get("ne_ms")
            timings["tts_cache_hits"] = tts_timings.get("cache_hits")
        else:
            speech = {}
        phases.append(self._phase(
            "tts", "ok" if english else "empty",
            {"english": english, "languages": list(languages), "prosody": emo["prosody"]},
            speech,
            (time.perf_counter() - t) * 1000,
        ))
        timings["tts_ms"] = phases[-1]["duration_ms"]

        timings["total_ms"] = (time.perf_counter() - t_total) * 1000

        pipeline_result = {
            "input_mode": inp.mode,
            "recognition_backend": (
                self._core.backend_name if inp.mode in ("images", "spell") and inp.image_paths
                else "n/a"
            ),
            "recognized_labels": labels,
            "confidences": confidences,
            "gloss": gloss,
            "english": english,
            "emotion": emo["emotion"],
            "emotion_confidence": emo["confidence"],
            "speech": speech,
            "timings": timings,
            "phases": phases,
        }

        # --- Phase 7: Evaluation (stub) ---
        t = time.perf_counter()
        eval_report = self._evaluator.evaluate(pipeline_result)
        phases.append(self._phase(
            "evaluation", "ok",
            {"pipeline_timings": timings},
            eval_report,
            (time.perf_counter() - t) * 1000,
        ))
        timings["evaluation_ms"] = phases[-1]["duration_ms"]
        pipeline_result["phases"] = phases
        pipeline_result["evaluation"] = eval_report

        return pipeline_result
