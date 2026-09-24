from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from modules.tts.synthesizer import LATENCY_TARGET_MS


class EvaluationRunner:
    """Phase 7 evaluation — per-run metrics plus unified module snapshot.

    Each pipeline run writes a timestamped JSON log. When
    ``logs/evaluation/unified_report.json`` exists (from ``evaluate_all.py``),
    headline module metrics are attached to the run report.
    """

    def __init__(self, log_dir: str = "logs/evaluation"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _load_unified_snapshot(self) -> Optional[Dict[str, Any]]:
        path = self.log_dir / "unified_report.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return data.get("module_summary")

    def evaluate(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        timings = pipeline_result.get("timings", {})
        labels = pipeline_result.get("recognized_labels") or []
        confidences = pipeline_result.get("confidences") or []
        speech = pipeline_result.get("speech") or {}

        avg_conf = None
        if confidences:
            valid = [c for c in confidences if c is not None]
            if valid:
                avg_conf = sum(valid) / len(valid)

        tts_cache_hits = sum(
            1 for info in speech.values()
            if isinstance(info, dict) and info.get("cached")
        )
        total_ms = timings.get("total_ms")
        latency_ok = total_ms is not None and total_ms < LATENCY_TARGET_MS

        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "evaluated",
            "metrics": {
                "frames_processed": len(labels) if labels else 0,
                "gloss_length": len(pipeline_result.get("gloss") or ""),
                "english_length": len(pipeline_result.get("english") or ""),
                "avg_recognition_confidence": avg_conf,
                "recognition_ms": timings.get("recognition_ms"),
                "nlp_ms": timings.get("nlp_ms"),
                "emotion_ms": timings.get("emotion_ms"),
                "tts_ms": timings.get("tts_ms"),
                "tts_translation_ms": timings.get("tts_translation_ms"),
                "tts_en_ms": timings.get("tts_en_ms"),
                "tts_ne_ms": timings.get("tts_ne_ms"),
                "tts_cache_hits": tts_cache_hits or timings.get("tts_cache_hits"),
                "total_latency_ms": total_ms,
                "latency_target_ms": LATENCY_TARGET_MS,
                "latency_ok": latency_ok,
            },
            "module_metrics_snapshot": self._load_unified_snapshot(),
            "notes": [
                "Per-run latency from pipeline timings (ms).",
                "Module accuracy/BLEU live in logs/evaluation/unified_report.json.",
                "Run scripts/evaluate_all.py --run-latency to refresh benchmarks.",
            ],
        }

        log_path = self.log_dir / f"run_{int(time.time())}.json"
        log_path.write_text(
            json.dumps({**report, "pipeline_result": pipeline_result}, indent=2),
            encoding="utf-8",
        )
        report["log_path"] = str(log_path)
        return report
