from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class EvaluationRunner:
    """Outline evaluation module (Phase 7).

    Records per-run metrics and writes a JSON log. Full benchmarking (accuracy,
    BLEU, latency targets, SUS user study) is planned for later; this stub keeps
    the prototype end-to-end and gives a place to plug in real evaluators.
    """

    def __init__(self, log_dir: str = "logs/evaluation"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        timings = pipeline_result.get("timings", {})
        labels = pipeline_result.get("recognized_labels") or []
        confidences = pipeline_result.get("confidences") or []

        avg_conf = None
        if confidences:
            valid = [c for c in confidences if c is not None]
            if valid:
                avg_conf = sum(valid) / len(valid)

        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "prototype_outline",
            "metrics": {
                "frames_processed": len(labels) if labels else 0,
                "gloss_length": len(pipeline_result.get("gloss") or ""),
                "english_length": len(pipeline_result.get("english") or ""),
                "avg_recognition_confidence": avg_conf,
                "total_latency_ms": timings.get("total_ms"),
                "latency_target_ms": 1500,
                "latency_ok": (timings.get("total_ms") or 9999) < 1500,
            },
            "notes": [
                "Prototype evaluation — accuracy/BLEU/SUS not yet measured.",
                "Replace with holdout tests and user-study scripts in Phase 7.",
            ],
        }

        log_path = self.log_dir / f"run_{int(time.time())}.json"
        log_path.write_text(json.dumps({**report, "pipeline_result": pipeline_result}, indent=2),
                            encoding="utf-8")
        report["log_path"] = str(log_path)
        return report
