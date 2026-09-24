from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from modules.evaluation.benchmark import run_latency_benchmark
from modules.tts.synthesizer import LATENCY_TARGET_MS

DEFAULT_LOG_DIR = "logs/evaluation"

MODULE_REPORT_PATHS = {
    "nlp": "nlp_bleu.json",
    "emotion": "emotion_accuracy.json",
    "recognition": "backend_comparison.json",
    "user_recognition": "user_data_accuracy.json",
    "wlasl": "wlasl_accuracy.json",
    "latency": "latency_benchmark.json",
    "usability": "usability_sus.json",
    "formative_usability": "formative_usability.json",
    "scenario_evaluation": "scenario_evaluation.json",
}


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_module_reports(log_dir: str = DEFAULT_LOG_DIR) -> Dict[str, Any]:
    """Load per-module JSON reports if they exist."""
    root = Path(log_dir)
    loaded: Dict[str, Any] = {}
    missing: list[str] = []
    for name, filename in MODULE_REPORT_PATHS.items():
        data = _load_json(root / filename)
        if data is None:
            missing.append(name)
        else:
            loaded[name] = data
    return {"reports": loaded, "missing": missing, "log_dir": str(root)}


def _summarize_backends(report: Dict[str, Any], grouped: bool) -> Dict[str, Any]:
    """Condense a per-backend recogniser report.

    ``evaluate_recognizers.py`` stores accuracy as a dict of image groups
    (``all`` / ``scene`` / ``closeup``); ``evaluate_on_user_data.py`` stores a
    single float. Both are keyed by backend name.
    """
    backends: Dict[str, Any] = {}
    for name, info in report.items():
        if name.startswith("_") or not isinstance(info, dict):
            continue
        accuracy = info.get("accuracy")
        if grouped and isinstance(accuracy, dict):
            counts = info.get("counts") or {}
            backends[name] = {
                "overall_accuracy": accuracy.get("all"),
                "scene_accuracy": accuracy.get("scene"),
                "closeup_accuracy": accuracy.get("closeup"),
                "n_samples": counts.get("all"),
            }
        elif isinstance(accuracy, (int, float)):
            backends[name] = {
                "overall_accuracy": accuracy,
                "n_samples": info.get("total"),
            }

    best = None
    if backends:
        scored = [
            (name, info["overall_accuracy"])
            for name, info in backends.items()
            if info.get("overall_accuracy") is not None
        ]
        if scored:
            best = max(scored, key=lambda kv: kv[1])[0]

    return {"backends": backends, "best_backend": best}


def _extract_module_summary(reports: Dict[str, Any]) -> Dict[str, Any]:
    """Pull headline numbers from each module report."""
    summary: Dict[str, Any] = {}

    nlp = reports.get("nlp")
    if nlp:
        # evaluate_nlp.py names the neural section after the backend it loaded
        # ("t5_gloss"), so read that key rather than guessing at a fixed name.
        backend = nlp.get("backend") or "t5_finetuned"
        t5 = nlp.get(backend) or nlp.get("t5_finetuned") or nlp.get("t5") or {}
        rules = nlp.get("rule_based") or {}
        t5_bleu = t5.get("bleu")
        rule_bleu = rules.get("bleu")
        summary["nlp"] = {
            "t5_bleu": t5_bleu,
            "rule_bleu": rule_bleu,
            "delta_bleu": nlp.get("delta_bleu"),
            "n_pairs": nlp.get("n"),
            "backend": backend,
        }
        # Objective 2 is a relative target: beat the rule-based baseline by 20%.
        if t5_bleu is not None and rule_bleu:
            summary["nlp"]["relative_gain"] = round(t5_bleu / rule_bleu - 1, 3)
            summary["nlp"]["fluency_target_met"] = t5_bleu >= rule_bleu * 1.20

    emotion = reports.get("emotion")
    if emotion:
        accuracy = emotion.get("accuracy")
        target = emotion.get("target_accuracy", 0.80)
        met = emotion.get("target_met")
        if met is None and accuracy is not None:
            met = accuracy >= target
        summary["emotion"] = {
            "accuracy": accuracy,
            "n_samples": emotion.get("n") or emotion.get("n_samples"),
            "target_accuracy": target,
            "target_met": met,
        }

    # Both recogniser reports are keyed by backend name at the top level.
    rec = reports.get("recognition")
    if rec:
        summary["recognition"] = _summarize_backends(rec, grouped=True)

    user = reports.get("user_recognition")
    if user:
        summary["user_recognition"] = _summarize_backends(user, grouped=False)
        # Whether hybrid_user was scored on data it trained on changes how the
        # number should be read, so carry the basis into the summary.
        evaluation = user.get("_evaluation") or {}
        if evaluation:
            summary["user_recognition"]["subset"] = evaluation.get("subset")
            summary["user_recognition"]["n_scored"] = evaluation.get("n_scored")

    wlasl = reports.get("wlasl")
    if wlasl:
        summary["wlasl"] = {
            "accuracy": wlasl.get("accuracy"),
            "num_classes": wlasl.get("num_classes"),
            "n": wlasl.get("n"),
            "target_signs": wlasl.get("target_signs", 50),
            "target_accuracy": wlasl.get("target_accuracy", 0.85),
            "signs_met": wlasl.get("signs_met"),
            "accuracy_met": wlasl.get("accuracy_met"),
        }

    latency = reports.get("latency")
    if latency:
        gloss = latency.get("gloss_pipeline") or {}
        tts = latency.get("tts") or {}
        summary["latency"] = {
            "gloss_total_p95_ms": (gloss.get("total") or {}).get("p95_ms"),
            "gloss_total_mean_ms": (gloss.get("total") or {}).get("mean_ms"),
            "gloss_warmup_ms": gloss.get("warmup_ms"),
            "tts_cold_p95_ms": (tts.get("cold") or {}).get("p95_ms"),
            "tts_warm_first_p95_ms": (tts.get("warm_first") or {}).get("p95_ms"),
            "tts_warm_repeat_p95_ms": (tts.get("warm_repeat") or {}).get("p95_ms"),
            "latency_target_ms": latency.get("latency_target_ms", LATENCY_TARGET_MS),
            "gloss_latency_ok_p95": gloss.get("latency_ok_p95"),
        }

    usability = reports.get("usability")
    if usability:
        overall = usability.get("overall") or {}
        summary["usability"] = {
            "sus_mean": overall.get("mean"),
            "sus_adjective": overall.get("adjective"),
            "target_sus": overall.get("target_score", 80.0),
            "target_met": overall.get("target_met"),
            "participants": usability.get("participants"),
            "participants_met": usability.get("participants_met"),
            "groups_missing": usability.get("groups_missing"),
            "task_success_rate": usability.get("task_success_rate"),
            "by_group": {
                group: stats.get("mean")
                for group, stats in (usability.get("by_group") or {}).items()
            },
        }

    formative = reports.get("formative_usability")
    scenarios = reports.get("scenario_evaluation")
    if formative or scenarios:
        substitute: Dict[str, Any] = {
            "method": "formative_and_scenarios",
            "replaces": "summative SUS study",
            "sus_target_met": False,
        }
        if formative:
            h = formative.get("heuristics") or {}
            w = formative.get("walkthrough") or {}
            substitute["heuristic_mean_severity"] = h.get("mean_severity")
            substitute["heuristic_actionable_issues"] = h.get("actionable_issues")
            substitute["walkthrough_task_success_rate"] = w.get("task_success_rate")
            substitute["formative_reason"] = formative.get("reason")
        if scenarios:
            substitute["scenario_success_rate"] = scenarios.get("success_rate")
            substitute["scenario_passes"] = scenarios.get("successes")
            substitute["scenario_total"] = scenarios.get("n_scenarios")
        summary["usability_substitute"] = substitute

    return summary


def _stale_recognition_notes(summary: Dict[str, Any]) -> list[str]:
    """Warn when a trained recogniser is missing from the stored reports.

    The demo always loads the best artefact on disk, so a report generated
    before a retrain would describe a model that is no longer being used.
    """
    from modules.recognition.registry import available_backends

    on_disk = set(available_backends())
    if not on_disk:
        return []

    notes: list[str] = []
    for key, script in (
        ("recognition", "scripts/evaluate_recognizers.py"),
        ("user_recognition", "scripts/evaluate_on_user_data.py"),
    ):
        scored = set((summary.get(key) or {}).get("backends") or {})
        if not scored:
            continue
        unscored = sorted(on_disk - scored)
        if unscored:
            notes.append(
                f"{', '.join(unscored)} trained but absent from {key} report; "
                f"re-run {script}."
            )
    return notes


def build_unified_report(
    log_dir: str = DEFAULT_LOG_DIR,
    latency_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate module metrics and optional latency benchmark into one report."""
    bundle = load_module_reports(log_dir)
    reports = dict(bundle["reports"])
    if latency_report is not None:
        reports["latency"] = latency_report

    summary = _extract_module_summary(reports)
    targets = {
        "latency_ms": LATENCY_TARGET_MS,
        "emotion_accuracy": 0.80,
        "nlp_bleu_min": 50.0,
        "sus_min": 80.0,
        "min_participants": 10,
    }

    status_notes: list[str] = _stale_recognition_notes(summary)
    if "latency" in summary:
        ok = summary["latency"].get("gloss_latency_ok_p95")
        if ok is True:
            status_notes.append("Gloss-pipeline p95 latency is under the 1.5 s target.")
        elif ok is False:
            status_notes.append(
                "Gloss-pipeline p95 latency exceeds 1.5 s (TTS network calls dominate)."
            )
    if summary.get("emotion", {}).get("target_met") is False:
        status_notes.append("Emotion accuracy is below the 80% proposal target.")

    usability = summary.get("usability")
    substitute = summary.get("usability_substitute")
    if usability:
        mean = usability.get("sus_mean")
        if usability.get("target_met") is True:
            status_notes.append(
                f"SUS mean {mean} meets the proposal target (≥ 80)."
            )
        elif usability.get("target_met") is False:
            status_notes.append(
                f"SUS mean {mean} is below the 80 proposal target."
            )
        if usability.get("participants_met") is False:
            status_notes.append(
                f"SUS study in progress: {usability.get('participants')} of 10 "
                "participants recorded so far."
            )
        if usability.get("groups_missing"):
            status_notes.append(
                "SUS groups not yet recruited: "
                + ", ".join(usability["groups_missing"])
                + "."
            )
        status_notes.append(
            "Participant session files are local-only; cite aggregate SUS statistics "
            "in the dissertation (see study/DATA_HANDLING.md)."
        )

    missing = list(bundle["missing"])
    if "usability" in missing:
        missing.remove("usability")
        if substitute:
            parts = []
            if substitute.get("scenario_success_rate") is not None:
                parts.append(
                    f"scenarios {substitute.get('scenario_passes')}/"
                    f"{substitute.get('scenario_total')} passed"
                )
            if substitute.get("walkthrough_task_success_rate") is not None:
                parts.append(
                    f"walkthrough {substitute['walkthrough_task_success_rate']:.0%} task success"
                )
            detail = f" ({', '.join(parts)})" if parts else ""
            status_notes.append(
                "Summative SUS study not conducted; formative heuristic review and "
                f"automated scenarios used instead{detail}. Not comparable to SUS ≥ 80."
            )
        else:
            status_notes.append(
                "No usability evaluation recorded. Run scripts/run_formative_review.py "
                "and scripts/evaluate_scenarios.py, or recruit participants for SUS."
            )

    # Formative/scenario reports are optional substitutes, not required module metrics.
    for optional in ("formative_usability", "scenario_evaluation"):
        if optional in missing:
            missing.remove(optional)
    if missing:
        status_notes.append(
            "Missing module reports: "
            + ", ".join(missing)
            + ". Run the per-module evaluate_* scripts."
        )

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "log_dir": str(Path(log_dir)),
        "targets": targets,
        "module_summary": summary,
        "module_reports": reports,
        "missing_reports": bundle["missing"],
        "notes": status_notes,
    }


def write_unified_report(
    report: Dict[str, Any],
    log_dir: str = DEFAULT_LOG_DIR,
    filename: str = "unified_report.json",
) -> Path:
    path = Path(log_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(path)
    return path


def run_unified_evaluation(
    log_dir: str = DEFAULT_LOG_DIR,
    run_latency: bool = True,
    use_nlp_model: bool = False,
) -> Dict[str, Any]:
    """Build unified report, optionally running the latency benchmark first."""
    latency_report = None
    if run_latency:
        latency_report = run_latency_benchmark(use_nlp_model=use_nlp_model)
        latency_path = Path(log_dir) / MODULE_REPORT_PATHS["latency"]
        latency_path.parent.mkdir(parents=True, exist_ok=True)
        latency_path.write_text(json.dumps(latency_report, indent=2), encoding="utf-8")

    report = build_unified_report(log_dir=log_dir, latency_report=latency_report)
    write_unified_report(report, log_dir=log_dir)
    return report
