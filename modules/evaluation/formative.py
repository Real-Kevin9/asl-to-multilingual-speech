"""Formative usability evaluation when a summative SUS study is not feasible.

When ≥10 participants cannot be recruited, the dissertation still needs
usability evidence. This module holds two instruments that do not require
participants:

1. **Nielsen's 10 heuristics** — severity-rated expert review (0 = no issue,
   4 = usability catastrophe).
2. **Structured developer walkthrough** — the same five pipeline tasks used in
   the planned SUS study, recorded by the researcher with explicit success/fail.

These are *formative* methods. They cannot produce a SUS score and must not be
presented as meeting the proposal's SUS ≥ 80 target.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Sequence

# Severity scale follows Nielsen's convention for heuristic evaluation.
SEVERITY_MIN = 0
SEVERITY_MAX = 4
SEVERITY_LABELS = {
    0: "no issue",
    1: "cosmetic",
    2: "minor",
    3: "major",
    4: "catastrophe",
}

HEURISTICS: Sequence[str] = (
    "Visibility of system status — the system keeps users informed about what is going on.",
    "Match between the system and the real world — language and concepts feel natural.",
    "User control and freedom — users can undo mistakes and are not trapped in states.",
    "Consistency and standards — the same words and actions mean the same thing throughout.",
    "Error prevention — the design prevents problems before they occur.",
    "Recognition rather than recall — options and instructions are visible when needed.",
    "Flexibility and efficiency — shortcuts exist for experienced users without blocking novices.",
    "Aesthetic and minimalist design — every extra unit of information competes with what matters.",
    "Help users recover from errors — error messages are plain language with a constructive next step.",
    "Help and documentation — help is easy to find and focused on the user's task.",
)

# Mirrors the SUS study tasks so formative and summative results are comparable.
WALKTHROUGH_TASKS: Sequence[Dict[str, str]] = (
    {
        "id": "fingerspell",
        "objective": "Recognition",
        "prompt": "Fingerspell a three-letter word (or run the CAT scenario) and confirm "
                  "the letters appear correctly.",
    },
    {
        "id": "sentence",
        "objective": "NLP",
        "prompt": "Provide a short gloss and check the English sentence reads naturally.",
    },
    {
        "id": "speech_en",
        "objective": "TTS (English)",
        "prompt": "Have the system speak in English and confirm audio is produced.",
    },
    {
        "id": "speech_ne",
        "objective": "TTS (Nepali)",
        "prompt": "Switch to Nepali and confirm speech is produced.",
    },
    {
        "id": "emotion",
        "objective": "Emotion",
        "prompt": "Provide a face image and note whether emotion detection changes prosody.",
    },
)


def validate_severity(value: int) -> int:
    severity = int(value)
    if not SEVERITY_MIN <= severity <= SEVERITY_MAX:
        raise ValueError(
            f"Severity {severity} out of range; use {SEVERITY_MIN}-{SEVERITY_MAX}."
        )
    return severity


def summarise_heuristics(scores: Sequence[int], notes: Sequence[str] | None = None) -> Dict[str, Any]:
    """Aggregate heuristic severities. Lower is better."""
    values = [validate_severity(s) for s in scores]
    if len(values) != len(HEURISTICS):
        raise ValueError(f"Expected {len(HEURISTICS)} scores, got {len(values)}.")

    notes = list(notes or [])
    issues = [
        {
            "heuristic": i + 1,
            "statement": HEURISTICS[i],
            "severity": values[i],
            "label": SEVERITY_LABELS[values[i]],
            "note": notes[i] if i < len(notes) and notes[i] else None,
        }
        for i in range(len(values))
        if values[i] > 0
    ]
    actionable = [i for i in issues if i["severity"] >= 2]

    return {
        "n_heuristics": len(values),
        "mean_severity": round(statistics.mean(values), 2),
        "max_severity": max(values),
        "issues_found": len(issues),
        "actionable_issues": len(actionable),
        "issues": issues,
    }


def summarise_walkthrough(tasks: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate developer walkthrough task outcomes."""
    tasks = list(tasks)
    attempts = len(tasks)
    successes = sum(1 for t in tasks if t.get("success"))
    return {
        "n_tasks": attempts,
        "successes": successes,
        "task_success_rate": round(successes / attempts, 3) if attempts else None,
        "tasks": tasks,
    }


def build_formative_report(
    heuristic_scores: Sequence[int],
    heuristic_notes: Sequence[str] | None,
    walkthrough_tasks: Iterable[Dict[str, Any]],
    *,
    reason: str,
    evaluator: str = "developer",
) -> Dict[str, Any]:
    """Combine heuristic review and walkthrough into one JSON report."""
    heuristics = summarise_heuristics(heuristic_scores, heuristic_notes)
    walkthrough = summarise_walkthrough(walkthrough_tasks)
    return {
        "method": "formative_substitute",
        "substitutes_for": "summative SUS study (≥10 participants)",
        "reason": reason,
        "evaluator": evaluator,
        "heuristics": heuristics,
        "walkthrough": walkthrough,
        "limitations": [
            "Results reflect the developer's perspective, not deaf/HoH signers.",
            "Heuristic severity is subjective and not comparable to SUS scores.",
            "Cannot support claims about real-world usability for the target population.",
        ],
    }
