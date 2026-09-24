"""System Usability Scale scoring for the participant study.

The proposal commits to a SUS score of at least 80 across at least ten
participants drawn from three groups (deaf/hard-of-hearing signers, interpreters,
hearing non-signers). This module holds the standard instrument and the scoring
rules so sessions and analysis cannot disagree about either.

Scoring follows Brooke (1996): odd-numbered statements are positively worded and
contribute ``response - 1``, even-numbered ones are negatively worded and
contribute ``5 - response``. The total is multiplied by 2.5, giving 0-100.

A SUS score is **not** a percentage. 68 is the historical average, so 68 is a C
grade rather than a poor one. Adjective bands follow Bangor et al. (2009).
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Sequence

# The ten standard statements, in order. Wording is fixed by the instrument;
# only the product name is substituted.
SUS_STATEMENTS: Sequence[str] = (
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
)

RESPONSE_MIN = 1
RESPONSE_MAX = 5

# Proposal target, and the published average for context.
TARGET_SCORE = 80.0
AVERAGE_SCORE = 68.0

PARTICIPANT_GROUPS = ("deaf_hoh", "interpreter", "hearing")


def score_responses(responses: Sequence[int]) -> float:
    """Convert ten 1-5 Likert responses into a 0-100 SUS score."""
    if len(responses) != len(SUS_STATEMENTS):
        raise ValueError(
            f"SUS needs exactly {len(SUS_STATEMENTS)} responses, got {len(responses)}."
        )

    total = 0
    for i, raw in enumerate(responses):
        value = int(raw)
        if not RESPONSE_MIN <= value <= RESPONSE_MAX:
            raise ValueError(
                f"Response {i + 1} is {value}; SUS responses must be "
                f"{RESPONSE_MIN}-{RESPONSE_MAX}."
            )
        # Odd-numbered statements (1-indexed) are positively worded.
        total += (value - 1) if i % 2 == 0 else (RESPONSE_MAX - value)

    return total * 2.5


def adjective(score: float) -> str:
    """Bangor et al. adjective band for a SUS score."""
    if score >= 90:
        return "best imaginable"
    if score >= 80.3:
        return "excellent"
    if score >= 68:
        return "good"
    if score >= 51:
        return "ok"
    if score >= 39:
        return "poor"
    return "worst imaginable"


def summarise(scores: Sequence[float]) -> Dict[str, Any]:
    """Aggregate SUS scores, with the proposal target checked explicitly."""
    values = [float(s) for s in scores]
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "stdev": None,
            "adjective": None,
            "target_score": TARGET_SCORE,
            "target_met": None,
        }

    mean = statistics.mean(values)
    return {
        "n": len(values),
        "mean": round(mean, 1),
        "median": round(statistics.median(values), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "stdev": round(statistics.stdev(values), 1) if len(values) > 1 else 0.0,
        "adjective": adjective(mean),
        "above_average": mean > AVERAGE_SCORE,
        "target_score": TARGET_SCORE,
        "target_met": mean >= TARGET_SCORE,
    }


def summarise_by_group(sessions: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Overall and per-group SUS, plus task success and fluency ratings.

    Per-group reporting is a requirement, not a nicety: a system that scores well
    overall can still be unusable for the group it is built for.
    """
    sessions = list(sessions)
    by_group: Dict[str, List[float]] = {}
    all_scores: List[float] = []
    fluency: List[float] = []
    task_attempts = 0
    task_successes = 0

    for session in sessions:
        score = session.get("sus_score")
        if score is None:
            continue
        all_scores.append(float(score))
        by_group.setdefault(session.get("group") or "unknown", []).append(float(score))

        rating = session.get("fluency_rating")
        if rating is not None:
            fluency.append(float(rating))

        for task in session.get("tasks") or []:
            task_attempts += 1
            if task.get("success"):
                task_successes += 1

    report: Dict[str, Any] = {
        "overall": summarise(all_scores),
        "by_group": {name: summarise(scores) for name, scores in sorted(by_group.items())},
        "participants": len(all_scores),
        "min_participants": 10,
        "participants_met": len(all_scores) >= 10,
        "groups_covered": sorted(by_group),
        "groups_missing": [g for g in PARTICIPANT_GROUPS if g not in by_group],
    }

    if fluency:
        report["fluency_rating"] = {
            "n": len(fluency),
            "mean": round(statistics.mean(fluency), 2),
            "scale": "1-5, higher is more natural English",
        }
    if task_attempts:
        report["task_success_rate"] = round(task_successes / task_attempts, 3)
        report["task_attempts"] = task_attempts

    return report
