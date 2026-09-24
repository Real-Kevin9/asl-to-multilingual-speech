"""Redact usability reports for dissertation citation.

Raw session files under ``logs/usability/`` may contain pseudonymous task notes
and optional free-text comments. They stay on the researcher's machine and are
never committed to git or shared as files.

This module produces aggregate-only summaries safe to **cite** in the dissertation
(mean SUS, per-group means, task success rates). It does not make raw participant
records safe to publish as a dataset or hand to third parties.
"""

from __future__ import annotations

from typing import Any, Dict, List


def redact_comments(qualitative: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop participant codes; keep only group and comment text for local write-up."""
    redacted = []
    for entry in qualitative or []:
        item = {"group": entry.get("group")}
        for key in ("liked", "disliked", "suggestion"):
            if entry.get(key):
                item[key] = entry[key]
        if len(item) > 1:
            redacted.append(item)
    return redacted


def build_public_summary(full_report: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate-only view for the dissertation — no individual identifiers."""
    overall = full_report.get("overall") or {}
    by_group = full_report.get("by_group") or {}

    return {
        "method": "System Usability Scale (SUS)",
        "participants": full_report.get("participants"),
        "min_participants": full_report.get("min_participants", 10),
        "participants_met": full_report.get("participants_met"),
        "groups_covered": full_report.get("groups_covered"),
        "groups_missing": full_report.get("groups_missing"),
        "overall": {
            "sus_mean": overall.get("mean"),
            "sus_median": overall.get("median"),
            "sus_min": overall.get("min"),
            "sus_max": overall.get("max"),
            "sus_stdev": overall.get("stdev"),
            "adjective": overall.get("adjective"),
            "target_score": overall.get("target_score", 80.0),
            "target_met": overall.get("target_met"),
        },
        "by_group": {
            group: {
                "n": stats.get("n"),
                "sus_mean": stats.get("mean"),
                "adjective": stats.get("adjective"),
            }
            for group, stats in by_group.items()
        },
        "task_success": full_report.get("tasks"),
        "fluency_rating": full_report.get("fluency_rating"),
        "task_success_rate": full_report.get("task_success_rate"),
        "anonymous_comments": redact_comments(full_report.get("qualitative") or []),
        "data_governance": {
            "raw_session_files": "not included — stored locally under logs/usability/",
            "external_sharing": "prohibited — cite aggregate statistics only",
            "identifiers_removed": True,
        },
    }
