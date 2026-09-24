"""Tests for formative usability and scenario evaluation helpers."""

import json

import pytest

from modules.evaluation.formative import (
    HEURISTICS,
    WALKTHROUGH_TASKS,
    build_formative_report,
    summarise_heuristics,
    summarise_walkthrough,
    validate_severity,
)
from modules.evaluation.unified import build_unified_report


def test_severity_range():
    assert validate_severity(0) == 0
    assert validate_severity(4) == 4
    with pytest.raises(ValueError):
        validate_severity(5)


def test_heuristic_summary_counts_actionable_issues():
    scores = [0, 1, 2, 3, 4, 0, 1, 2, 0, 1]
    summary = summarise_heuristics(scores)
    assert summary["issues_found"] == 7
    assert summary["actionable_issues"] == 4  # severities 2, 3, 4, 2
    assert summary["max_severity"] == 4


def test_walkthrough_success_rate():
    tasks = [
        {"id": "a", "success": True},
        {"id": "b", "success": False},
        {"id": "c", "success": True},
    ]
    summary = summarise_walkthrough(tasks)
    assert summary["successes"] == 2
    assert summary["task_success_rate"] == round(2 / 3, 3)


def test_build_formative_report_structure():
    report = build_formative_report(
        [0] * len(HEURISTICS),
        [],
        [{"id": t["id"], "success": True} for t in WALKTHROUGH_TASKS],
        reason="test",
    )
    assert report["method"] == "formative_substitute"
    assert report["heuristics"]["mean_severity"] == 0.0
    assert report["walkthrough"]["successes"] == len(WALKTHROUGH_TASKS)


def test_unified_report_includes_usability_substitute(tmp_path):
    log_dir = tmp_path / "eval"
    log_dir.mkdir()
    (log_dir / "formative_usability.json").write_text(
        json.dumps(build_formative_report(
            [0, 1, 2, 0, 1, 2, 1, 0, 2, 1],
            [],
            [{"id": "fingerspell", "success": True}, {"id": "sentence", "success": False}],
            reason="no participants",
        )),
        encoding="utf-8",
    )
    (log_dir / "scenario_evaluation.json").write_text(
        json.dumps({"n_scenarios": 5, "successes": 4, "success_rate": 0.8}),
        encoding="utf-8",
    )

    report = build_unified_report(log_dir=str(log_dir))
    sub = report["module_summary"]["usability_substitute"]

    assert sub["scenario_success_rate"] == 0.8
    assert sub["sus_target_met"] is False
    assert any("formative" in n.lower() for n in report["notes"])
