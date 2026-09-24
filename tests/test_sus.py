import json
from pathlib import Path

import pytest

from modules.evaluation.unified import build_unified_report
from modules.evaluation.sus import (
    SUS_STATEMENTS,
    adjective,
    score_responses,
    summarise,
    summarise_by_group,
)


def test_all_best_answers_score_100():
    # Agree fully with positive statements, disagree fully with negative ones.
    best = [5 if i % 2 == 0 else 1 for i in range(10)]
    assert score_responses(best) == 100.0


def test_all_worst_answers_score_zero():
    worst = [1 if i % 2 == 0 else 5 for i in range(10)]
    assert score_responses(worst) == 0.0


def test_neutral_answers_score_fifty():
    assert score_responses([3] * 10) == 50.0


def test_known_response_pattern():
    # Odd items contribute (r-1), even items (5-r): 3+1+3+2+3 + 2+3+1+3+2 = 23 -> 57.5
    responses = [4, 4, 4, 3, 4, 3, 4, 4, 4, 3]
    assert score_responses(responses) == 57.5


def test_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="exactly 10"):
        score_responses([3] * 9)


def test_out_of_range_response_is_rejected():
    with pytest.raises(ValueError, match="must be 1-5"):
        score_responses([3, 3, 3, 3, 9, 3, 3, 3, 3, 3])


def test_instrument_has_ten_statements():
    assert len(SUS_STATEMENTS) == 10


def test_adjective_bands():
    assert adjective(95) == "best imaginable"
    assert adjective(82) == "excellent"
    assert adjective(70) == "good"
    assert adjective(60) == "ok"
    assert adjective(45) == "poor"
    assert adjective(20) == "worst imaginable"


def test_summarise_flags_the_target():
    assert summarise([85.0, 90.0])["target_met"] is True
    assert summarise([70.0, 75.0])["target_met"] is False
    assert summarise([])["n"] == 0


def test_group_summary_reports_gaps_and_tasks():
    sessions = [
        {
            "group": "deaf_hoh",
            "sus_score": 82.5,
            "fluency_rating": 4,
            "tasks": [{"success": True}, {"success": False}],
        },
        {"group": "hearing", "sus_score": 90.0, "tasks": [{"success": True}]},
    ]

    report = summarise_by_group(sessions)

    assert report["participants"] == 2
    assert report["participants_met"] is False
    assert report["groups_missing"] == ["interpreter"]
    assert report["by_group"]["deaf_hoh"]["mean"] == 82.5
    assert report["task_success_rate"] == round(2 / 3, 3)
    assert report["fluency_rating"]["mean"] == 4.0


def test_paper_questionnaire_matches_the_instrument():
    # The printed sheet and the session script must ask identical questions,
    # otherwise the two sets of responses cannot be pooled.
    questionnaire = Path(__file__).resolve().parents[1] / "study" / "questionnaire.md"
    text = questionnaire.read_text(encoding="utf-8")
    for statement in SUS_STATEMENTS:
        assert statement in text, f"missing from questionnaire.md: {statement}"


def test_nlp_summary_follows_the_backend_key(tmp_path):
    # evaluate_nlp.py keys the neural results by backend name, not a fixed label.
    log_dir = tmp_path / "eval"
    log_dir.mkdir()
    (log_dir / "nlp_bleu.json").write_text(
        json.dumps({
            "n": 300,
            "backend": "t5_gloss",
            "rule_based": {"bleu": 19.19},
            "t5_gloss": {"bleu": 57.62},
            "delta_bleu": 38.43,
        }),
        encoding="utf-8",
    )

    summary = build_unified_report(log_dir=str(log_dir))["module_summary"]["nlp"]

    assert summary["t5_bleu"] == 57.62
    assert summary["backend"] == "t5_gloss"
    assert summary["fluency_target_met"] is True


def test_unified_report_picks_up_usability(tmp_path):
    log_dir = tmp_path / "eval"
    log_dir.mkdir()
    (log_dir / "usability_sus.json").write_text(
        json.dumps(summarise_by_group([
            {"group": "deaf_hoh", "sus_score": 72.5},
            {"group": "hearing", "sus_score": 77.5},
        ])),
        encoding="utf-8",
    )

    report = build_unified_report(log_dir=str(log_dir))
    usability = report["module_summary"]["usability"]

    assert usability["sus_mean"] == 75.0
    assert usability["target_met"] is False
    assert usability["by_group"]["deaf_hoh"] == 72.5
    assert report["targets"]["sus_min"] == 80.0

    notes = " ".join(report["notes"])
    assert "below the 80 proposal target" in notes
    assert "2 of 10 participants recorded so far" in notes
    assert "interpreter" in notes
