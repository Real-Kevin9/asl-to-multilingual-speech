"""Tests for usability data redaction."""

from modules.evaluation.privacy import build_public_summary, redact_comments


def test_redact_comments_drops_participant_codes():
    out = redact_comments([
        {"participant": "P01", "group": "hearing", "liked": "fast", "disliked": None},
        {"participant": "P02", "group": "deaf_hoh", "suggestion": "bigger text"},
    ])
    assert "participant" not in str(out)
    assert out[0] == {"group": "hearing", "liked": "fast"}
    assert out[1] == {"group": "deaf_hoh", "suggestion": "bigger text"}


def test_public_summary_has_no_raw_responses():
    full = {
        "participants": 2,
        "overall": {"mean": 82.5, "target_met": True, "adjective": "excellent"},
        "by_group": {"hearing": {"n": 2, "mean": 82.5, "adjective": "excellent"}},
        "qualitative": [{"participant": "P01", "group": "hearing", "liked": "ok"}],
        "tasks": {"fingerspell": {"success_rate": 0.5}},
    }
    public = build_public_summary(full)

    assert public["overall"]["sus_mean"] == 82.5
    assert public["data_governance"]["identifiers_removed"] is True
    assert "P01" not in str(public)
    assert public["anonymous_comments"][0]["group"] == "hearing"
