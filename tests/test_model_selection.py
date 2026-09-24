import json

import pytest

from pipeline.model_selection import (
    format_model_report,
    resolve_models,
    resolve_nlp_use_model,
)


def _write_recognition_models(base, names):
    files = {
        "hybrid_user": "hybrid_user_model.keras",
        "hybrid": "hybrid_model.keras",
        "cnn_lstm": "cnn_lstm_model.keras",
        "landmark_mlp": "recognition_model.joblib",
    }
    base.mkdir(parents=True, exist_ok=True)
    for name in names:
        (base / files[name]).write_bytes(b"stub")


def test_auto_recognition_prefers_user_finetuned(tmp_path):
    rec_dir = tmp_path / "recognition"
    _write_recognition_models(rec_dir, ["hybrid", "cnn_lstm", "hybrid_user"])
    (rec_dir / "hybrid_user_metadata.json").write_text(
        json.dumps({
            "metrics": {"val_accuracy": 0.94},
            "user_finetune": {"after": 0.946},
        }),
        encoding="utf-8",
    )

    selection = resolve_models(
        recognition_dir=str(rec_dir),
        nlp_model_dir=str(tmp_path / "missing_nlp"),
        emotion_model_path=str(tmp_path / "missing.keras"),
        emotion_metadata_path=str(tmp_path / "missing.json"),
        eval_dir=str(tmp_path / "eval"),
    )

    assert selection.recognition.backend == "hybrid_user"
    assert selection.recognition.available is True
    assert selection.recognition.metrics["user_holdout_accuracy"] == 0.946


def test_missing_artefacts_degrade_gracefully(tmp_path):
    selection = resolve_models(
        recognition_dir=str(tmp_path / "recognition"),
        nlp_model_dir=str(tmp_path / "nlp"),
        emotion_model_path=str(tmp_path / "emotion.keras"),
        emotion_metadata_path=str(tmp_path / "emotion.json"),
        eval_dir=str(tmp_path / "eval"),
    )

    assert selection.recognition.available is False
    assert selection.nlp.backend == "rule_based"
    assert selection.emotion.backend == "fallback_neutral"
    # TTS never depends on a trained artefact.
    assert selection.tts.available is True
    assert "Models in use" in format_model_report(selection)


def test_nlp_auto_uses_local_model_when_present(tmp_path):
    nlp_dir = tmp_path / "t5_gloss_en"
    nlp_dir.mkdir()
    (nlp_dir / "config.json").write_text("{}", encoding="utf-8")

    assert resolve_nlp_use_model("auto", str(nlp_dir)) is True
    assert resolve_nlp_use_model("off", str(nlp_dir)) is False
    assert resolve_nlp_use_model("auto", str(tmp_path / "nope")) is False
    assert resolve_nlp_use_model("on", str(tmp_path / "nope")) is True

    selection = resolve_models(
        recognition_dir=str(tmp_path / "recognition"),
        nlp_model_dir=str(nlp_dir),
        emotion_model_path=str(tmp_path / "emotion.keras"),
        emotion_metadata_path=str(tmp_path / "emotion.json"),
        eval_dir=str(tmp_path / "eval"),
    )
    assert selection.nlp.backend == "t5_gloss"


def test_unknown_nlp_mode_rejected():
    with pytest.raises(ValueError):
        resolve_nlp_use_model("maybe")
