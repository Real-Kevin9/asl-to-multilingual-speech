import pytest

from modules.recognition.registry import BACKENDS, available_backends, create_recognizer


def test_backends_listed():
    assert BACKENDS == (
        "auto", "hybrid_user", "hybrid", "cnn_lstm", "landmark_mlp", "wlasl_video",
    )


def test_user_adapted_model_is_preferred_by_auto():
    """A model fine-tuned on this machine's camera beats the generic one."""
    found = available_backends()
    if "hybrid_user" in found:
        assert found[0] == "hybrid_user"


def test_landmark_mlp_is_always_offered():
    """The original MLP must stay available even once better models exist."""
    assert "landmark_mlp" in BACKENDS
    assert available_backends()[-1] == "landmark_mlp"


def test_available_backends_reports_existing_models():
    found = available_backends()
    assert isinstance(found, list)
    # The landmark MLP is committed in the repo, so it should always be found.
    assert "landmark_mlp" in found


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        create_recognizer(backend="does_not_exist")


def test_missing_model_dir_raises():
    with pytest.raises(FileNotFoundError):
        create_recognizer(backend="auto", model_dir="models/definitely_missing")


def test_landmark_backend_exposes_uniform_interface():
    recognizer = create_recognizer(backend="landmark_mlp")
    assert recognizer.backend_name == "landmark_mlp"
    assert hasattr(recognizer, "predict_frame")
