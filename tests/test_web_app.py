"""Tests for the FastAPI web dashboard helpers."""

from pathlib import Path

from app.backend.session import RealtimeSession, _decode_frame, safe_audio_path


def test_safe_audio_path_rejects_traversal(tmp_path, monkeypatch):
    root = tmp_path / "tts"
    root.mkdir()
    good = root / "hello_en.mp3"
    good.write_bytes(b"fake")

    monkeypatch.setattr(
        "app.backend.session.REPO_ROOT",
        tmp_path.parent,
    )
    # REPO_ROOT / logs / tts -> tmp_path.parent / logs / tts
    logs_tts = tmp_path.parent / "logs" / "tts"
    logs_tts.mkdir(parents=True)
    target = logs_tts / "hello_en.mp3"
    target.write_bytes(b"fake")

    assert safe_audio_path("../secret.mp3") is None
    assert safe_audio_path("hello_en.mp3") == target.resolve()


def test_assemble_word_gloss():
    from modules.recognition.wlasl_live import assemble_word_gloss

    assert assemble_word_gloss(["help", "go"]) == "HELP GO"
    assert assemble_word_gloss(["", " drink "]) == "DRINK"


def test_clip_recorder_subsamples():
    from modules.recognition.wlasl_live import ClipRecorder
    import numpy as np

    rec = ClipRecorder(target_fps=10.0, max_seconds=1.0)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    t0 = 100.0
    rec.start(now=t0)
    assert rec.add_frame(frame, now=t0) is True
    assert rec.frame_count == 1
    # Too soon for another keep at 10 fps
    assert rec.add_frame(frame, now=t0 + 0.05) is True
    assert rec.frame_count == 1
    assert rec.add_frame(frame, now=t0 + 0.12) is True
    assert rec.frame_count == 2
    frames = rec.stop()
    assert len(frames) == 2
    assert not rec.recording


def test_realtime_session_clear():
    session = RealtimeSession()
    session.labels = ["H", "E"]
    session.clear()
    assert session.labels == []


def test_decode_frame_invalid():
    assert _decode_frame("not-valid-base64") is None
    assert _decode_frame("") is None
