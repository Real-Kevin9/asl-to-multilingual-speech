"""Unit tests for WLASL data helpers (no network / no video downloads)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from modules.recognition.wlasl_data import (
    assemble_word_gloss,
    filter_samples,
    gloss_counts,
    sample_frame_indices,
    select_top_glosses,
    stratified_split,
)
from modules.recognition.wlasl_live import ClipRecorder


def _fake_samples(glosses_with_counts):
    samples = []
    i = 0
    for gloss, n in glosses_with_counts:
        for _ in range(n):
            samples.append({
                "filepath": f"data/data_0/{i:05d}.mp4",
                "gloss": {"label": gloss},
                "metadata": {"size_bytes": 1000},
            })
            i += 1
    return samples


def test_select_top_glosses_orders_by_frequency():
    samples = _fake_samples([("go", 5), ("help", 3), ("drink", 4), ("book", 2)])
    top = select_top_glosses(samples, num_classes=3)
    assert top == ["go", "drink", "help"]


def test_filter_samples():
    samples = _fake_samples([("go", 2), ("help", 1)])
    selected = filter_samples(samples, ["go"])
    assert len(selected) == 2
    assert all(r["gloss"] == "go" for r in selected)
    assert selected[0]["bbox"] is None


def test_assemble_word_gloss():
    assert assemble_word_gloss(["help", "go"]) == "HELP GO"


def test_stratified_split_keeps_both_sides():
    records = [{"gloss": "go", "video_path": f"v{i}"} for i in range(10)]
    records += [{"gloss": "help", "video_path": f"h{i}"} for i in range(5)]
    train, val = stratified_split(records, val_fraction=0.2, seed=0)
    assert {r["gloss"] for r in train} == {"go", "help"}
    assert {r["gloss"] for r in val} == {"go", "help"}
    assert len(train) + len(val) == len(records)


def test_sample_frame_indices_length():
    assert len(sample_frame_indices(100, 16)) == 16
    assert len(sample_frame_indices(3, 16)) == 16
    assert sample_frame_indices(1, 4) == [0, 0, 0, 0]


def test_gloss_counts():
    samples = _fake_samples([("a", 2), ("b", 1)])
    counts = gloss_counts(samples)
    assert counts["a"] == 2
    assert counts["b"] == 1


def test_clip_recorder():
    rec = ClipRecorder(target_fps=10.0)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    rec.start(now=100.0)
    rec.add_frame(frame, now=100.0)
    rec.add_frame(frame, now=100.12)
    assert rec.frame_count == 2
