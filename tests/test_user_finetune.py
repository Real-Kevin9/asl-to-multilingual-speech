"""Tests for the user-data fine-tuning helpers."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "finetune_on_user_data",
    Path(__file__).resolve().parents[1] / "scripts" / "finetune_on_user_data.py",
)
finetune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finetune)


def make_items(counts):
    items = []
    for label, n in counts.items():
        for i in range(n):
            items.append({"label": label, "file": f"{label}/{i}.jpg", "group": "scene"})
    return items


def test_split_is_a_partition():
    items = make_items({"A": 20, "B": 20, "C": 20})
    train, val = finetune.split_per_class(items, 0.4)
    assert sorted(train + val) == list(range(len(items)))
    assert not set(train) & set(val)


def test_every_class_appears_on_both_sides():
    items = make_items({"A": 20, "B": 20, "C": 20})
    train, val = finetune.split_per_class(items, 0.4)
    for side in (train, val):
        assert {items[i]["label"] for i in side} == {"A", "B", "C"}


def test_validation_takes_the_trailing_block():
    """Consecutive webcam frames are near-duplicates, so a random split would leak."""
    items = make_items({"A": 10})
    train, val = finetune.split_per_class(items, 0.4)
    assert train == [0, 1, 2, 3, 4, 5]
    assert val == [6, 7, 8, 9]


def test_single_sample_class_stays_in_training():
    items = make_items({"A": 1, "B": 10})
    train, val = finetune.split_per_class(items, 0.4)
    assert 0 in train
    assert 0 not in val
