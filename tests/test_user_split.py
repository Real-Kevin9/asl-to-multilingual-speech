import json

import pytest

from modules.recognition.user_split import (
    held_out_user_files,
    load_manifest_items,
    split_per_class,
)


def _make_cache(tmp_path, per_class):
    cache = tmp_path / "user_crops"
    cache.mkdir()
    items = []
    for label, count in per_class.items():
        for i in range(count):
            items.append({"file": f"{label}/user_{label}_{i:03d}_0.jpg", "label": label})
    (cache / "manifest.json").write_text(
        json.dumps({"classes": sorted(per_class), "items": items}), encoding="utf-8"
    )
    return cache, items


def _make_raw(tmp_path, items):
    raw = tmp_path / "user_samples"
    for it in items:
        label = it["label"]
        (raw / label).mkdir(parents=True, exist_ok=True)
        (raw / label / f"{it['file'].split('/')[-1]}").write_bytes(b"stub")
    return raw


def test_split_holds_out_trailing_block_per_class():
    items = [{"label": "A", "file": f"A/{i}.jpg"} for i in range(20)]
    items += [{"label": "B", "file": f"B/{i}.jpg"} for i in range(16)]

    train, val = split_per_class(items, 0.4)

    assert len(train) == 12 + 10
    assert len(val) == 8 + 6
    # Held-out frames are the trailing ones, never interleaved.
    assert sorted(val)[:8] == list(range(12, 20))
    assert set(train).isdisjoint(val)


def test_split_reproduces_the_recorded_finetune_counts():
    # The real cache: A-Y without J (no samples), del=18, space=16 -> 309 / 205.
    per_class = {chr(c): 20 for c in range(ord("A"), ord("Y") + 1)}
    del per_class["J"]
    per_class["del"] = 18
    per_class["space"] = 16
    items = [
        {"label": label, "file": f"{label}/{i}.jpg"}
        for label, count in per_class.items()
        for i in range(count)
    ]

    train, val = split_per_class(items, 0.4)

    assert (len(train), len(val)) == (309, 205)


def test_held_out_files_map_back_to_raw_images(tmp_path):
    cache, items = _make_cache(tmp_path, {"A": 10, "B": 10})
    raw = _make_raw(tmp_path, items)

    paths = held_out_user_files(
        user_cache=str(cache), user_dir=str(raw), val_fraction=0.4, expected_total=20
    )

    assert len(paths) == 8
    assert all(p.exists() for p in paths)
    assert {p.parent.name for p in paths} == {"A", "B"}


def test_rebuilt_cache_is_rejected(tmp_path):
    """A cache that no longer matches the metadata reconstructs a different split."""
    cache, items = _make_cache(tmp_path, {"A": 10})
    raw = _make_raw(tmp_path, items)

    with pytest.raises(ValueError, match="rebuilt"):
        held_out_user_files(
            user_cache=str(cache), user_dir=str(raw), expected_total=999
        )


def test_missing_cache_reports_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_manifest_items(str(tmp_path / "nope"))
