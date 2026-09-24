import numpy as np
import pytest

from modules.recognition.crop_dataset import (
    _contiguous_split,
    _select_by_group,
    image_group,
)


def test_image_group_classification(tmp_path):
    from pathlib import Path

    assert image_group(Path("A/1.jpg")) == "scene"
    assert image_group(Path("A/1234.jpg")) == "scene"
    assert image_group(Path("A/A (12).jpg")) == "closeup"


def test_user_captures_count_as_scene_framing():
    """Own-webcam captures are scene-framed, so they must not be filed as close-ups."""
    from pathlib import Path

    assert image_group(Path("A/user_A_000_12345.jpg")) == "scene"
    assert image_group(Path("space/user_space_004_999.jpg")) == "scene"


def test_select_by_group_respects_quotas(tmp_path):
    class_dir = tmp_path / "A"
    class_dir.mkdir()
    for i in range(10):
        (class_dir / f"{i}.jpg").touch()
    for i in range(10):
        (class_dir / f"A ({i}).jpg").touch()

    picked = _select_by_group(class_dir, scene_limit=3, closeup_limit=5)
    names = [p.name for p in picked]

    assert sum(1 for n in names if n[0].isdigit()) == 3
    assert sum(1 for n in names if not n[0].isdigit()) == 5


def _items(n_per_bucket=10):
    items = []
    for label in ("A", "B"):
        for group in ("scene", "closeup"):
            for i in range(n_per_bucket):
                items.append({"file": f"{label}/{group}{i}.jpg", "label": label, "group": group})
    return items


def test_contiguous_split_is_a_partition():
    items = _items()
    train_idx, val_idx = _contiguous_split(items, 0.2)

    assert set(train_idx) & set(val_idx) == set()
    assert len(train_idx) + len(val_idx) == len(items)


def test_contiguous_split_covers_every_class_and_group():
    items = _items()
    _, val_idx = _contiguous_split(items, 0.2)

    labels = {items[i]["label"] for i in val_idx}
    groups = {items[i]["group"] for i in val_idx}
    assert labels == {"A", "B"}
    assert groups == {"scene", "closeup"}


def test_contiguous_split_takes_a_trailing_block_not_a_random_sample():
    """Neighbouring scene frames are near-duplicates; the split must not interleave."""
    items = _items(n_per_bucket=10)
    train_idx, val_idx = _contiguous_split(items, 0.2)

    # Within each bucket the validation indices must be the final, contiguous ones.
    buckets = {}
    for idx, item in enumerate(items):
        buckets.setdefault((item["label"], item["group"]), []).append(idx)

    for bucket_indices in buckets.values():
        chosen = [i for i in bucket_indices if i in set(val_idx)]
        assert chosen == bucket_indices[len(bucket_indices) - len(chosen):]


def test_hybrid_model_accepts_both_streams():
    from modules.recognition.cnn_lstm import build_hybrid_model

    model = build_hybrid_model(num_classes=6, augment=False)

    assert len(model.inputs) == 2
    assert tuple(model.inputs[0].shape) == (None, 160, 160, 3)
    assert tuple(model.inputs[1].shape) == (None, 63)
    assert model.output_shape == (None, 6)

    # The proposal's BiLSTM path must survive the fusion.
    assert tuple(model.get_layer("spatial_sequence").output.shape) == (None, 25, 1280)
    assert tuple(model.get_layer("bilstm_2").output.shape) == (None, 128)


def test_hybrid_model_predicts_probabilities():
    from modules.recognition.cnn_lstm import build_hybrid_model

    model = build_hybrid_model(num_classes=5, augment=False)
    crops = np.random.randint(0, 255, (2, 160, 160, 3)).astype("float32")
    geom = np.random.randn(2, 63).astype("float32")

    probs = model.predict([crops, geom], verbose=0)
    assert probs.shape == (2, 5)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
