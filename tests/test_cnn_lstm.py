from pathlib import Path

import numpy as np
import pytest

from modules.recognition.crop_dataset import _select_class_files


def test_static_model_shapes():
    """MobileNetV2 feature map must become a sequence the BiLSTM can read."""
    from modules.recognition.cnn_lstm import build_static_model

    model = build_static_model(num_classes=5, input_shape=(160, 160, 3), augment=False)

    seq = model.get_layer("spatial_sequence")
    # 160x160 input -> 5x5x1280 feature map -> 25 timesteps of 1280 features.
    assert tuple(seq.output.shape) == (None, 25, 1280)

    # Two bidirectional LSTM layers, per the proposal.
    assert tuple(model.get_layer("bilstm_1").output.shape) == (None, 25, 256)
    assert tuple(model.get_layer("bilstm_2").output.shape) == (None, 128)
    assert model.output_shape == (None, 5)


def test_static_model_predicts_probabilities():
    from modules.recognition.cnn_lstm import build_static_model

    model = build_static_model(num_classes=4, augment=False)
    batch = np.random.randint(0, 255, (2, 160, 160, 3)).astype("float32")
    probs = model.predict(batch, verbose=0)

    assert probs.shape == (2, 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_backbone_frozen_by_default():
    from modules.recognition.cnn_lstm import build_static_model

    model = build_static_model(num_classes=3, augment=False, freeze_backbone=True)
    backbone = next(l for l in model.layers if l.name.startswith("mobilenetv2"))
    assert backbone.trainable is False


def test_select_class_files_interleaves_groups(tmp_path):
    """Both the webcam-scene and close-up groups must be represented."""
    class_dir = tmp_path / "A"
    class_dir.mkdir()
    for i in range(5):
        (class_dir / f"{i}.jpg").touch()          # scene group (digit-named)
        (class_dir / f"A ({i}).jpg").touch()      # close-up group (letter-named)

    picked = _select_class_files(class_dir, per_class_limit=4)
    names = [p.name for p in picked]

    assert len(picked) == 4
    assert any(n[0].isdigit() for n in names)
    assert any(not n[0].isdigit() for n in names)


def test_select_class_files_no_limit_returns_all(tmp_path):
    class_dir = tmp_path / "B"
    class_dir.mkdir()
    for i in range(3):
        (class_dir / f"{i}.jpg").touch()

    assert len(_select_class_files(class_dir, per_class_limit=None)) == 3


def test_train_val_split_is_a_partition_covering_all_classes(tmp_path):
    """Regression test for the train/val split.

    Keras shuffles the file list before slicing off the validation portion, and
    only when ``shuffle=True``. Using different shuffle flags for the two
    subsets silently produced a validation set that covered just the
    alphabetically last classes and overlapped the training data.
    """
    cv2 = pytest.importorskip("cv2")
    from modules.recognition.crop_dataset import load_crop_datasets

    classes = ["A", "B", "C", "D"]
    per_class = 20
    for cls in classes:
        d = tmp_path / cls
        d.mkdir()
        for i in range(per_class):
            img = np.full((32, 32, 3), (classes.index(cls) * 60) % 255, dtype=np.uint8)
            cv2.imwrite(str(d / f"{cls}_{i}.jpg"), img)

    train_ds, val_ds, names = load_crop_datasets(
        str(tmp_path), image_size=32, batch_size=8, validation_split=0.25
    )

    y_train = np.concatenate([y.numpy() for _, y in train_ds])
    y_val = np.concatenate([y.numpy() for _, y in val_ds])

    assert names == classes
    # No sample is dropped or duplicated across the two subsets.
    assert len(y_train) + len(y_val) == len(classes) * per_class
    # Every class must be represented on both sides of the split.
    assert set(y_train.tolist()) == set(range(len(classes)))
    assert set(y_val.tolist()) == set(range(len(classes)))
