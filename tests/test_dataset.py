import numpy as np
from sklearn.preprocessing import LabelEncoder

from modules.recognition.dataset import load_preprocessed_data


def test_load_preprocessed_data(tmp_path):
    features = np.array([[0.0] * 63, [1.0] * 63], dtype=np.float32)
    labels = ["A", "B"]

    np.save(tmp_path / "features.npy", features)
    with open(tmp_path / "labels.txt", "w", encoding="utf-8") as handle:
        handle.write("A\nB\n")

    loaded_features, loaded_labels, encoder = load_preprocessed_data(
        str(tmp_path / "features.npy"),
        str(tmp_path / "labels.txt"),
    )

    assert loaded_features.shape == (2, 63)
    assert loaded_labels.shape == (2,)
    assert isinstance(encoder, LabelEncoder)
    assert set(encoder.classes_) == {"A", "B"}
