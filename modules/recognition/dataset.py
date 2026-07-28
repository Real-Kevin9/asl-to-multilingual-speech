from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_preprocessed_data(
    features_path: str,
    labels_path: str,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    features_file = Path(features_path)
    labels_file = Path(labels_path)

    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    features = np.load(features_file, allow_pickle=True)
    if features.dtype == object:
        features = np.stack(features)

    with labels_file.open("r", encoding="utf-8") as handle:
        labels = [line.strip() for line in handle if line.strip()]

    if len(features) != len(labels):
        raise ValueError("Number of features and labels do not match.")

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    return features.astype(np.float32), encoded_labels.astype(np.int64), encoder
