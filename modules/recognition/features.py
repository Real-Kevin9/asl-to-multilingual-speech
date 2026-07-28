from __future__ import annotations

import numpy as np


def normalize_landmark_features(features: np.ndarray) -> np.ndarray:
    """Make hand-landmark vectors translation- and scale-invariant.

    Each sample is expected to be a flat 63-vector (21 landmarks x (x, y, z)).
    Landmarks are re-centred on the wrist (landmark 0) and divided by the
    largest wrist-to-landmark distance, so the same sign looks the same
    regardless of where the hand sits in the frame or how big it appears.

    Args:
        features: Array of shape (N, 63) or (63,).

    Returns:
        Normalized array with the same shape as the input. Inputs that are not
        63-dimensional are returned unchanged.
    """
    single = features.ndim == 1
    if single:
        features = features.reshape(1, -1)

    if features.ndim != 2 or features.shape[1] != 63:
        return features[0] if single else features

    pts = features.reshape(-1, 21, 3).astype(float)
    centered = pts - pts[:, 0:1, :]
    dists = np.linalg.norm(centered, axis=2)
    max_dist = dists.max(axis=1)
    max_dist[max_dist == 0] = 1.0
    scaled = centered / max_dist[:, None, None]
    out = scaled.reshape(features.shape)

    return out[0] if single else out
