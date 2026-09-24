"""FER2013 loading and remapping to the project's four emotion classes.

Proposal objective 4 targets ``happy / sad / angry / neutral``. Classic FER2013
has seven labels; disgust, fear and surprise are dropped so the classifier index
space matches ``modules.emotion.detector.EMOTIONS``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Stable order used by EmotionDetector and the trained CNN.
EMOTIONS = ["happy", "sad", "angry", "neutral"]

# FER2013 / fer2013-enhanced integer labels → name
FER7_NAMES = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

# Keep only proposal classes; map FER id → local class index in EMOTIONS.
FER7_TO_PROJECT = {
    0: EMOTIONS.index("angry"),
    3: EMOTIONS.index("happy"),
    4: EMOTIONS.index("sad"),
    6: EMOTIONS.index("neutral"),
}

HF_DATASET_ID = "abhilash88/fer2013-enhanced"
DEFAULT_PROCESSED_DIR = "data/processed/fer2013_4class"
IMAGE_SIZE = 48


def remap_fer_label(fer_id: int) -> Optional[int]:
    """Return project class index, or None if the FER label is dropped."""
    return FER7_TO_PROJECT.get(int(fer_id))


def image_to_array(image) -> np.ndarray:
    """Convert a HF/PIL image, pixel list, or ndarray to float32 (48, 48, 1) in [0, 1]."""
    if hasattr(image, "convert"):
        arr = np.asarray(image.convert("L"), dtype=np.float32)
    elif isinstance(image, str):
        arr = np.fromstring(image, sep=" ", dtype=np.float32)
    else:
        arr = np.asarray(image, dtype=np.float32)

    if arr.ndim == 1:
        side = int(np.sqrt(arr.size))
        arr = arr.reshape(side, side)
    elif arr.ndim == 3:
        arr = arr.mean(axis=-1)

    if arr.shape != (IMAGE_SIZE, IMAGE_SIZE):
        try:
            import cv2

            arr = cv2.resize(arr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        except Exception:
            from PIL import Image

            arr = np.asarray(
                Image.fromarray(arr.astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE)),
                dtype=np.float32,
            )
    if arr.max() > 1.0:
        # FER pixels are 0–255; tolerate out-of-range inputs by clipping after scale.
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr.reshape(IMAGE_SIZE, IMAGE_SIZE, 1).astype(np.float32)


def download_and_prepare(
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    dataset_id: str = HF_DATASET_ID,
    max_per_split: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """Download FER2013-enhanced, keep 4 classes, write ``.npz`` arrays per split."""
    from datasets import load_dataset

    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_id)
    stats: Dict[str, object] = {"dataset_id": dataset_id, "splits": {}}

    for split_name, split in ds.items():
        limit = None
        if max_per_split:
            limit = max_per_split.get(split_name)

        xs: List[np.ndarray] = []
        ys: List[int] = []
        for i, row in enumerate(split):
            if limit is not None and len(xs) >= limit:
                break
            fer_id = int(row.get("emotion", -1))
            mapped = remap_fer_label(fer_id)
            if mapped is None:
                continue
            img = row.get("image")
            if img is None and "pixels" in row:
                # Fallback: space-separated pixel string
                pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.float32)
                img = pixels.reshape(IMAGE_SIZE, IMAGE_SIZE)
            xs.append(image_to_array(img))
            ys.append(mapped)

        x_arr = np.stack(xs, axis=0) if xs else np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
        y_arr = np.asarray(ys, dtype=np.int64)
        np.savez_compressed(processed / f"{split_name}.npz", x=x_arr, y=y_arr)

        counts = {name: int(np.sum(y_arr == idx)) for idx, name in enumerate(EMOTIONS)}
        stats["splits"][split_name] = {
            "n": int(len(y_arr)),
            "counts": counts,
            "path": str(processed / f"{split_name}.npz"),
        }

    meta = {
        "emotions": EMOTIONS,
        "fer7_to_project": {str(k): v for k, v in FER7_TO_PROJECT.items()},
        "image_size": IMAGE_SIZE,
        "stats": stats,
    }
    (processed / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_split(
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(processed_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run scripts/download_fer2013.py first."
        )
    data = np.load(path)
    return data["x"], data["y"]


def available_splits(processed_dir: str = DEFAULT_PROCESSED_DIR) -> List[str]:
    root = Path(processed_dir)
    return sorted(p.stem for p in root.glob("*.npz"))
