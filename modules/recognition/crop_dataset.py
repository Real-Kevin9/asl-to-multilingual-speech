"""Hand-crop dataset preparation for the CNN+LSTM recogniser.

The raw ASL-alphabet folder contains two visually different groups of images:

* digit-named files (``1.jpg``, ``2.jpg``, ...) — full webcam scenes showing the
  whole upper body with a raised hand;
* letter-named files (``A (1).jpg``, ...) — the standard Kaggle close-up hand
  photos.

Feeding those directly to a CNN would teach it backgrounds and faces rather than
handshapes, and would not match live webcam frames. So every image is passed
through MediaPipe once, cropped to a padded square around the hand, resized, and
cached to disk. Training then reads the cached crops, which is both consistent
and fast (MediaPipe runs only once, not once per epoch).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from modules.recognition.hand_crop import DEFAULT_CROP_SIZE, crop_hand
from modules.recognition.preprocess import ASLPreprocessor


def _select_class_files(class_dir: Path, per_class_limit: Optional[int]) -> List[Path]:
    """Pick files for one class, interleaving the two image groups.

    Alternating between the webcam-scene group and the Kaggle close-up group
    keeps both framings represented even when ``per_class_limit`` is small,
    which matters for generalising to live webcam input.
    """
    files = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
    if per_class_limit is None:
        return files

    scenes = [f for f in files if image_group(f) == "scene"]
    closeups = [f for f in files if image_group(f) != "scene"]

    selected: List[Path] = []
    i = 0
    while len(selected) < per_class_limit and (i < len(scenes) or i < len(closeups)):
        if i < len(scenes):
            selected.append(scenes[i])
        if len(selected) < per_class_limit and i < len(closeups):
            selected.append(closeups[i])
        i += 1

    return selected[:per_class_limit]


def image_group(path: Path) -> str:
    """Classify a source image into one of the two visual domains.

    ``scene``   whole upper body with a raised hand, i.e. webcam framing. These
                are the digit-named corpus files and anything captured by
                ``scripts/capture_samples.py`` (``user_`` prefix).
    ``closeup`` hand filling the frame (the letter-named Kaggle files).
    """
    name = path.name
    if name.startswith("user_") or name[0].isdigit():
        return "scene"
    return "closeup"


def _select_by_group(
    class_dir: Path,
    scene_limit: Optional[int],
    closeup_limit: Optional[int],
) -> List[Path]:
    """Select files with an explicit per-group quota.

    Held-out testing showed the recogniser was far weaker on the ``scene`` group
    (webcam-style framing) than on ``closeup``, so the two groups need to be
    balanced deliberately rather than left to whatever the directory order gives.
    """
    files = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
    scenes = [f for f in files if image_group(f) == "scene"]
    closeups = [f for f in files if image_group(f) == "closeup"]

    if scene_limit is not None:
        scenes = scenes[:scene_limit]
    if closeup_limit is not None:
        closeups = closeups[:closeup_limit]
    return scenes + closeups


def build_crop_cache(
    raw_dir: str,
    cache_dir: str,
    per_class_limit: Optional[int] = 150,
    size: int = DEFAULT_CROP_SIZE,
    model_asset_path: Optional[str] = "models/recognition/hand_landmarker.task",
    padding: float = 0.35,
    scene_limit: Optional[int] = None,
    closeup_limit: Optional[int] = None,
) -> Tuple[int, int, List[str]]:
    """Create the cached hand-crop dataset.

    Alongside each crop the source landmark vector is stored (``landmarks.npy``
    plus ``manifest.json``). The hybrid recogniser consumes both streams, and
    caching the landmarks means MediaPipe does not have to run again at training
    time.

    Args:
        per_class_limit: Interleaved quota across both groups (legacy behaviour).
        scene_limit / closeup_limit: Explicit per-group quotas; when either is
            given they take precedence over ``per_class_limit``.

    Returns:
        ``(written, skipped_no_hand, class_names)``. Images where MediaPipe finds
        no hand are skipped rather than cached as a centre crop, so the CNN is
        not trained on frames that contain no visible hand.
    """
    raw_path = Path(raw_dir)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    by_group = scene_limit is not None or closeup_limit is not None

    preprocessor = ASLPreprocessor(model_asset_path=model_asset_path)
    written = 0
    skipped = 0
    class_names: List[str] = []
    items: List[dict] = []
    landmark_rows: List[np.ndarray] = []

    try:
        for class_dir in sorted(p for p in raw_path.iterdir() if p.is_dir()):
            label = class_dir.name
            class_names.append(label)
            out_dir = cache_path / label
            out_dir.mkdir(parents=True, exist_ok=True)

            if by_group:
                files = _select_by_group(class_dir, scene_limit, closeup_limit)
            else:
                files = _select_class_files(class_dir, per_class_limit)

            kept = 0
            for src in files:
                image = cv2.imread(str(src))
                if image is None:
                    skipped += 1
                    continue

                landmarks = preprocessor.extract_landmarks(str(src))
                if not np.any(landmarks):
                    # No hand detected — this frame carries no usable signal.
                    skipped += 1
                    continue

                crop = crop_hand(image, landmarks, size=size, padding=padding)
                rel = f"{label}/{src.stem}.jpg"
                cv2.imwrite(str(cache_path / rel), crop)

                items.append({"file": rel, "label": label, "group": image_group(src)})
                landmark_rows.append(landmarks.astype(np.float32))
                written += 1
                kept += 1

            print(f"  {label}: cached {kept}/{len(files)} crops")
    finally:
        preprocessor.close()

    manifest = {
        "classes": sorted({it["label"] for it in items}),
        "crop_size": size,
        "items": items,
    }
    (cache_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.save(cache_path / "landmarks.npy", np.stack(landmark_rows) if landmark_rows else np.zeros((0, 63), np.float32))
    print(f"Wrote manifest ({len(items)} items) and landmarks.npy")

    return written, skipped, class_names


def load_crop_datasets(
    cache_dir: str,
    image_size: int = DEFAULT_CROP_SIZE,
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
):
    """Load cached crops as training/validation ``tf.data`` datasets.

    Returns:
        ``(train_ds, val_ds, class_names)``. Images stay in 0-255 float range;
        rescaling to ``[-1, 1]`` happens inside the model.
    """
    from tensorflow import keras

    # Skip class folders that ended up empty. The ``nothing`` class is the usual
    # case: those frames contain no hand by definition, so no crop can exist.
    # "no hand detected" is handled directly at inference instead of being a
    # class the CNN has to learn.
    cache_path = Path(cache_dir)
    non_empty = sorted(
        d.name for d in cache_path.iterdir() if d.is_dir() and any(d.glob("*.jpg"))
    )
    skipped = sorted(
        d.name for d in cache_path.iterdir() if d.is_dir() and not any(d.glob("*.jpg"))
    )
    if skipped:
        print(f"Skipping empty class folders: {skipped}")

    common = dict(
        directory=cache_dir,
        labels="inferred",
        label_mode="int",
        class_names=non_empty,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        seed=seed,
        validation_split=validation_split,
    )

    # Both subsets MUST use the same shuffle flag and seed. Keras shuffles the
    # file list *before* slicing off the validation portion, and only when
    # shuffle=True. Passing shuffle=False here would take the alphabetically
    # last files instead — a validation set covering only the last few classes
    # and overlapping the training split.
    train_ds = keras.utils.image_dataset_from_directory(subset="training", shuffle=True, **common)
    val_ds = keras.utils.image_dataset_from_directory(subset="validation", shuffle=True, **common)
    class_names = list(train_ds.class_names)

    import tensorflow as tf

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def _contiguous_split(items: List[dict], validation_split: float) -> Tuple[List[int], List[int]]:
    """Split by trailing block within each (class, group), not at random.

    The ``scene`` images are consecutive frames from a continuous recording, so
    neighbouring files are almost identical. A random split therefore leaks
    near-duplicates into validation and reports an accuracy the model cannot
    reproduce on genuinely new footage — which is exactly what happened with the
    first CNN (96.8 % validation, 46 % on unseen scene images).

    Taking a contiguous trailing block per class and group keeps
    temporally-adjacent frames on the same side of the split.
    """
    buckets: Dict[Tuple[str, str], List[int]] = {}
    for idx, item in enumerate(items):
        buckets.setdefault((item["label"], item["group"]), []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for key in sorted(buckets):
        group_indices = buckets[key]  # already in cache-write (sorted filename) order
        n_val = int(round(len(group_indices) * validation_split))
        n_val = min(max(n_val, 1), max(len(group_indices) - 1, 1)) if len(group_indices) > 1 else 0
        split_at = len(group_indices) - n_val
        train_idx.extend(group_indices[:split_at])
        val_idx.extend(group_indices[split_at:])

    return train_idx, val_idx


def load_hybrid_datasets(
    cache_dir: str,
    image_size: int = DEFAULT_CROP_SIZE,
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
):
    """Load ``((crop, landmarks), label)`` datasets for the hybrid recogniser.

    Returns:
        ``(train_ds, val_ds, class_names, stats)``.
    """
    import tensorflow as tf

    from modules.recognition.features import normalize_landmark_features

    cache_path = Path(cache_dir)
    manifest_path = cache_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Rebuild the cache with scripts/prepare_hand_crops.py."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest["items"]
    landmarks = np.load(cache_path / "landmarks.npy")

    # Same geometric normalization the landmark MLP uses: wrist-centred and
    # scaled, so the stream is translation- and scale-invariant.
    landmarks = normalize_landmark_features(landmarks.astype(np.float32))

    class_names = sorted({it["label"] for it in items})
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    train_idx, val_idx = _contiguous_split(items, validation_split)

    def build(indices: List[int], training: bool):
        paths = [str(cache_path / items[i]["file"]) for i in indices]
        lms = landmarks[indices]
        labels = [class_to_idx[items[i]["label"]] for i in indices]
        return make_dataset(paths, lms, labels, image_size, batch_size, training, seed)

    stats = {
        "total": len(items),
        "train": len(train_idx),
        "val": len(val_idx),
        "val_by_group": {
            g: sum(1 for i in val_idx if items[i]["group"] == g)
            for g in sorted({it["group"] for it in items})
        },
        "train_by_group": {
            g: sum(1 for i in train_idx if items[i]["group"] == g)
            for g in sorted({it["group"] for it in items})
        },
    }

    return build(train_idx, True), build(val_idx, False), class_names, stats


def make_dataset(
    paths: List[str],
    landmarks: np.ndarray,
    labels: List[int],
    image_size: int = DEFAULT_CROP_SIZE,
    batch_size: int = 32,
    training: bool = False,
    seed: int = 42,
):
    """Build a ``((crop, landmarks), label)`` dataset from explicit lists.

    Shared by normal training and by user-data fine-tuning so both paths apply
    identical loading and augmentation.
    """
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((paths, landmarks, labels))

    def _degrade_resolution(img):
        """Randomly downscale then upscale, simulating a small hand crop.

        In webcam-scene frames the hand occupies a small part of a 480x640
        image, so its crop is heavily upscaled and soft. Close-up training
        images are sharp. Without this the CNN only ever learns sharp hands and
        collapses on real webcam input — the measured 46 % scene accuracy.
        Applied to half the training samples so the model sees both sharp and
        degraded versions.
        """
        scale = tf.random.uniform([], 0.3, 1.0)
        small = tf.maximum(tf.cast(float(image_size) * scale, tf.int32), 24)
        img = tf.image.resize(img, (small, small), method="area")
        return tf.image.resize(img, (image_size, image_size), method="bilinear")

    def _load(path, lm, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, (image_size, image_size))
        img = tf.cast(img, tf.float32)

        if training:
            img = tf.cond(
                tf.random.uniform([]) < 0.5,
                lambda: _degrade_resolution(img),
                lambda: img,
            )
            img = tf.clip_by_value(img, 0.0, 255.0)

        return (img, lm), label

    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
