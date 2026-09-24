"""The train/held-out split over your own captured samples.

`scripts/finetune_on_user_data.py` fine-tunes on part of
``data/raw/user_samples/test`` and holds the rest back. Any evaluation of
``hybrid_user`` has to use exactly that held-out portion, otherwise the model is
scored on images it trained on and the comparison against the other backends is
meaningless.

The split is deterministic — a trailing slice of each class in crop-cache
manifest order, no RNG — so it can be reconstructed after the fact rather than
stored. This module is the single definition of it; the fine-tune script and the
evaluation script both import from here so the two cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_USER_CACHE = "data/processed/user_crops"
DEFAULT_USER_DIR = "data/raw/user_samples/test"

# Matches the --val-fraction default of scripts/finetune_on_user_data.py.
DEFAULT_VAL_FRACTION = 0.4


def split_per_class(
    items: Sequence[Dict[str, str]],
    val_fraction: float,
) -> Tuple[List[int], List[int]]:
    """Hold out a trailing block of each class; frames are sequential.

    Returns positions into ``items``, not manifest row numbers.
    """
    buckets: Dict[str, List[int]] = {}
    for i, it in enumerate(items):
        buckets.setdefault(it["label"], []).append(i)
    train: List[int] = []
    val: List[int] = []
    for label in sorted(buckets):
        idx = buckets[label]
        n_val = max(1, int(round(len(idx) * val_fraction))) if len(idx) > 1 else 0
        train.extend(idx[: len(idx) - n_val])
        val.extend(idx[len(idx) - n_val:])
    return train, val


def load_manifest_items(cache_dir: str = DEFAULT_USER_CACHE) -> List[Dict[str, str]]:
    """Load the crop-cache manifest items in cache order."""
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. The held-out split is derived from the crop "
            "cache; rebuild it with scripts/finetune_on_user_data.py --rebuild-cache."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))["items"]


def _kept_items(
    items: Sequence[Dict[str, str]],
    class_names: Optional[Iterable[str]],
) -> List[Dict[str, str]]:
    if class_names is None:
        return list(items)
    allowed = set(class_names)
    return [it for it in items if it["label"] in allowed]


def held_out_user_files(
    user_cache: str = DEFAULT_USER_CACHE,
    user_dir: str = DEFAULT_USER_DIR,
    class_names: Optional[Iterable[str]] = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    expected_total: Optional[int] = None,
) -> List[Path]:
    """Raw image paths the fine-tune never trained on.

    Args:
        user_cache: Crop cache whose manifest order defines the split.
        user_dir: Raw sample directory the crops were built from.
        class_names: Restrict to the model's classes, as the fine-tune does.
        val_fraction: Must match the fine-tune's ``--val-fraction``.
        expected_total: If given (``user_train + user_val`` from the model
            metadata), the cache is rejected when it no longer matches. A
            rebuilt cache would silently reconstruct a different split.

    Raises:
        FileNotFoundError: No manifest, or none of the raw images resolve.
        ValueError: The cache no longer matches ``expected_total``.
    """
    items = _kept_items(load_manifest_items(user_cache), class_names)
    if expected_total is not None and len(items) != expected_total:
        raise ValueError(
            f"Crop cache at {user_cache} holds {len(items)} usable items but the model "
            f"metadata was written against {expected_total}. The cache has been rebuilt "
            "since fine-tuning, so the held-out split cannot be reconstructed. Re-run "
            "scripts/finetune_on_user_data.py before evaluating."
        )

    _, val_positions = split_per_class(items, val_fraction)

    root = Path(user_dir)
    paths: List[Path] = []
    missing = 0
    for pos in val_positions:
        # Crops are cached as "<label>/<source stem>.jpg" (see build_crop_cache),
        # so the original frame is recoverable from the crop's own name.
        rel = Path(items[pos]["file"])
        source = root / rel.parent.name / f"{rel.stem}.jpg"
        if source.exists():
            paths.append(source)
        else:
            missing += 1

    if not paths:
        raise FileNotFoundError(
            f"None of the {len(val_positions)} held-out crops map back to images under "
            f"{user_dir}. Check that --user-dir matches the directory used for fine-tuning."
        )
    if missing:
        print(f"[user_split] {missing} held-out crops have no source image under {user_dir}.")

    return paths
