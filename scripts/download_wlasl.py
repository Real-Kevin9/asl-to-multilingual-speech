"""Download and preprocess a top-K WLASL subset for word-level recognition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.wlasl_data import (  # noqa: E402
    DEFAULT_FRAME_SIZE,
    DEFAULT_FRAMES,
    DEFAULT_NUM_CLASSES,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    prepare_wlasl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download top-K WLASL glosses and cache frame tensors",
    )
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--frame-size", type=int, default=DEFAULT_FRAME_SIZE)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--train-views",
        type=int,
        default=4,
        help="Temporal windows per training video (data multiplication).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse videos already under data/raw/wlasl/videos/",
    )
    args = parser.parse_args()

    meta = prepare_wlasl(
        num_classes=args.num_classes,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_workers=args.max_workers,
        skip_download=args.skip_download,
        train_views=args.train_views,
    )
    keys = (
        "num_classes", "n_train", "n_val", "n_train_videos", "num_frames",
        "frame_size", "train_views", "licence",
    )
    print(json.dumps({k: meta[k] for k in keys if k in meta}, indent=2))
    print("\nNext: ./.venv/bin/python scripts/train_wlasl.py")


if __name__ == "__main__":
    main()
