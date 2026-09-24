"""Download FER2013-enhanced and prepare the 4-class emotion arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.emotion.data import (  # noqa: E402
    DEFAULT_PROCESSED_DIR,
    HF_DATASET_ID,
    download_and_prepare,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 4-class FER2013 for emotion CNN")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--dataset-id", default=HF_DATASET_ID)
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        help="Optional cap per split (for smoke tests).",
    )
    args = parser.parse_args()

    max_map = None
    if args.max_per_split is not None:
        max_map = {"train": args.max_per_split, "validation": args.max_per_split,
                   "test": args.max_per_split, "val": args.max_per_split}

    print(f"Downloading {args.dataset_id} and filtering to happy/sad/angry/neutral ...")
    meta = download_and_prepare(
        processed_dir=args.processed_dir,
        dataset_id=args.dataset_id,
        max_per_split=max_map,
    )
    print(json.dumps(meta["stats"], indent=2))
    print(f"\nWrote arrays under {args.processed_dir}")
    print("Next: ./.venv/bin/python scripts/train_emotion.py")


if __name__ == "__main__":
    main()
