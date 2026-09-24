"""Download and clean the ASLG-PC12 gloss↔English corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.nlp.data import (  # noqa: E402
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    HF_DATASET_ID,
    download_aslg_pc12,
    prepare_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ASLG-PC12 and write cleaned JSONL under data/processed/aslg_pc12/"
    )
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--dataset-id", default=HF_DATASET_ID)
    parser.add_argument("--max-train", type=int, default=20000)
    parser.add_argument("--max-val", type=int, default=2000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Downloading {args.dataset_id} ...")
    meta = download_aslg_pc12(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        dataset_id=args.dataset_id,
    )
    print(json.dumps(meta, indent=2))

    train, val, train_path, val_path = prepare_splits(
        processed_dir=args.processed_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_train=args.max_train,
        max_val=args.max_val,
    )
    print(f"Train split: {len(train)} → {train_path}")
    print(f"Val split  : {len(val)} → {val_path}")
    print("\nNext: ./.venv/bin/python scripts/train_nlp_t5.py")


if __name__ == "__main__":
    main()
