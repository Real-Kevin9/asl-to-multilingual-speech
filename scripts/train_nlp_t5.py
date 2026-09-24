"""CLI: fine-tune t5-small on ASLG-PC12 gloss→English."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.nlp.trainer import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    train_from_processed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune t5-small on ASLG-PC12")
    parser.add_argument("--processed-dir", default="data/processed/aslg_pc12")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-train", type=int, default=20000)
    parser.add_argument("--max-val", type=int, default=2000)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(
        f"Fine-tuning {args.model_name} on ≤{args.max_train} train / "
        f"≤{args.max_val} val pairs for {args.epochs} epoch(s) ..."
    )
    meta = train_from_processed(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_train=args.max_train,
        max_val=args.max_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))
    print(f"\nSaved model to {args.output_dir}")
    print("Evaluate with: ./.venv/bin/python scripts/evaluate_nlp.py")


if __name__ == "__main__":
    main()
