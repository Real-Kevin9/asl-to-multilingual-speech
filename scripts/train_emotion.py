"""CLI: train the facial emotion CNN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.emotion.trainer import DEFAULT_OUTPUT_DIR, EmotionTrainer  # noqa: E402
from modules.emotion.data import DEFAULT_PROCESSED_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 4-class emotion CNN on FER2013")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--arch",
        default="v2",
        choices=["v1", "v2"],
        help="v1 reproduces the Update 6 model; v2 adds augmentation and depth.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early-stopping patience on val_accuracy.",
    )
    args = parser.parse_args()

    print(
        f"Training emotion CNN {args.arch} "
        f"({args.epochs} epochs, batch {args.batch_size}) ..."
    )
    trainer = EmotionTrainer(output_dir=args.output_dir)
    meta = trainer.train(
        processed_dir=args.processed_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_train=args.max_train,
        seed=args.seed,
        arch=args.arch,
        patience=args.patience,
    )
    print(json.dumps({
        "val_accuracy": meta["val_accuracy"],
        "train_samples": meta["train_samples"],
        "val_samples": meta["val_samples"],
        "epochs_ran": meta["epochs_ran"],
        "model_path": meta["model_path"],
    }, indent=2))
    print(f"\nSaved model to {meta['model_path']}")
    print("Evaluate with: ./.venv/bin/python scripts/evaluate_emotion.py")


if __name__ == "__main__":
    main()
