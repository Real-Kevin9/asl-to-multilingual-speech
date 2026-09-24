"""Train the WLASL MobileNetV2 + BiLSTM word-level recogniser."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.keras_env import configure_keras_home  # noqa: E402

configure_keras_home()

from modules.recognition.wlasl_data import DEFAULT_PROCESSED_DIR  # noqa: E402
from modules.recognition.wlasl_trainer import WLASLTrainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WLASL video recogniser")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", default="models/recognition")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--clip-batch-size", type=int, default=12,
                        help="Clips loaded at once during feature extraction (lower = less RAM).")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--aug-copies", type=int, default=1)
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()

    trainer = WLASLTrainer(output_dir=args.output_dir)
    report = trainer.train(
        processed_dir=args.processed_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        clip_batch_size=args.clip_batch_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        aug_copies=args.aug_copies,
        rebuild_features=args.rebuild_features,
    )
    print(json.dumps({
        "num_classes": report["num_classes"],
        "val_accuracy": report["val_accuracy"],
        "val_macro_recall": report.get("val_macro_recall"),
        "signs_met": report["signs_met"],
        "accuracy_met": report["accuracy_met"],
        "model_path": report["model_path"],
        "head_path": report.get("head_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
