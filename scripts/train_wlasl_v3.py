"""Train WLASL landmark Transformer + HGB ensemble (v3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.wlasl_landmarks import DEFAULT_LANDMARK_DIR  # noqa: E402
from modules.recognition.wlasl_seq_trainer import train_landmark_models  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WLASL landmark models")
    parser.add_argument("--processed-dir", default=DEFAULT_LANDMARK_DIR)
    parser.add_argument("--output-dir", default="models/recognition")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--aug-copies", type=int, default=6)
    args = parser.parse_args()

    report = train_landmark_models(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        aug_copies=args.aug_copies,
    )
    print(json.dumps({
        "val_accuracy": report["val_accuracy"],
        "transformer_val_accuracy": report["transformer_val_accuracy"],
        "hgb_val_accuracy": report["hgb_val_accuracy"],
        "val_macro_recall": report["val_macro_recall"],
        "signs_met": report["signs_met"],
        "accuracy_met": report["accuracy_met"],
    }, indent=2))


if __name__ == "__main__":
    main()
