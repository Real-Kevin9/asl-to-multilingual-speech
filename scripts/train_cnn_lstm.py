from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.cnn_lstm_trainer import CNNLSTMTrainer
from modules.recognition.hand_crop import DEFAULT_CROP_SIZE


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the MobileNetV2 + BiLSTM sign recogniser on cached hand crops"
    )
    parser.add_argument("--cache-dir", default="data/processed/hand_crops")
    parser.add_argument("--output-dir", default="models/recognition")
    parser.add_argument("--image-size", type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lstm-units", type=int, nargs=2, default=[128, 64])
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=0,
        help="Extra epochs unfreezing the MobileNetV2 top (slower; 0 disables).",
    )
    parser.add_argument("--fine-tune-at", type=int, default=100)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    args = parser.parse_args()

    if not Path(args.cache_dir).exists():
        print(f"ERROR: crop cache not found at {args.cache_dir}")
        print("Run scripts/prepare_hand_crops.py first.")
        sys.exit(1)

    trainer = CNNLSTMTrainer(output_dir=args.output_dir)
    metrics = trainer.train(
        cache_dir=args.cache_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        lstm_units=tuple(args.lstm_units),
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_at=args.fine_tune_at,
        fine_tune_lr=args.fine_tune_lr,
    )

    print("\n=== Summary ===")
    print(f"val_accuracy : {metrics['val_accuracy']:.4f}")
    print(f"macro F1     : {metrics['macro_f1']:.4f}")
    print(f"weighted F1  : {metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
