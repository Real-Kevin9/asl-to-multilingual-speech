from __future__ import annotations

import argparse

import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.trainer import train_recognition_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASL recognition model")
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/preprocessed_features.npy",
        help="Path to preprocessed feature file.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="data/processed/labels.txt",
        help="Path to label file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/recognition",
        help="Directory to save trained model.",
    )
    args = parser.parse_args()

    train_recognition_model(args.features, args.labels, args.output_dir)


if __name__ == "__main__":
    main()
