"""Train WLASL chase v4 (geometry + extra branches + frozen v3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.wlasl_chase_v4 import train_chase_v4  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="WLASL chase v4")
    parser.add_argument("--processed-dir", default="data/processed/wlasl_landmarks_v4")
    parser.add_argument("--output-dir", default="models/recognition/chase_v4")
    parser.add_argument("--expert-dir", default="models/experts/v3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-transformer-seeds", type=int, default=3)
    parser.add_argument("--n-tgcn-seeds", type=int, default=2)
    parser.add_argument("--aug-copies", type=int, default=4)
    args = parser.parse_args()
    report = train_chase_v4(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        expert_dir=args.expert_dir,
        epochs=args.epochs,
        n_transformer_seeds=args.n_transformer_seeds,
        n_tgcn_seeds=args.n_tgcn_seeds,
        aug_copies=args.aug_copies,
    )
    print(json.dumps({
        "val_accuracy": report["val_accuracy"],
        "accuracy_met": report["accuracy_met"],
        "beat_v3": report["beat_v3"],
        "blend_weights": report["blend_weights"],
    }, indent=2))


if __name__ == "__main__":
    main()
