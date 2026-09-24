"""Train boosted WLASL landmark models (multi-seed + TGCN ensemble)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.wlasl_boost_trainer import train_boosted_landmark_models  # noqa: E402
from modules.recognition.wlasl_landmarks import DEFAULT_LANDMARK_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Boosted WLASL landmark training")
    parser.add_argument("--processed-dir", default=DEFAULT_LANDMARK_DIR)
    parser.add_argument("--output-dir", default="models/recognition")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--aug-copies", type=int, default=6)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-tgcn-seeds", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--n-tta", type=int, default=7)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--tgcn-hidden", type=int, default=96)
    parser.add_argument("--tgcn-stages", type=int, default=12)
    parser.add_argument("--no-tgcn", action="store_true")
    args = parser.parse_args()

    report = train_boosted_landmark_models(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        aug_copies=args.aug_copies,
        n_seeds=args.n_seeds,
        train_tgcn_flag=not args.no_tgcn,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        n_tgcn_seeds=args.n_tgcn_seeds,
        n_tta=args.n_tta,
        transformer_lr=args.lr,
        tgcn_hidden=args.tgcn_hidden,
        tgcn_stages=args.tgcn_stages,
    )
    print(json.dumps({
        "val_accuracy": report["val_accuracy"],
        "accuracy_met": report["accuracy_met"],
        "signs_met": report["signs_met"],
        "blend_weights": report["blend_weights"],
        "transformer_val_accuracy": report["transformer_val_accuracy"],
        "tgcn_val_accuracy": report["tgcn_val_accuracy"],
    }, indent=2))


if __name__ == "__main__":
    main()
