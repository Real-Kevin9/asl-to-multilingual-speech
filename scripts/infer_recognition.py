from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repo root on path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.infer import predict_image_label


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer ASL label for a single image")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument(
        "--model",
        type=str,
        default="models/recognition/recognition_model.joblib",
        help="Path to trained model",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="models/recognition/label_encoder.joblib",
        help="Path to label encoder",
    )
    args = parser.parse_args()

    out = predict_image_label(args.image, args.model, args.encoder)
    print(out)


if __name__ == "__main__":
    main()
