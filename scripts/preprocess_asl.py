from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os
from urllib.request import urlopen
import tempfile

# Ensure project root is on sys.path when running as a script
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.preprocess import preprocess_dataset


# Default MediaPipe hand landmarker model (the Tasks API uses a .task bundle)
DEFAULT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEFAULT_MODEL_PATH = repo_root / "models" / "recognition" / "hand_landmarker.task"


def download_model(url: str, output_path: Path) -> Path:
    """Download MediaPipe model from URL if not already present."""
    if output_path.exists():
        print(f"Model already exists at {output_path}")
        return output_path

    print(f"Downloading MediaPipe hand landmarker model from {url}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urlopen(url) as response:
            with open(output_path, "wb") as out_file:
                out_file.write(response.read())
        print(f"Model downloaded to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the ASL alphabet dataset")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/asl_alphabet/asl_alphabet_train",
        help="Directory containing ASL alphabet class folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed feature metadata",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to hand_landmarker.task model file. Defaults to the bundled model "
        "if present; otherwise falls back to zero vectors.",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download the MediaPipe hand_landmarker.task model if it is missing.",
    )
    parser.add_argument(
        "--per-class-limit",
        type=int,
        default=None,
        help="Only process the first N images per class (quick outline dataset).",
    )
    args = parser.parse_args()

    # Resolve the model path: explicit arg > bundled default (if present).
    model_path: Path | None = None
    if args.model_path:
        model_path = Path(args.model_path)
    elif DEFAULT_MODEL_PATH.exists():
        model_path = DEFAULT_MODEL_PATH

    if args.download_model and model_path is not None and not model_path.exists():
        try:
            download_model(DEFAULT_MODEL_URL, model_path)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"Warning: model download failed ({exc}); using zero-vector fallback.")
            model_path = None

    if model_path is not None and model_path.exists():
        print(f"Using model: {model_path}")
    else:
        model_path = None
        print("No model available. Preprocessing will use fallback (zero vectors).")
        print("Run with --download-model or see MEDIAPIPE_SETUP.md to enable real landmarks.")

    print(f"Input directory: {args.input_dir}")
    if args.per_class_limit:
        print(f"Per-class limit: {args.per_class_limit} images")

    features, labels = preprocess_dataset(
        args.input_dir,
        fail_on_missing=False,
        model_asset_path=str(model_path) if model_path else None,
        per_class_limit=args.per_class_limit,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "preprocessed_features.npy"
    import numpy as np

    # Convert to numpy array if not already
    if not isinstance(features, np.ndarray):
        features = np.array(features, dtype=np.float32)
    
    np.save(output_path, features)
    with open(output_dir / "labels.txt", "w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(label + "\n")

    print(f"Saved {len(features)} samples to {output_path}")
    
    # Report feature statistics
    if features is not None and len(features) > 0:
        features_arr = np.array(features) if not isinstance(features, np.ndarray) else features
        nonzero_count = np.count_nonzero(features_arr)
        total_count = features_arr.size
        print(f"Feature statistics: {nonzero_count}/{total_count} non-zero values ({100*nonzero_count/total_count:.1f}%)")


if __name__ == "__main__":
    main()
