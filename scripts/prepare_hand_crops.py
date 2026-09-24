from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.crop_dataset import build_crop_cache
from modules.recognition.hand_crop import DEFAULT_CROP_SIZE


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the cached MediaPipe hand-crop dataset for CNN+LSTM training"
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw/asl_alphabet/asl_alphabet_train",
        help="Directory containing one subfolder per class.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/processed/hand_crops",
        help="Where the cropped images are written.",
    )
    parser.add_argument(
        "--per-class-limit",
        type=int,
        default=150,
        help="Images per class to crop (interleaves webcam-scene and close-up groups).",
    )
    parser.add_argument(
        "--scene-limit",
        type=int,
        default=None,
        help="Images per class from the webcam-scene group (digit-named files). "
             "Overrides --per-class-limit when set.",
    )
    parser.add_argument(
        "--closeup-limit",
        type=int,
        default=None,
        help="Images per class from the Kaggle close-up group (letter-named files).",
    )
    parser.add_argument("--size", type=int, default=DEFAULT_CROP_SIZE, help="Output crop size.")
    parser.add_argument("--padding", type=float, default=0.35, help="Bounding-box padding fraction.")
    parser.add_argument(
        "--model-path",
        default="models/recognition/hand_landmarker.task",
        help="MediaPipe hand_landmarker.task path.",
    )
    args = parser.parse_args()

    model_path = args.model_path if Path(args.model_path).exists() else None
    if model_path is None:
        print(f"ERROR: MediaPipe model not found at {args.model_path}.")
        print("Hand cropping requires it. See MEDIAPIPE_SETUP.md.")
        sys.exit(1)

    print(f"Input      : {args.input_dir}")
    print(f"Cache dir  : {args.cache_dir}")
    print(f"Crop size  : {args.size}px  padding={args.padding}")
    if args.scene_limit is not None or args.closeup_limit is not None:
        print(f"Per class  : scene={args.scene_limit}  closeup={args.closeup_limit}\n")
    else:
        print(f"Per class  : {args.per_class_limit} (interleaved)\n")

    written, skipped, classes = build_crop_cache(
        raw_dir=args.input_dir,
        cache_dir=args.cache_dir,
        per_class_limit=args.per_class_limit,
        size=args.size,
        model_asset_path=model_path,
        padding=args.padding,
        scene_limit=args.scene_limit,
        closeup_limit=args.closeup_limit,
    )

    total = written + skipped
    pct = (100.0 * written / total) if total else 0.0
    print(f"\nDone. Cached {written} crops across {len(classes)} classes.")
    print(f"Skipped {skipped} images with no detected hand ({pct:.1f}% usable).")


if __name__ == "__main__":
    main()
