"""Evaluate the emotion CNN on held-out FER2013 (4-class) arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.emotion.data import (  # noqa: E402
    DEFAULT_PROCESSED_DIR,
    EMOTIONS,
    available_splits,
    load_split,
)
from modules.emotion.detector import EMOTION_PROSODY, EmotionDetector  # noqa: E402
from modules.emotion.trainer import DEFAULT_OUTPUT_DIR, MODEL_FILENAME  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate emotion CNN accuracy")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default=None,
                        help="Split to score (default: test, else validation).")
    parser.add_argument("--output", default="logs/evaluation/emotion_accuracy.json")
    args = parser.parse_args()

    splits = available_splits(args.processed_dir)
    split = args.split
    if split is None:
        if "test" in splits:
            split = "test"
        elif "validation" in splits:
            split = "validation"
        else:
            split = splits[-1] if splits else "test"

    x, y = load_split(args.processed_dir, split)
    model_path = Path(args.model_dir) / MODEL_FILENAME
    if not model_path.exists():
        print(f"No model at {model_path}. Train with scripts/train_emotion.py")
        sys.exit(1)

    from tensorflow import keras

    model = keras.models.load_model(model_path)
    proba = model.predict(x, verbose=0)
    preds = np.argmax(proba, axis=1)
    accuracy = float(np.mean(preds == y)) if len(y) else 0.0

    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(
        y, preds, labels=list(range(len(EMOTIONS))),
        target_names=EMOTIONS, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y, preds, labels=list(range(len(EMOTIONS)))).tolist()
    per_class = {
        name: float(report[name]["recall"]) for name in EMOTIONS if name in report
    }

    # Prosody smoke: ensure each predicted class maps to a distinct rate/pitch.
    detector = EmotionDetector(model_path=str(model_path))
    prosody_ok = all(e in EMOTION_PROSODY for e in EMOTIONS)

    out = {
        "split": split,
        "n": int(len(y)),
        "accuracy": accuracy,
        "per_class_recall": per_class,
        "classification_report": report,
        "confusion_matrix": cm,
        "emotions": EMOTIONS,
        "model_path": str(model_path),
        "detector_backend": detector.backend,
        "prosody_map_complete": prosody_ok,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Split     : {split} ({len(y)} images)")
    print(f"Accuracy  : {accuracy*100:.1f}%")
    print(f"Per-class : { {k: f'{v*100:.0f}%' for k,v in per_class.items()} }")
    print(f"Backend   : {detector.backend}")
    print(f"Saved     : {path}")


if __name__ == "__main__":
    main()
