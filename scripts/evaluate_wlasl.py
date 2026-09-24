"""Evaluate the trained WLASL word-level recogniser on the held-out val clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.keras_env import configure_keras_home  # noqa: E402

configure_keras_home()

from modules.recognition.wlasl_data import (  # noqa: E402
    DEFAULT_PROCESSED_DIR,
    iter_manifest,
    load_clip_array,
)
from modules.recognition.wlasl_landmarks import DEFAULT_LANDMARK_DIR, load_seq  # noqa: E402
from modules.recognition.wlasl_predictor import WLASLPredictor  # noqa: E402
from modules.recognition.wlasl_trainer import METADATA_FILENAME, MODEL_FILENAME  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WLASL video recogniser")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--model-dir", default="models/recognition")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument(
        "--out",
        default="logs/evaluation/wlasl_accuracy.json",
        help="Where to write the JSON report",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    meta_path = model_dir / METADATA_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    processed_dir = args.processed_dir
    if processed_dir is None:
        processed_dir = meta.get("processed_dir") or (
            DEFAULT_LANDMARK_DIR
            if str(meta.get("inference", "")).startswith("landmark")
            else DEFAULT_PROCESSED_DIR
        )

    predictor = WLASLPredictor(
        model_path=str(model_dir / MODEL_FILENAME),
        metadata_path=str(meta_path),
    )
    rows = iter_manifest(processed_dir, args.split)
    if not rows:
        raise SystemExit(f"No rows in {processed_dir}/{args.split}.json")

    y_true, y_pred = [], []
    per_class = {name: {"correct": 0, "total": 0} for name in predictor.class_names}
    use_landmarks = predictor._mode == "landmarks"

    for row in rows:
        if use_landmarks:
            seq = load_seq(row["clip_path"])
            result = predictor.predict_clip(clip_array=seq)
        else:
            clip = load_clip_array(row["clip_path"])
            result = predictor.predict_clip(clip_array=clip)
        truth = row["gloss"]
        pred = result["label"]
        y_true.append(truth)
        y_pred.append(pred)
        if truth in per_class:
            per_class[truth]["total"] += 1
            if truth == pred:
                per_class[truth]["correct"] += 1

    accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    recalls = {
        name: (stats["correct"] / stats["total"] if stats["total"] else None)
        for name, stats in per_class.items()
    }
    report = {
        "backend": predictor.backend_name,
        "processed_dir": processed_dir,
        "split": args.split,
        "n": len(rows),
        "num_classes": len(predictor.class_names),
        "accuracy": accuracy,
        "target_signs": 50,
        "target_accuracy": 0.85,
        "signs_met": len(predictor.class_names) >= 50,
        "accuracy_met": accuracy >= 0.85,
        "per_class_recall": recalls,
        "class_names": predictor.class_names,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "n": report["n"],
        "num_classes": report["num_classes"],
        "accuracy": round(accuracy, 4),
        "signs_met": report["signs_met"],
        "accuracy_met": report["accuracy_met"],
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
