"""Measure recogniser accuracy on your own captured webcam samples.

This is the number that actually matters. Accuracy on the public corpus says how
well a model fits somebody else's recordings; this says how well it works for
you, on your camera, in your room.

By default only the images ``hybrid_user`` was **not** fine-tuned on are scored.
The fine-tune trains on 60 % of this same directory, so scoring the whole folder
would inflate ``hybrid_user`` while every other backend is measured on unseen
data — a comparison that looks decisive and means nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np  # noqa: E402

from modules.recognition.preprocess import ASLPreprocessor  # noqa: E402
from modules.recognition.registry import available_backends, create_recognizer  # noqa: E402
from modules.recognition.user_split import (  # noqa: E402
    DEFAULT_USER_CACHE,
    DEFAULT_VAL_FRACTION,
    held_out_user_files,
)

HAND_MODEL = "models/recognition/hand_landmarker.task"
FINETUNED_BACKENDS = ("hybrid_user",)


def collect_all_samples(root: Path):
    samples = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(class_dir.glob("*.jpg")):
            samples.append((img, class_dir.name))
    return samples


def collect_holdout_samples(root: Path, user_cache: str, val_fraction: float):
    """The fine-tune's held-out portion, reconstructed from the crop cache."""
    metadata_path = Path("models/recognition/hybrid_user_metadata.json")
    class_names = None
    expected_total = None
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        class_names = meta.get("class_names")
        finetune = meta.get("user_finetune") or {}
        if finetune.get("user_train") is not None and finetune.get("user_val") is not None:
            expected_total = finetune["user_train"] + finetune["user_val"]

    paths = held_out_user_files(
        user_cache=user_cache,
        user_dir=str(root),
        class_names=class_names,
        val_fraction=val_fraction,
        expected_total=expected_total,
    )
    return [(p, p.parent.name) for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate backends on your own samples")
    parser.add_argument("--data-dir", default="data/raw/user_samples/test")
    parser.add_argument("--backends", nargs="+", default=None,
                        help="Defaults to every backend with a trained model.")
    parser.add_argument("--report", default="logs/evaluation/user_data_accuracy.json")
    parser.add_argument("--all-samples", action="store_true",
                        help="Score every image instead of only the held-out portion. "
                             "Inflates hybrid_user, which trained on most of them.")
    parser.add_argument("--user-cache", default=DEFAULT_USER_CACHE,
                        help="Crop cache whose manifest order defines the held-out split.")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
                        help="Must match the --val-fraction used when fine-tuning.")
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.exists():
        print(f"No data at {root}. Record some first:")
        print("  ./.venv/bin/python scripts/capture_samples.py --split test")
        sys.exit(1)

    subset = "all" if args.all_samples else "held_out"
    if args.all_samples:
        samples = collect_all_samples(root)
        print("WARNING: scoring every image. Any backend fine-tuned on this directory "
              "(hybrid_user) trained on most of them, so its accuracy is not comparable.")
    else:
        try:
            samples = collect_holdout_samples(root, args.user_cache, args.val_fraction)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Cannot reconstruct the held-out split: {exc}")
            print("Re-run with --all-samples if you accept the contaminated comparison.")
            sys.exit(1)

    if not samples:
        print(f"No images found under {root}.")
        sys.exit(1)

    print(f"Evaluating on {len(samples)} of your own images "
          f"across {len({l for _, l in samples})} classes "
          f"[subset: {subset}]\n")

    pre = ASLPreprocessor(model_asset_path=HAND_MODEL)
    prepared = []
    no_hand = 0
    for path, label in samples:
        lm = pre.extract_landmarks(str(path))
        if not np.any(lm) and label != "nothing":
            no_hand += 1
            continue
        prepared.append((path, label, lm))
    pre.close()
    print(f"Hand detected in {len(prepared)}/{len(samples)} (skipped {no_hand})\n")

    backends = args.backends or available_backends()
    results = {}
    for backend in backends:
        try:
            rec = create_recognizer(backend)
        except Exception as exc:
            print(f"Skipping {backend}: {exc}")
            continue

        correct = 0
        per_class = defaultdict(lambda: [0, 0])
        confusions = defaultdict(int)
        for path, label, lm in prepared:
            pred = rec.predict_frame(image_path=str(path), landmarks=lm)
            per_class[label][1] += 1
            if pred["label"] == label:
                correct += 1
                per_class[label][0] += 1
            else:
                confusions[f"{label}->{pred['label']}"] += 1

        acc = correct / len(prepared) if prepared else 0.0
        weak = sorted(
            ((c, v[0] / v[1]) for c, v in per_class.items() if v[1]),
            key=lambda kv: kv[1],
        )[:8]

        contaminated = args.all_samples and backend in FINETUNED_BACKENDS
        results[backend] = {
            "accuracy": acc,
            "correct": correct,
            "total": len(prepared),
            "contaminated": contaminated,
            "weakest_classes": {c: round(a, 3) for c, a in weak},
            "top_confusions": dict(sorted(confusions.items(), key=lambda kv: -kv[1])[:12]),
        }

        flag = "  [TRAINED ON THIS DATA]" if contaminated else ""
        print(f"--- {backend} ---{flag}")
        print(f"  accuracy : {acc*100:.1f}%  ({correct}/{len(prepared)})")
        print(f"  weakest  : {[(c, f'{a*100:.0f}%') for c, a in weak[:6]]}")
        print(f"  confusions: {list(results[backend]['top_confusions'].items())[:6]}\n")

    # Scalar/metadata entries are ignored by the per-backend report parsers.
    results["_evaluation"] = {
        "subset": subset,
        "val_fraction": args.val_fraction if not args.all_samples else None,
        "n_images": len(samples),
        "n_scored": len(prepared),
        "data_dir": str(root),
        "note": (
            "Held-out portion of the user samples; hybrid_user never trained on these."
            if subset == "held_out"
            else "Every user sample, including images hybrid_user was fine-tuned on."
        ),
    }

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved report to {out}")


if __name__ == "__main__":
    main()
