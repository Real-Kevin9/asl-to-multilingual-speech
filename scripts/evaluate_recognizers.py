"""Compare recognition backends on identical, never-seen images.

Update 1 reported 96.81 % for the CNN and ~93 % for the MLP, but those came from
different subsets and different representations, so they were not comparable.
This script fixes that: it builds a held-out set from images **neither** model
was trained on and runs both backends over exactly the same samples.

It also reports accuracy separately for the two image groups in the corpus:

* ``scene``   — digit-named files, whole upper body with a raised hand
                (what a laptop webcam actually looks like);
* ``closeup`` — letter-named Kaggle files, hand filling the frame.

That split is the important one. A model can score well overall while failing on
the framing that matters for live use.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np  # noqa: E402

from modules.recognition.crop_dataset import _select_by_group, _select_class_files  # noqa: E402
from modules.recognition.preprocess import ASLPreprocessor  # noqa: E402
from modules.recognition.registry import available_backends, create_recognizer  # noqa: E402

HAND_MODEL = "models/recognition/hand_landmarker.task"


def training_files(
    class_dir: Path,
    cnn_limit: int,
    mlp_limit: int,
    hybrid_scene: int,
    hybrid_closeup: int,
) -> set:
    """Files seen by **any** model, so they can be excluded.

    Every backend must be excluded, not just some. Each model was trained with a
    different selection rule, and missing one silently turns the "held-out" set
    into that model's training set.
    """
    seen = set()
    # CNN+LSTM: interleaved selection used by the first crop cache.
    seen.update(p.name for p in _select_class_files(class_dir, cnn_limit))
    # Landmark MLP: first N sorted (scripts/preprocess_asl.py).
    ordered = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
    seen.update(p.name for p in ordered[:mlp_limit])
    # Hybrid: explicit per-group quotas.
    seen.update(p.name for p in _select_by_group(class_dir, hybrid_scene, hybrid_closeup))
    return seen


def build_holdout(
    root: Path,
    per_group: int,
    cnn_limit: int,
    mlp_limit: int,
    hybrid_scene: int,
    hybrid_closeup: int,
) -> List[tuple]:
    """Return [(path, label, group)] of unseen images, balanced across groups."""
    samples = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        seen = training_files(class_dir, cnn_limit, mlp_limit, hybrid_scene, hybrid_closeup)
        files = sorted(class_dir.glob("*.jpg"))
        scenes = [f for f in files if f.name[0].isdigit() and f.name not in seen]
        closeups = [f for f in files if not f.name[0].isdigit() and f.name not in seen]
        for f in scenes[:per_group]:
            samples.append((f, class_dir.name, "scene"))
        for f in closeups[:per_group]:
            samples.append((f, class_dir.name, "closeup"))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out comparison of recognition backends")
    parser.add_argument("--input-dir", default="data/raw/asl_alphabet/asl_alphabet_train")
    parser.add_argument("--per-group", type=int, default=15,
                        help="Unseen images per class per group.")
    parser.add_argument("--cnn-limit", type=int, default=150,
                        help="per-class-limit used when building the crop cache.")
    parser.add_argument("--mlp-limit", type=int, default=100,
                        help="per-class-limit used when training the MLP.")
    parser.add_argument("--hybrid-scene-limit", type=int, default=400,
                        help="--scene-limit used when building the hybrid crop cache.")
    parser.add_argument("--hybrid-closeup-limit", type=int, default=300,
                        help="--closeup-limit used when building the hybrid crop cache.")
    parser.add_argument("--backends", nargs="+", default=None,
                        help="Defaults to every backend with a trained model.")
    parser.add_argument("--report", default="logs/evaluation/backend_comparison.json")
    args = parser.parse_args()

    root = Path(args.input_dir)
    samples = build_holdout(
        root, args.per_group, args.cnn_limit, args.mlp_limit,
        args.hybrid_scene_limit, args.hybrid_closeup_limit,
    )
    print(f"Held-out samples: {len(samples)} "
          f"({sum(1 for s in samples if s[2] == 'scene')} scene / "
          f"{sum(1 for s in samples if s[2] == 'closeup')} closeup)\n")

    # Extract landmarks once; both backends consume the same detections so the
    # comparison is not confounded by differing hand detection.
    print("Extracting landmarks...")
    pre = ASLPreprocessor(model_asset_path=HAND_MODEL)
    prepared = []
    no_hand = 0
    for path, label, group in samples:
        lm = pre.extract_landmarks(str(path))
        if not np.any(lm):
            no_hand += 1
            continue
        prepared.append((path, label, group, lm))
    pre.close()
    print(f"Usable (hand detected): {len(prepared)}  |  skipped: {no_hand}\n")

    results: Dict[str, dict] = {}
    for backend in (args.backends or available_backends()):
        try:
            rec = create_recognizer(backend)
        except Exception as exc:
            print(f"Skipping {backend}: {exc}")
            continue

        correct = defaultdict(int)
        total = defaultdict(int)
        per_class_wrong = defaultdict(int)
        confusions = defaultdict(int)

        for path, label, group, lm in prepared:
            pred = rec.predict_frame(image_path=str(path), landmarks=lm)
            ok = pred["label"] == label
            total[group] += 1
            total["all"] += 1
            if ok:
                correct[group] += 1
                correct["all"] += 1
            else:
                per_class_wrong[label] += 1
                confusions[f"{label}->{pred['label']}"] += 1

        acc = {g: (correct[g] / total[g] if total[g] else 0.0) for g in total}
        results[backend] = {
            "accuracy": acc,
            "counts": dict(total),
            "top_confusions": dict(sorted(confusions.items(), key=lambda kv: -kv[1])[:12]),
            "worst_classes": dict(sorted(per_class_wrong.items(), key=lambda kv: -kv[1])[:8]),
        }

        print(f"--- {backend} ---")
        for group in ("all", "scene", "closeup"):
            if group in acc:
                print(f"  {group:8s} {acc[group]*100:5.1f}%  ({correct[group]}/{total[group]})")
        print(f"  top confusions: {list(results[backend]['top_confusions'].items())[:6]}\n")

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved report to {out}")


if __name__ == "__main__":
    main()
