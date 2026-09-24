"""Adapt the hybrid recogniser to your own camera, lighting and hand.

A model cannot learn a domain its training data does not contain. The corpus was
recorded by other people in other rooms, which is why accuracy measured on the
corpus (92 % on webcam-style frames) does not carry over to your webcam.

This script mixes your captured samples into a short fine-tuning run. Corpus data
is kept in the mix deliberately: training on your few hundred images alone would
overwrite everything the model already knows (catastrophic forgetting). Your
samples are oversampled so they still carry weight despite being far fewer.

Part of your data is held out and never trained on, so the reported
before/after numbers are honest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np  # noqa: E402

from modules.recognition.crop_dataset import (  # noqa: E402
    build_crop_cache,
    make_dataset,
)
from modules.recognition.features import normalize_landmark_features  # noqa: E402
from modules.recognition.hand_crop import DEFAULT_CROP_SIZE  # noqa: E402
from modules.recognition.user_split import (  # noqa: E402
    DEFAULT_VAL_FRACTION,
    split_per_class,
)

HAND_MODEL = "models/recognition/hand_landmarker.task"


def load_cache(cache_dir: Path):
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    landmarks = normalize_landmark_features(
        np.load(cache_dir / "landmarks.npy").astype(np.float32)
    )
    return manifest["items"], landmarks


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the hybrid model on your own samples")
    parser.add_argument("--user-dir", default="data/raw/user_samples/test",
                        help="Directory of your captured samples (one folder per class).")
    parser.add_argument("--user-cache", default="data/processed/user_crops")
    parser.add_argument("--corpus-cache", default="data/processed/hand_crops_v2")
    parser.add_argument("--model-path", default="models/recognition/hybrid_model.keras")
    parser.add_argument("--metadata-path", default="models/recognition/hybrid_metadata.json")
    parser.add_argument("--output-path", default="models/recognition/hybrid_user_model.keras")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
                        help="Fraction of your samples held out for honest evaluation. "
                             "scripts/evaluate_on_user_data.py must use the same value.")
    parser.add_argument("--user-repeat", type=int, default=8,
                        help="How many times to repeat your samples per epoch, so a few "
                             "hundred images are not drowned out by the corpus.")
    parser.add_argument("--corpus-samples", type=int, default=4000,
                        help="Corpus images mixed in to prevent catastrophic forgetting.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--image-size", type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Re-crop your samples even if the cache already exists.")
    args = parser.parse_args()

    user_dir = Path(args.user_dir)
    if not user_dir.exists():
        print(f"No samples at {user_dir}. Record some with scripts/capture_samples.py.")
        sys.exit(1)

    user_cache = Path(args.user_cache)
    if args.rebuild_cache or not (user_cache / "manifest.json").exists():
        print(f"Cropping your samples into {user_cache} ...")
        build_crop_cache(
            raw_dir=str(user_dir),
            cache_dir=str(user_cache),
            per_class_limit=None,
            size=args.image_size,
            model_asset_path=HAND_MODEL,
        )

    from tensorflow import keras

    meta = json.loads(Path(args.metadata_path).read_text(encoding="utf-8"))
    class_names = list(meta["class_names"])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    all_user_items, user_lms = load_cache(user_cache)
    # Keep the manifest row number so landmark rows stay aligned after filtering.
    kept = [i for i, it in enumerate(all_user_items) if it["label"] in class_to_idx]
    user_items = [all_user_items[i] for i in kept]
    if not user_items:
        print("None of your classes match the model's classes.")
        sys.exit(1)

    tr_pos, va_pos = split_per_class(user_items, args.val_fraction)
    print(f"Your samples: {len(tr_pos)} for training, {len(va_pos)} held out")

    def user_arrays(positions):
        rows = [kept[p] for p in positions]
        paths = [str(user_cache / all_user_items[r]["file"]) for r in rows]
        lms = user_lms[rows]
        labels = [class_to_idx[all_user_items[r]["label"]] for r in rows]
        return paths, lms, labels

    u_tr_paths, u_tr_lms, u_tr_labels = user_arrays(tr_pos)
    u_va_paths, u_va_lms, u_va_labels = user_arrays(va_pos)

    corpus_cache = Path(args.corpus_cache)
    c_items, c_lms = load_cache(corpus_cache)
    keep = [i for i, it in enumerate(c_items) if it["label"] in class_to_idx]
    rng = np.random.default_rng(42)
    if len(keep) > args.corpus_samples:
        keep = list(rng.choice(keep, args.corpus_samples, replace=False))
    c_paths = [str(corpus_cache / c_items[i]["file"]) for i in keep]
    c_lm = c_lms[keep]
    c_labels = [class_to_idx[c_items[i]["label"]] for i in keep]
    print(f"Corpus images mixed in: {len(c_paths)}")

    # Oversample your data so it is not swamped by the corpus.
    mix_paths = c_paths + u_tr_paths * args.user_repeat
    mix_lms = np.concatenate([c_lm] + [u_tr_lms] * args.user_repeat, axis=0)
    mix_labels = c_labels + u_tr_labels * args.user_repeat
    print(f"Training set: {len(mix_paths)} images "
          f"({len(u_tr_paths)} of yours x{args.user_repeat})")

    train_ds = make_dataset(mix_paths, mix_lms, mix_labels, args.image_size,
                            args.batch_size, training=True)
    user_val_ds = make_dataset(u_va_paths, u_va_lms, u_va_labels, args.image_size,
                               args.batch_size, training=False)

    model = keras.models.load_model(args.model_path)

    def score(ds, labels):
        probs = model.predict(ds, verbose=0)
        return float(np.mean(np.argmax(probs, axis=1) == np.array(labels)))

    before = score(user_val_ds, u_va_labels)
    print(f"\nBEFORE fine-tuning: {before*100:.1f}% on your held-out samples")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=user_val_ds, epochs=args.epochs)

    after = score(user_val_ds, u_va_labels)
    print(f"\nAFTER fine-tuning : {after*100:.1f}% on your held-out samples")
    print(f"Change            : {(after-before)*100:+.1f} points")

    model.save(args.output_path)
    meta_out = dict(meta)
    meta_out["architecture"] += " (fine-tuned on user samples)"
    meta_out["user_finetune"] = {
        "user_dir": str(user_dir),
        "user_train": len(tr_pos),
        "user_val": len(va_pos),
        "before": before,
        "after": after,
    }
    Path(args.output_path).with_name("hybrid_user_metadata.json").write_text(
        json.dumps(meta_out, indent=2), encoding="utf-8"
    )
    print(f"\nSaved adapted model to {args.output_path}")
    print("Use it with:  --backend hybrid_user")


if __name__ == "__main__":
    main()
