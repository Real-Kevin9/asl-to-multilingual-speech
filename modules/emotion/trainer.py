"""Train the facial emotion CNN on the 4-class FER2013 subset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from modules.emotion.data import (
    DEFAULT_PROCESSED_DIR,
    EMOTIONS,
    available_splits,
    load_split,
)
from modules.emotion.model import ARCHITECTURES, build_emotion_cnn

DEFAULT_OUTPUT_DIR = "models/emotion"
MODEL_FILENAME = "emotion_model.keras"
METADATA_FILENAME = "emotion_metadata.json"


class EmotionTrainer:
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def _resolve_splits(
        self, processed_dir: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        splits = set(available_splits(processed_dir))
        x_train, y_train = load_split(processed_dir, "train")
        if "validation" in splits:
            x_val, y_val = load_split(processed_dir, "validation")
        elif "val" in splits:
            x_val, y_val = load_split(processed_dir, "val")
        elif "test" in splits:
            x_val, y_val = load_split(processed_dir, "test")
        else:
            # Hold out 15% of train if no val split exists.
            n = len(x_train)
            cut = max(1, int(0.85 * n))
            x_val, y_val = x_train[cut:], y_train[cut:]
            x_train, y_train = x_train[:cut], y_train[:cut]
        return x_train, y_train, x_val, y_val

    def train(
        self,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        max_train: Optional[int] = None,
        seed: int = 42,
        arch: str = "v2",
        patience: int = 8,
    ) -> Dict[str, object]:
        from tensorflow import keras

        if arch not in ARCHITECTURES:
            raise ValueError(f"Unknown arch {arch!r}; expected one of {sorted(ARCHITECTURES)}.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        x_train, y_train, x_val, y_val = self._resolve_splits(processed_dir)

        rng = np.random.default_rng(seed)
        if max_train is not None and len(x_train) > max_train:
            idx = rng.choice(len(x_train), size=max_train, replace=False)
            x_train, y_train = x_train[idx], y_train[idx]

        model = ARCHITECTURES[arch](learning_rate=learning_rate)
        # Augmented runs improve in longer, noisier plateaus than v1 did, so the
        # v1 patience of 3 would stop them well before they converge.
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=patience,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(2, patience // 3),
                min_lr=1e-6,
            ),
        ]

        # Happy dominates FER; without weights angry/sad collapse.
        counts = np.bincount(y_train, minlength=len(EMOTIONS)).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        class_weight = {
            i: float(counts.sum() / (len(EMOTIONS) * counts[i]))
            for i in range(len(EMOTIONS))
        }

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        model_path = self.output_dir / MODEL_FILENAME
        model.save(model_path)

        # Evaluate on validation
        proba = model.predict(x_val, verbose=0)
        preds = np.argmax(proba, axis=1)
        accuracy = float(np.mean(preds == y_val)) if len(y_val) else 0.0

        from sklearn.metrics import classification_report, confusion_matrix

        report = classification_report(
            y_val,
            preds,
            labels=list(range(len(EMOTIONS))),
            target_names=EMOTIONS,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_val, preds, labels=list(range(len(EMOTIONS)))).tolist()

        descriptions = {
            "v1": "small ConvNet 48x48 grayscale → 4 emotions",
            "v2": "VGG-style ConvNet 48x48 grayscale, in-model augmentation → 4 emotions",
        }
        meta = {
            "architecture": descriptions[arch],
            "arch": arch,
            "emotions": EMOTIONS,
            "train_samples": int(len(x_train)),
            "val_samples": int(len(y_val)),
            "epochs_requested": epochs,
            "epochs_ran": len(history.history.get("loss", [])),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "val_accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
            "model_path": str(model_path),
            "history": {
                "accuracy": [float(v) for v in history.history.get("accuracy", [])],
                "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
                "loss": [float(v) for v in history.history.get("loss", [])],
                "val_loss": [float(v) for v in history.history.get("val_loss", [])],
            },
        }
        (self.output_dir / METADATA_FILENAME).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        (self.output_dir / "emotion_confusion_matrix.json").write_text(
            json.dumps({"emotions": EMOTIONS, "matrix": cm}, indent=2),
            encoding="utf-8",
        )
        return meta
