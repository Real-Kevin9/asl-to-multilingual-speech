"""Training loop for the hybrid MobileNetV2 + BiLSTM + geometry recogniser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from modules.recognition.cnn_lstm import DEFAULT_LSTM_UNITS, build_hybrid_model
from modules.recognition.crop_dataset import load_hybrid_datasets
from modules.recognition.hand_crop import DEFAULT_CROP_SIZE

MODEL_FILENAME = "hybrid_model.keras"
METADATA_FILENAME = "hybrid_metadata.json"


class HybridTrainer:
    """Trains the two-stream recogniser on cached crops plus landmarks."""

    def __init__(self, output_dir: str = "models/recognition"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.class_names: List[str] = []
        self.history: dict = {}
        self.stats: dict = {}

    def train(
        self,
        cache_dir: str,
        image_size: int = DEFAULT_CROP_SIZE,
        batch_size: int = 32,
        epochs: int = 12,
        validation_split: float = 0.2,
        lstm_units=DEFAULT_LSTM_UNITS,
        dropout: float = 0.4,
        learning_rate: float = 1e-3,
        fine_tune_epochs: int = 0,
        fine_tune_at: int = 100,
        fine_tune_lr: float = 1e-5,
    ) -> dict:
        from tensorflow import keras

        train_ds, val_ds, class_names, stats = load_hybrid_datasets(
            cache_dir,
            image_size=image_size,
            batch_size=batch_size,
            validation_split=validation_split,
        )
        self.class_names = class_names
        self.stats = stats
        print(f"Classes ({len(class_names)}): {class_names}")
        print(f"Split: {stats['train']} train / {stats['val']} val")
        print(f"  train by group: {stats['train_by_group']}")
        print(f"  val   by group: {stats['val_by_group']}")

        self.model = build_hybrid_model(
            num_classes=len(class_names),
            input_shape=(image_size, image_size, 3),
            lstm_units=lstm_units,
            dropout=dropout,
            freeze_backbone=True,
            augment=True,
            learning_rate=learning_rate,
        )

        model_path = self.output_dir / MODEL_FILENAME
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(model_path), monitor="val_accuracy", save_best_only=True, verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
            ),
        ]

        print("\n=== Stage 1: frozen backbone ===")
        hist = self.model.fit(
            train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks
        )
        self.history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}

        if fine_tune_epochs > 0:
            print("\n=== Stage 2: fine-tuning MobileNetV2 top layers ===")
            backbone = next(
                l for l in self.model.layers if l.name.startswith("mobilenetv2")
            )
            backbone.trainable = True
            for layer in backbone.layers[:fine_tune_at]:
                layer.trainable = False
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            hist_ft = self.model.fit(
                train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=callbacks
            )
            for key, vals in hist_ft.history.items():
                self.history.setdefault(key, []).extend(float(v) for v in vals)

        metrics = self.evaluate(val_ds)
        self.save(image_size=image_size, metrics=metrics)
        return metrics

    def evaluate(self, val_ds) -> dict:
        from sklearn.metrics import classification_report, confusion_matrix

        y_true: List[int] = []
        y_pred: List[int] = []
        for inputs, labels in val_ds:
            probs = self.model.predict(inputs, verbose=0)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(np.argmax(probs, axis=1).tolist())

        accuracy = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
        print(f"\nHybrid validation accuracy: {accuracy:.4f}")

        label_ids = list(range(len(self.class_names)))
        print(classification_report(
            y_true, y_pred, labels=label_ids, target_names=self.class_names, zero_division=0
        ))
        report = classification_report(
            y_true, y_pred, labels=label_ids, target_names=self.class_names,
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=label_ids).tolist()
        (self.output_dir / "hybrid_confusion_matrix.json").write_text(
            json.dumps({"labels": self.class_names, "matrix": cm}, indent=2), encoding="utf-8"
        )

        return {
            "val_accuracy": accuracy,
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "per_class_f1": {c: float(report[c]["f1-score"]) for c in self.class_names},
        }

    def save(self, image_size: int, metrics: Optional[dict] = None) -> None:
        model_path = self.output_dir / MODEL_FILENAME
        self.model.save(model_path)

        metadata = {
            "architecture": "MobileNetV2 + 2-layer BiLSTM + landmark geometry stream",
            "input_size": image_size,
            "landmark_dim": 63,
            "preprocessing": (
                "MediaPipe hand bbox crop -> square resize -> rescale [-1,1]; "
                "landmarks wrist-centred and scale-normalized"
            ),
            "class_names": self.class_names,
            "split": self.stats,
            "metrics": metrics or {},
            "history": self.history,
        }
        (self.output_dir / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"Saved hybrid model to {model_path}")
