"""Training loop for the MobileNetV2 + BiLSTM recogniser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from modules.recognition.cnn_lstm import DEFAULT_LSTM_UNITS, build_static_model
from modules.recognition.crop_dataset import load_crop_datasets
from modules.recognition.hand_crop import DEFAULT_CROP_SIZE

MODEL_FILENAME = "cnn_lstm_model.keras"
METADATA_FILENAME = "cnn_lstm_metadata.json"


class CNNLSTMTrainer:
    """Trains the CNN+LSTM classifier on cached hand crops.

    Training has two optional stages:

    1. **Head training** with the ImageNet backbone frozen. The BiLSTM and the
       classifier learn on top of stable features. This is fast and is the main
       stage on CPU-only machines.
    2. **Fine-tuning** where the top of MobileNetV2 is unfrozen and trained with
       a much smaller learning rate. Optional, and slower.
    """

    def __init__(self, output_dir: str = "models/recognition"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.class_names: List[str] = []
        self.history = {}

    def train(
        self,
        cache_dir: str,
        image_size: int = DEFAULT_CROP_SIZE,
        batch_size: int = 32,
        epochs: int = 12,
        validation_split: float = 0.2,
        lstm_units=DEFAULT_LSTM_UNITS,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        fine_tune_epochs: int = 0,
        fine_tune_at: int = 100,
        fine_tune_lr: float = 1e-5,
    ) -> dict:
        from tensorflow import keras

        train_ds, val_ds, class_names = load_crop_datasets(
            cache_dir,
            image_size=image_size,
            batch_size=batch_size,
            validation_split=validation_split,
        )
        self.class_names = class_names
        print(f"Classes ({len(class_names)}): {class_names}")

        self.model = build_static_model(
            num_classes=len(class_names),
            input_shape=(image_size, image_size, 3),
            lstm_units=lstm_units,
            dropout=dropout,
            freeze_backbone=True,
            augment=True,
            learning_rate=learning_rate,
        )
        self.model.summary()

        model_path = self.output_dir / MODEL_FILENAME
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(model_path), monitor="val_accuracy", save_best_only=True, verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
            ),
        ]

        print("\n=== Stage 1: training BiLSTM head (backbone frozen) ===")
        hist = self.model.fit(
            train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks
        )
        self.history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}

        if fine_tune_epochs > 0:
            print("\n=== Stage 2: fine-tuning MobileNetV2 top layers ===")
            backbone = next(
                layer for layer in self.model.layers if layer.name.startswith("mobilenetv2")
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
        """Evaluate on the validation split and print a per-class report."""
        from sklearn.metrics import classification_report, confusion_matrix

        y_true: List[int] = []
        y_pred: List[int] = []
        for batch_x, batch_y in val_ds:
            probs = self.model.predict(batch_x, verbose=0)
            y_true.extend(batch_y.numpy().tolist())
            y_pred.extend(np.argmax(probs, axis=1).tolist())

        accuracy = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
        print(f"\nCNN+LSTM validation accuracy: {accuracy:.4f}")

        # Pass explicit labels so a class absent from the validation split does
        # not shrink the report or make the confusion matrix non-square.
        label_ids = list(range(len(self.class_names)))
        print(classification_report(
            y_true, y_pred, labels=label_ids,
            target_names=self.class_names, zero_division=0,
        ))

        report = classification_report(
            y_true, y_pred, labels=label_ids, target_names=self.class_names,
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=label_ids).tolist()

        classes_in_val = len(set(y_true))
        if classes_in_val < len(self.class_names):
            print(
                f"WARNING: validation split covers only {classes_in_val}/"
                f"{len(self.class_names)} classes."
            )

        cm_path = self.output_dir / "cnn_lstm_confusion_matrix.json"
        cm_path.write_text(
            json.dumps({"labels": self.class_names, "matrix": cm}, indent=2), encoding="utf-8"
        )
        print(f"Saved confusion matrix to {cm_path}")

        return {
            "val_accuracy": accuracy,
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "per_class_f1": {c: float(report[c]["f1-score"]) for c in self.class_names},
        }

    def save(self, image_size: int, metrics: Optional[dict] = None) -> None:
        model_path = self.output_dir / MODEL_FILENAME
        # ModelCheckpoint already wrote the best weights; save again so the file
        # exists even if training ran without improving.
        self.model.save(model_path)

        metadata = {
            "architecture": "MobileNetV2 + 2-layer BiLSTM",
            "input_size": image_size,
            "preprocessing": "MediaPipe hand bbox crop -> square resize -> rescale [-1,1]",
            "class_names": self.class_names,
            "metrics": metrics or {},
            "history": self.history,
        }
        meta_path = self.output_dir / METADATA_FILENAME
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(f"Saved CNN+LSTM model to {model_path}")
        print(f"Saved metadata to {meta_path}")


def train_cnn_lstm_model(cache_dir: str, output_dir: str = "models/recognition", **kwargs) -> dict:
    trainer = CNNLSTMTrainer(output_dir=output_dir)
    return trainer.train(cache_dir, **kwargs)
