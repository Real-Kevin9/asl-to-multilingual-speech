from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from modules.recognition.dataset import load_preprocessed_data
from modules.recognition.features import normalize_landmark_features


class RecognitionTrainer:
    def __init__(self, output_dir: str = "models/recognition"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model: MLPClassifier | None = None
        self.scaler: StandardScaler | None = None

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        # per-sample normalization (centering on wrist + scale)
        features = normalize_landmark_features(features)

        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # feature scaling across dataset
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        self.scaler = scaler

        # small MLP as a stronger baseline than logistic regression
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=200,
            early_stopping=True,
            random_state=42,
        )
        model.fit(x_train, y_train)
        self.model = model

        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Recognition model test accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        self.save_model()

    def save_model(self) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        model_path = self.output_dir / "recognition_model.joblib"
        import joblib

        joblib.dump(self.model, model_path)
        # Save label encoder if present
        try:
            encoder = getattr(self, "label_encoder", None)
            if encoder is not None:
                joblib.dump(encoder, self.output_dir / "label_encoder.joblib")
        except Exception:
            pass

        # Save scaler if present
        try:
            if getattr(self, "scaler", None) is not None:
                joblib.dump(self.scaler, self.output_dir / "scaler.joblib")
        except Exception:
            pass

        print(f"Saved recognition model to {model_path}")


def train_recognition_model(
    features_path: str,
    labels_path: str,
    output_dir: str = "models/recognition",
) -> None:
    features, labels, encoder = load_preprocessed_data(features_path, labels_path)
    trainer = RecognitionTrainer(output_dir=output_dir)
    trainer.label_encoder = encoder
    trainer.train(features, labels)
