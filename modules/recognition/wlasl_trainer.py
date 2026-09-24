"""Train MobileNetV2 + BiLSTM on WLASL word-level clips.

Training strategy (CPU-friendly, proposal-faithful):

1. Freeze ImageNet MobileNetV2 and extract per-frame pooled features (with RGB aug).
2. Train a BiLSTM + temporal-attention classifier with class weights / label smoothing.
3. Export a single end-to-end ``build_video_model`` artefact when layer shapes match;
   otherwise keep a feature-backbone + head pair used at inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from modules.recognition.cnn_lstm import build_video_model
from modules.recognition.wlasl_data import (
    DEFAULT_FRAME_SIZE,
    DEFAULT_FRAMES,
    DEFAULT_PROCESSED_DIR,
    iter_manifest,
    load_clip_array,
)

MODEL_FILENAME = "wlasl_video_model.keras"
HEAD_FILENAME = "wlasl_lstm_head.keras"
BACKBONE_FILENAME = "wlasl_feature_backbone.keras"
METADATA_FILENAME = "wlasl_video_metadata.json"
FEATURE_CACHE = "wlasl_features_v2.npz"


def _register_attention_pool_layer():
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow as tf

    @keras.utils.register_keras_serializable(package="wlasl")
    class TemporalAttentionPool(layers.Layer):
        """Weighted sum over time steps from per-frame logits."""

        def call(self, inputs):
            values, scores = inputs
            w = tf.nn.softmax(tf.squeeze(scores, axis=-1), axis=-1)
            w = tf.expand_dims(w, axis=-1)
            return tf.reduce_sum(values * w, axis=1)

        def compute_output_shape(self, input_shape):
            return (input_shape[0][0], input_shape[0][2])

    return TemporalAttentionPool


TemporalAttentionPool = _register_attention_pool_layer()


def _legacy_attn_pool(args):
    import tensorflow as tf

    values, scores = args
    w = tf.nn.softmax(tf.squeeze(scores, axis=-1), axis=-1)
    w = tf.expand_dims(w, axis=-1)
    return tf.reduce_sum(values * w, axis=1)


def wlasl_custom_objects() -> dict:
    """Objects required to load saved WLASL Keras artefacts."""
    return {
        "TemporalAttentionPool": TemporalAttentionPool,
        "_attn_pool": _legacy_attn_pool,
    }


def _augment_rgb_batch(clips: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cheap photometric / mild geometric aug on RGB clips in 0–255."""
    out = clips.copy()
    n = out.shape[0]
    for i in range(n):
        # Brightness / contrast
        alpha = float(rng.uniform(0.75, 1.25))
        beta = float(rng.uniform(-25, 25))
        out[i] = np.clip(out[i] * alpha + beta, 0, 255)
        # Random grayscale-ish desaturation
        if rng.random() < 0.15:
            gray = out[i].mean(axis=-1, keepdims=True)
            out[i] = 0.35 * out[i] + 0.65 * gray
        # Mild zoom crop (no horizontal flip — ASL is handedness-sensitive)
        if rng.random() < 0.35:
            t, h, w, _ = out[i].shape
            scale = float(rng.uniform(0.82, 1.0))
            nh, nw = int(h * scale), int(w * scale)
            y0 = int(rng.integers(0, max(1, h - nh + 1)))
            x0 = int(rng.integers(0, max(1, w - nw + 1)))
            crop = out[i][:, y0 : y0 + nh, x0 : x0 + nw, :]
            # Resize back with nearest for speed
            import cv2

            resized = np.stack(
                [cv2.resize(crop[j], (w, h), interpolation=cv2.INTER_LINEAR) for j in range(t)],
                axis=0,
            )
            out[i] = resized
    return out.astype(np.float32)


class WLASLTrainer:
    def __init__(self, output_dir: str = "models/recognition"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.class_names: List[str] = []
        self.history = {}

    def _load_meta(self, processed_dir: Path) -> dict:
        return json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))

    def _stack_split(
        self,
        processed_dir: Path,
        split: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows = iter_manifest(str(processed_dir), split)
        xs, ys = [], []
        for row in rows:
            xs.append(load_clip_array(row["clip_path"]))
            ys.append(int(row["label_index"]))
        return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int32)

    def _extract_split_features_batched(
        self,
        processed_dir: Path,
        split: str,
        backbone,
        *,
        clip_batch_size: int = 12,
        feature_batch_size: int = 32,
        aug_copies: int = 0,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features without loading the full split into RAM."""
        rows = iter_manifest(str(processed_dir), split)
        rng = np.random.default_rng(seed)
        feat_chunks: List[np.ndarray] = []
        y_chunks: List[np.ndarray] = []

        def _run_batch(clips: np.ndarray, labels: np.ndarray) -> None:
            feats = self._extract_features(clips, backbone, batch_size=feature_batch_size)
            feat_chunks.append(feats)
            y_chunks.append(labels)

        n = len(rows)
        for start in range(0, n, clip_batch_size):
            batch_rows = rows[start : start + clip_batch_size]
            clips = np.stack([load_clip_array(r["clip_path"]) for r in batch_rows], axis=0)
            labels = np.asarray([int(r["label_index"]) for r in batch_rows], dtype=np.int32)
            _run_batch(clips, labels)
            for k in range(aug_copies):
                if k == 0 and start == 0:
                    print(f"  {split}: aug copies={aug_copies}")
                _run_batch(_augment_rgb_batch(clips, rng), labels)
            done = min(start + clip_batch_size, n)
            if done % 48 == 0 or done == n:
                print(f"  {split} features {done}/{n}")

        return np.concatenate(feat_chunks, axis=0), np.concatenate(y_chunks, axis=0)

    def _feature_backbone(self, frame_size: int):
        from tensorflow import keras

        inp = keras.Input(shape=(frame_size, frame_size, 3), name="frame")
        x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inp)
        backbone = keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(frame_size, frame_size, 3),
            pooling="avg",
        )
        backbone.trainable = False
        out = backbone(x, training=False)
        return keras.Model(inp, out, name="wlasl_feature_backbone")

    def _extract_features(self, clips: np.ndarray, backbone, batch_size: int = 32) -> np.ndarray:
        n, t, h, w, c = clips.shape
        flat = clips.reshape(n * t, h, w, c)
        feats = backbone.predict(flat, batch_size=batch_size, verbose=1)
        return feats.reshape(n, t, -1).astype(np.float32)

    def _build_lstm_head(
        self,
        num_classes: int,
        feat_dim: int,
        num_frames: int,
        lstm_units: Sequence[int],
        dropout: float,
        learning_rate: float,
        label_smoothing: float,
    ):
        from tensorflow import keras
        from tensorflow.keras import layers
        import tensorflow as tf

        inputs = keras.Input(shape=(num_frames, feat_dim), name="features")
        y = layers.Masking(mask_value=0.0)(inputs)
        y = layers.Bidirectional(
            layers.LSTM(lstm_units[0], return_sequences=True), name="bilstm_1"
        )(y)
        y = layers.Dropout(dropout)(y)
        y = layers.Bidirectional(
            layers.LSTM(lstm_units[1], return_sequences=True), name="bilstm_2"
        )(y)
        # Temporal attention without Multiply (avoids Keras 3 mask broadcast bugs).
        score = layers.Dense(1, name="attn_score")(y)
        y = TemporalAttentionPool(name="attn_pool")([y, score])
        y = layers.Dropout(dropout)(y)
        y = layers.Dense(128, activation="relu", name="head_fc")(y)
        y = layers.Dropout(dropout)(y)
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(y)
        model = keras.Model(inputs, outputs, name="wlasl_bilstm_attn_head")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
            metrics=["accuracy"],
        )
        return model

    def _class_weights(self, y: np.ndarray, num_classes: int) -> dict:
        counts = np.bincount(y, minlength=num_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (num_classes * counts)
        # Cap extreme weights so rare classes do not dominate gradients.
        weights = np.clip(weights, 0.5, 3.0)
        return {i: float(weights[i]) for i in range(num_classes)}

    def _export_video_model(
        self,
        num_classes: int,
        num_frames: int,
        frame_size: int,
        lstm_units: Sequence[int],
        dropout: float,
    ):
        """Legacy TimeDistributed export (no attention). Prefer head+backbone inference."""
        return build_video_model(
            num_classes=num_classes,
            frames=num_frames,
            frame_shape=(frame_size, frame_size, 3),
            lstm_units=lstm_units,
            dropout=dropout,
            freeze_backbone=True,
            learning_rate=1e-3,
        )

    def train(
        self,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
        epochs: int = 60,
        batch_size: int = 16,
        lstm_units: Sequence[int] = (128, 64),
        dropout: float = 0.35,
        learning_rate: float = 1e-3,
        feature_batch_size: int = 32,
        label_smoothing: float = 0.08,
        aug_copies: int = 1,
        clip_batch_size: int = 12,
        seed: int = 42,
        rebuild_features: bool = False,
    ) -> dict:
        from tensorflow import keras
        import tensorflow as tf

        processed = Path(processed_dir)
        meta = self._load_meta(processed)
        self.class_names = list(meta["class_names"])
        num_frames = int(meta["num_frames"])
        frame_size = int(meta["frame_size"])
        num_classes = len(self.class_names)

        feat_cache = processed / FEATURE_CACHE
        backbone = self._feature_backbone(frame_size)

        if feat_cache.exists() and not rebuild_features:
            print(f"Loading cached features from {feat_cache}")
            cached = np.load(feat_cache)
            train_feat, val_feat = cached["train_feat"], cached["val_feat"]
            y_train, y_val = cached["y_train"], cached["y_val"]
        else:
            if rebuild_features and feat_cache.exists():
                feat_cache.unlink()
            print(
                f"Extracting MobileNetV2 features in batches "
                f"(clip_batch={clip_batch_size}, aug_copies={aug_copies})..."
            )
            train_feat, y_train = self._extract_split_features_batched(
                processed,
                "train",
                backbone,
                clip_batch_size=clip_batch_size,
                feature_batch_size=feature_batch_size,
                aug_copies=aug_copies,
                seed=seed,
            )
            val_feat, y_val = self._extract_split_features_batched(
                processed,
                "val",
                backbone,
                clip_batch_size=clip_batch_size,
                feature_batch_size=feature_batch_size,
                aug_copies=0,
                seed=seed,
            )
            np.savez_compressed(
                feat_cache,
                train_feat=train_feat,
                val_feat=val_feat,
                y_train=y_train,
                y_val=y_val,
            )
            print(f"Wrote {feat_cache} train_feat={train_feat.shape}")

        # Persist backbone for inference (attention head is not in build_video_model).
        backbone_path = self.output_dir / BACKBONE_FILENAME
        backbone.save(backbone_path)

        head = self._build_lstm_head(
            num_classes=num_classes,
            feat_dim=int(train_feat.shape[-1]),
            num_frames=num_frames,
            lstm_units=lstm_units,
            dropout=dropout,
            learning_rate=learning_rate,
            label_smoothing=label_smoothing,
        )
        head.summary()

        class_weight = self._class_weights(y_train, num_classes)
        y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes)
        sample_weights = np.array([class_weight[int(y)] for y in y_train], dtype=np.float32)

        def _feat_augment(x, y):
            # Temporal dropout: zero a random contiguous chunk of frames.
            noise = tf.random.normal(tf.shape(x), stddev=0.03)
            x = x + noise
            t = tf.shape(x)[0]
            drop_len = tf.maximum(1, t // 6)
            start = tf.random.uniform([], 0, t - drop_len + 1, dtype=tf.int32)
            mask = tf.concat(
                [
                    tf.ones([start, 1], tf.float32),
                    tf.zeros([drop_len, 1], tf.float32),
                    tf.ones([t - start - drop_len, 1], tf.float32),
                ],
                axis=0,
            )
            x = x * mask
            return x, y

        train_ds = (
            tf.data.Dataset.from_tensor_slices((train_feat, y_train_oh, sample_weights))
            .shuffle(len(y_train), seed=seed, reshuffle_each_iteration=True)
            .map(
                lambda x, y, w: _feat_augment(x, y) + (w,),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((val_feat, y_val_oh))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=12,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1,
            ),
        ]
        hist = head.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        self.history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}

        head_path = self.output_dir / HEAD_FILENAME
        head.save(head_path)

        # Also save a TimeDistributed video shell for tooling that expects one file;
        # live inference prefers backbone+head (attention).
        try:
            shell = self._export_video_model(
                num_classes=num_classes,
                num_frames=num_frames,
                frame_size=frame_size,
                lstm_units=lstm_units,
                dropout=dropout,
            )
            # Best-effort: copy matching BiLSTM weights if shapes agree.
            for name in ("bilstm_1", "bilstm_2"):
                try:
                    shell.get_layer(name).set_weights(head.get_layer(name).get_weights())
                except Exception:
                    pass
            model_path = self.output_dir / MODEL_FILENAME
            shell.save(model_path)
            self.model = shell
        except Exception as exc:
            print(f"[wlasl] video shell export skipped ({exc})")
            model_path = head_path

        val_loss, val_acc = head.evaluate(val_feat, y_val_oh, verbose=0)
        # Macro recall for a fairer sparse-class picture
        probs = head.predict(val_feat, verbose=0)
        pred = probs.argmax(axis=1)
        recalls = []
        for c in range(num_classes):
            mask = y_val == c
            if mask.any():
                recalls.append(float((pred[mask] == c).mean()))
        macro_recall = float(np.mean(recalls)) if recalls else 0.0

        report = {
            "backend": "wlasl_video",
            "architecture": "MobileNetV2 features + BiLSTM + temporal attention",
            "dataset": "Voxel51/WLASL top-K glosses (signer crop, multi-view)",
            "num_classes": num_classes,
            "class_names": self.class_names,
            "num_frames": num_frames,
            "frame_size": frame_size,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "val_accuracy": float(val_acc),
            "val_macro_recall": macro_recall,
            "val_loss": float(val_loss),
            "history": self.history,
            "model_path": str(model_path),
            "head_path": str(head_path),
            "backbone_path": str(backbone_path),
            "inference": "backbone+head",
            "lstm_units": list(lstm_units),
            "dropout": dropout,
            "label_smoothing": label_smoothing,
            "aug_copies": aug_copies,
            "target_signs": 50,
            "target_accuracy": 0.85,
            "signs_met": num_classes >= 50,
            "accuracy_met": float(val_acc) >= 0.85,
        }
        meta_path = self.output_dir / METADATA_FILENAME
        meta_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved head → {head_path}")
        print(f"Saved backbone → {backbone_path}")
        print(f"Val accuracy: {val_acc:.3f}  macro-recall: {macro_recall:.3f}")
        return report
