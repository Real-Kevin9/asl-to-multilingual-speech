"""MobileNetV2 + BiLSTM sign-recognition architectures.

This module implements the recogniser described in the interim report
(Section 4.2, Phase 2): a MobileNetV2 CNN produces spatial feature vectors
which are consumed in order by a **two-layer bidirectional LSTM**.

Two builders are provided because the project has two recognition sub-tasks:

``build_static_model``
    For single-image (static alphabet) recognition. MobileNetV2 turns a hand
    crop into an ``H x W x C`` feature map, which is flattened into an
    ``H*W``-step sequence of ``C``-dimensional vectors and read by the BiLSTM.
    This keeps the exact CNN->BiLSTM structure of the proposal while remaining
    trainable on the static ASL-alphabet dataset, which has no time axis.

``build_video_model``
    For dynamic signs (WLASL, future work). MobileNetV2 is applied per frame
    via ``TimeDistributed`` and the BiLSTM runs over the real temporal
    sequence, which is the literal architecture from the report.

Note on augmentation: horizontal flipping is deliberately **not** used, because
ASL handshapes are handedness-sensitive and mirroring can change or invalidate
a sign.
"""

from __future__ import annotations

from typing import Sequence, Tuple

DEFAULT_INPUT_SHAPE: Tuple[int, int, int] = (160, 160, 3)
DEFAULT_LSTM_UNITS: Tuple[int, int] = (128, 64)


def _augmentation_layers(layers):
    """Light geometric/photometric augmentation applied to 0-255 float images."""
    return [
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.12),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomBrightness(0.15, value_range=(0.0, 255.0)),
        layers.RandomContrast(0.15),
    ]


def build_static_model(
    num_classes: int,
    input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    lstm_units: Sequence[int] = DEFAULT_LSTM_UNITS,
    dropout: float = 0.3,
    freeze_backbone: bool = True,
    augment: bool = True,
    learning_rate: float = 1e-3,
):
    """Build the MobileNetV2 -> spatial-sequence -> BiLSTM classifier.

    Args:
        num_classes: Number of sign classes.
        input_shape: Hand-crop input shape; must be a MobileNetV2-supported size.
        lstm_units: Units for the two bidirectional LSTM layers.
        dropout: Dropout applied after each recurrent block.
        freeze_backbone: Keep ImageNet weights fixed (fast, avoids overfitting on
            a small dataset). Set False for the fine-tuning stage.
        augment: Include augmentation layers (inactive at inference time).
        learning_rate: Adam learning rate.

    Returns:
        A compiled ``keras.Model``.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape, name="hand_crop")

    x = inputs
    if augment:
        for layer in _augmentation_layers(layers):
            x = layer(x)

    # MobileNetV2 expects inputs scaled to [-1, 1].
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(x)

    backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = not freeze_backbone
    # Keep BatchNorm in inference mode while the backbone is frozen.
    features = backbone(x, training=not freeze_backbone)

    _, feat_h, feat_w, feat_c = features.shape
    # Flatten the spatial grid into an ordered sequence of feature vectors:
    # (H, W, C) -> (H*W, C). Each "timestep" is one spatial cell of the map.
    sequence = layers.Reshape((feat_h * feat_w, feat_c), name="spatial_sequence")(features)

    y = layers.Bidirectional(
        layers.LSTM(lstm_units[0], return_sequences=True), name="bilstm_1"
    )(sequence)
    y = layers.Dropout(dropout)(y)
    y = layers.Bidirectional(layers.LSTM(lstm_units[1]), name="bilstm_2")(y)
    y = layers.Dropout(dropout)(y)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(y)

    model = keras.Model(inputs, outputs, name="mobilenetv2_bilstm_static")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_hybrid_model(
    num_classes: int,
    input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    landmark_dim: int = 63,
    lstm_units: Sequence[int] = DEFAULT_LSTM_UNITS,
    dropout: float = 0.4,
    freeze_backbone: bool = True,
    augment: bool = True,
    learning_rate: float = 1e-3,
):
    """MobileNetV2 + BiLSTM fused with a hand-geometry stream.

    Motivation (measured, not assumed). On held-out images the two
    representations fail in opposite directions:

    ==============  ==================  ====================
    Backend         webcam-style frames  close-up frames
    ==============  ==================  ====================
    CNN pixels      46 %                97 %
    Landmarks       79 %                55 %
    ==============  ==================  ====================

    Pixels carry fine detail (finger overlap, thumb position) but are sensitive
    to resolution, lighting and skin tone. Landmark geometry is invariant to all
    of those but discards appearance. Neither alone is good enough, so this model
    keeps the proposal's MobileNetV2 -> BiLSTM path and concatenates the
    normalized landmark vector before the classifier, letting the network lean on
    whichever stream is reliable for a given frame.

    Args:
        landmark_dim: Size of the (already wrist-centred, scale-normalized)
            landmark vector.

    Returns:
        A compiled ``keras.Model`` taking ``[hand_crop, landmarks]``.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    image_in = keras.Input(shape=input_shape, name="hand_crop")
    landmark_in = keras.Input(shape=(landmark_dim,), name="landmarks")

    x = image_in
    if augment:
        for layer in _augmentation_layers(layers):
            x = layer(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(x)

    backbone = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    backbone.trainable = not freeze_backbone
    features = backbone(x, training=not freeze_backbone)

    _, feat_h, feat_w, feat_c = features.shape
    sequence = layers.Reshape((feat_h * feat_w, feat_c), name="spatial_sequence")(features)

    v = layers.Bidirectional(
        layers.LSTM(lstm_units[0], return_sequences=True), name="bilstm_1"
    )(sequence)
    v = layers.Dropout(dropout)(v)
    v = layers.Bidirectional(layers.LSTM(lstm_units[1]), name="bilstm_2")(v)
    v = layers.Dropout(dropout)(v)

    g = layers.Dense(128, activation="relu", name="geometry_1")(landmark_in)
    g = layers.BatchNormalization()(g)
    g = layers.Dropout(dropout)(g)
    g = layers.Dense(64, activation="relu", name="geometry_2")(g)

    fused = layers.Concatenate(name="fusion")([v, g])
    fused = layers.Dropout(dropout)(fused)
    fused = layers.Dense(128, activation="relu", name="head")(fused)
    fused = layers.Dropout(dropout)(fused)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(fused)

    model = keras.Model([image_in, landmark_in], outputs, name="mobilenetv2_bilstm_hybrid")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_video_model(
    num_classes: int,
    frames: int = 16,
    frame_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    lstm_units: Sequence[int] = DEFAULT_LSTM_UNITS,
    dropout: float = 0.3,
    freeze_backbone: bool = True,
    learning_rate: float = 1e-3,
):
    """Build the temporal MobileNetV2 + BiLSTM model for dynamic signs.

    Applies MobileNetV2 to every frame (shared weights) and feeds the resulting
    per-frame feature vectors to the BiLSTM in time order. Provided so the
    WLASL/dynamic-sign extension can be trained without redesigning anything.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(frames,) + frame_shape, name="frames")

    x = layers.TimeDistributed(
        layers.Rescaling(1.0 / 127.5, offset=-1.0), name="rescale"
    )(inputs)

    backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=frame_shape,
        pooling="avg",
    )
    backbone.trainable = not freeze_backbone
    x = layers.TimeDistributed(backbone, name="cnn_per_frame")(x)

    y = layers.Bidirectional(
        layers.LSTM(lstm_units[0], return_sequences=True), name="bilstm_1"
    )(x)
    y = layers.Dropout(dropout)(y)
    y = layers.Bidirectional(layers.LSTM(lstm_units[1]), name="bilstm_2")(y)
    y = layers.Dropout(dropout)(y)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(y)

    model = keras.Model(inputs, outputs, name="mobilenetv2_bilstm_video")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
