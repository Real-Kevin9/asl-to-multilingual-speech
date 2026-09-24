"""CNNs for 48×48 grayscale facial emotion classification.

``build_emotion_cnn`` is the original three-block network kept so the Update 6
result stays reproducible. ``build_emotion_cnn_v2`` is the deeper VGG-style
network with built-in augmentation used from Update 11 onward.

Augmentation lives inside the model rather than in the input pipeline so it is
saved with the artefact and is automatically inert at inference — ``EmotionDetector``
needs no changes and cannot accidentally augment a live frame.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from modules.emotion.data import EMOTIONS, IMAGE_SIZE


def build_emotion_cnn(
    num_classes: int = len(EMOTIONS),
    input_shape: Tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, 1),
    dropout: float = 0.35,
    learning_rate: float = 1e-3,
):
    """Build and compile the emotion CNN used by ``EmotionDetector``."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape, name="face")
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion")(x)

    model = keras.Model(inputs, outputs, name="emotion_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_emotion_cnn_v2(
    num_classes: int = len(EMOTIONS),
    input_shape: Tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, 1),
    learning_rate: float = 1e-3,
    width: int = 32,
):
    """Deeper VGG-style network with augmentation, sized for CPU training.

    Three changes over v1, each aimed at a specific weakness of the first run:

    * **Augmentation.** v1 trained on 18k unaugmented 48×48 faces and overfitted
      early. Flips, small rotations, zooms, shifts and contrast jitter are the
      standard remedy for FER and cost nothing at inference.
    * **Paired convolutions and global pooling.** v1 used one convolution per
      block and a ``Flatten`` into a dense layer, putting most of its parameters
      in a single fully-connected layer. Doubling the convolutions and pooling
      globally spends that budget on features instead.
    * **Width kept modest.** Training is CPU-only, so the widest block is 8×
      ``width`` rather than the 512 filters a GPU run would use.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape, name="face")

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.10)(x)
    x = layers.RandomZoom(0.10)(x)
    x = layers.RandomTranslation(0.10, 0.10)(x)
    x = layers.RandomContrast(0.10)(x)

    for block, (filters, dropout) in enumerate(
        [(width, 0.25), (width * 2, 0.30), (width * 4, 0.35), (width * 8, 0.40)]
    ):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        # The final block is already 6×6; pooling it again loses too much.
        if block < 3:
            x = layers.MaxPooling2D()(x)
        x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion")(x)

    model = keras.Model(inputs, outputs, name="emotion_cnn_v2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


ARCHITECTURES = {
    "v1": build_emotion_cnn,
    "v2": build_emotion_cnn_v2,
}
