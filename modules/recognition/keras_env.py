"""Keras environment configuration.

Keras caches downloaded weights (e.g. MobileNetV2 ImageNet weights) in
``$KERAS_HOME``, defaulting to ``~/.keras``. That location is not always
writable, and keeping the weights inside the project makes runs reproducible and
self-contained. ``KERAS_HOME`` is read when ``keras`` is first imported, so this
must run *before* any TensorFlow/Keras import.
"""

from __future__ import annotations

import os
from pathlib import Path

# modules/recognition/keras_env.py -> repo root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KERAS_HOME = _REPO_ROOT / "models" / ".keras"


def configure_keras_home(path: str | os.PathLike | None = None) -> Path:
    """Point ``KERAS_HOME`` at a writable project-local directory.

    Respects an existing ``KERAS_HOME`` if the user already set one.
    """
    if "KERAS_HOME" in os.environ:
        return Path(os.environ["KERAS_HOME"])

    keras_home = Path(path) if path is not None else DEFAULT_KERAS_HOME
    keras_home.mkdir(parents=True, exist_ok=True)
    os.environ["KERAS_HOME"] = str(keras_home)
    return keras_home
