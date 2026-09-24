"""Sign-recognition package.

``KERAS_HOME`` is configured here, at package-import time, and this must happen
before TensorFlow/Keras is imported anywhere in the process. That ordering is
not obvious: importing ``mediapipe`` pulls in TensorFlow transitively, so any
module that touches MediaPipe would otherwise lock Keras to the default
``~/.keras`` cache (which is not always writable) before we get a chance to
redirect it.
"""

from modules.recognition.keras_env import configure_keras_home

configure_keras_home()
