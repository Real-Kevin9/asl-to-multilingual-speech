import numpy as np

from modules.recognition.preprocess import ASLPreprocessor


def test_extract_landmarks_returns_vector_for_missing_hand():
    preprocessor = ASLPreprocessor()
    vector = preprocessor.extract_landmarks("does_not_exist.jpg")
    assert vector.shape == (63,)
    assert np.allclose(vector, np.zeros(63, dtype=np.float32))
