import numpy as np

from modules.recognition.hand_crop import crop_hand, landmark_bbox


def _fake_landmarks(x_min=0.4, x_max=0.6, y_min=0.3, y_max=0.5):
    """Build a 63-vector whose 21 points span the given normalized box."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[:, 0] = np.linspace(x_min, x_max, 21)
    pts[:, 1] = np.linspace(y_min, y_max, 21)
    return pts.reshape(-1)


def test_bbox_returns_none_without_hand():
    assert landmark_bbox(np.zeros(63, dtype=np.float32), 640, 480) is None


def test_bbox_is_square_and_inside_image():
    box = landmark_bbox(_fake_landmarks(), 640, 480, padding=0.3)
    assert box is not None
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= 640
    assert 0 <= y1 < y2 <= 480
    # Square box keeps the handshape undistorted when resized.
    assert abs((x2 - x1) - (y2 - y1)) <= 2


def test_crop_hand_returns_requested_size():
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    crop = crop_hand(image, _fake_landmarks(), size=160)
    assert crop.shape == (160, 160, 3)


def test_crop_hand_falls_back_to_centre_square_without_landmarks():
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    crop = crop_hand(image, np.zeros(63, dtype=np.float32), size=96)
    assert crop.shape == (96, 96, 3)
