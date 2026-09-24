from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - environment dependent
    cv2 = None

# Default size fed to the MobileNetV2 backbone. 160x160 is one of the input
# resolutions with official ImageNet weights and yields a 5x5 feature map,
# i.e. a 25-step spatial sequence for the BiLSTM.
DEFAULT_CROP_SIZE = 160


def landmark_bbox(
    landmarks: np.ndarray,
    image_width: int,
    image_height: int,
    padding: float = 0.35,
) -> Optional[Tuple[int, int, int, int]]:
    """Compute a padded pixel bounding box around the hand landmarks.

    MediaPipe returns landmarks normalized to [0, 1] relative to the image, so
    they are scaled back to pixels here.

    Args:
        landmarks: Flat 63-vector (21 landmarks x x,y,z) as produced by
            :meth:`ASLPreprocessor.extract_landmarks`.
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.
        padding: Fraction of the box size added as margin on every side. Some
            padding matters because the fingertips sit right on the box edge and
            the CNN benefits from a little surrounding context.

    Returns:
        ``(x1, y1, x2, y2)`` in pixels, or ``None`` when no hand was detected
        (an all-zero landmark vector).
    """
    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] != 21 or not np.any(pts):
        return None

    xs = pts[:, 0] * image_width
    ys = pts[:, 1] * image_height

    x1, x2 = float(np.min(xs)), float(np.max(xs))
    y1, y2 = float(np.min(ys)), float(np.max(ys))

    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)

    # Expand to a square box so the resize step does not distort the handshape.
    side = max(box_w, box_h) * (1.0 + 2 * padding)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    nx1 = int(round(cx - side / 2))
    ny1 = int(round(cy - side / 2))
    nx2 = int(round(cx + side / 2))
    ny2 = int(round(cy + side / 2))

    # Clamp to image bounds.
    nx1 = max(0, min(nx1, image_width - 1))
    ny1 = max(0, min(ny1, image_height - 1))
    nx2 = max(nx1 + 1, min(nx2, image_width))
    ny2 = max(ny1 + 1, min(ny2, image_height))

    return nx1, ny1, nx2, ny2


def crop_hand(
    image: np.ndarray,
    landmarks: Optional[np.ndarray],
    size: int = DEFAULT_CROP_SIZE,
    padding: float = 0.35,
) -> np.ndarray:
    """Crop the hand region from an image and resize it to ``size`` x ``size``.

    Cropping is what makes the CNN branch viable: the raw dataset mixes tight
    hand photos with full webcam scenes (whole person visible), and live webcam
    frames look like the latter. Cropping to the hand normalizes all of those
    into one consistent, hand-centred distribution.

    When no landmarks are available the largest possible centre square is used
    so the function always returns a usable image.
    """
    if cv2 is None:
        raise ImportError("opencv (cv2) is required for hand cropping")

    height, width = image.shape[:2]
    box = landmark_bbox(landmarks, width, height, padding=padding) if landmarks is not None else None

    if box is None:
        side = min(width, height)
        x1 = (width - side) // 2
        y1 = (height - side) // 2
        crop = image[y1:y1 + side, x1:x1 + side]
    else:
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        crop = image

    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def crop_hand_from_path(
    image_path: str,
    landmarks: Optional[np.ndarray],
    size: int = DEFAULT_CROP_SIZE,
    padding: float = 0.35,
) -> Optional[np.ndarray]:
    """Read an image from disk and return the resized hand crop (BGR)."""
    if cv2 is None:
        raise ImportError("opencv (cv2) is required for hand cropping")

    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return crop_hand(image, landmarks, size=size, padding=padding)
