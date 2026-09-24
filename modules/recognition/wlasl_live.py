"""Live webcam helpers for word-level WLASL recognition.

The model classifies a short clip, not a single frame. The user starts recording,
performs one sign, then stops; we subsample the buffer and call ``predict_clip``.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import numpy as np


class ClipRecorder:
    """Buffers BGR frames while the user performs one word-level sign."""

    def __init__(
        self,
        target_fps: float = 12.0,
        max_seconds: float = 4.0,
    ):
        self.target_fps = float(target_fps)
        self.max_seconds = float(max_seconds)
        self._frames: List[np.ndarray] = []
        self._recording = False
        self._started_at: Optional[float] = None
        self._last_keep_at: float = 0.0

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def duration(self, now: Optional[float] = None) -> float:
        if not self._recording or self._started_at is None:
            return 0.0
        return max(0.0, (now or time.monotonic()) - self._started_at)

    def start(self, now: Optional[float] = None) -> None:
        now = now or time.monotonic()
        self._frames = []
        self._recording = True
        self._started_at = now
        self._last_keep_at = 0.0

    def cancel(self) -> None:
        self._frames = []
        self._recording = False
        self._started_at = None

    def add_frame(self, bgr: np.ndarray, now: Optional[float] = None) -> bool:
        """Keep a subsampled frame. Returns False if max duration was hit."""
        if not self._recording:
            return False
        now = now or time.monotonic()
        if self._started_at is None:
            self._started_at = now

        min_gap = 1.0 / max(self.target_fps, 1.0)
        if self._frames and (now - self._last_keep_at) < min_gap:
            return True

        self._frames.append(bgr.copy())
        self._last_keep_at = now

        if self.duration(now) >= self.max_seconds:
            return False
        return True

    def stop(self) -> List[np.ndarray]:
        frames = list(self._frames)
        self.cancel()
        return frames


def assemble_word_gloss(words: Sequence[str]) -> str:
    """Join committed WLASL glosses into a space-separated gloss string."""
    cleaned = [w.strip().upper() for w in words if w and w.strip()]
    return " ".join(cleaned)
