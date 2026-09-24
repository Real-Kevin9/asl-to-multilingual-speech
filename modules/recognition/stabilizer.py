"""Temporal stabilisation of per-frame sign predictions.

A raw classifier emits one label per frame. Committing those directly makes the
system feel twitchy and produces wrong letters, because frames captured while
the hand is still moving into position get classified as whatever handshape they
momentarily resemble.

:class:`SignStabilizer` requires a sign to be *held* before it counts. A label is
committed only when it has dominated a rolling window for ``hold_time`` seconds
with sufficient confidence, which both introduces the deliberate ~1 s dwell a
user expects and filters out transitional poses.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from typing import Callable, Deque, Optional, Tuple

# Labels that mean "no sign is being made"; they reset the dwell instead of
# being treated as a candidate.
IDLE_LABELS = {"nothing", "none", ""}


class SignStabilizer:
    """Commits a sign only after it has been held steadily.

    Args:
        hold_time: Seconds a sign must be held before it is committed.
        min_confidence: Mean confidence the winning label must reach.
        agreement: Fraction of samples in the window that must share the winning
            label. Below 1.0 so occasional misclassified frames do not restart
            the dwell.
        cooldown: Quiet period after a commit. This is what allows double letters
            (``LL``): keep holding and the next commit lands one dwell later,
            rather than the same sign firing on every subsequent frame.
        min_samples: Minimum predictions in the window, so a commit cannot happen
            off one or two frames if the camera stalls.
        time_fn: Clock source; injectable to keep tests deterministic.
    """

    def __init__(
        self,
        hold_time: float = 0.9,
        min_confidence: float = 0.6,
        agreement: float = 0.65,
        cooldown: float = 0.5,
        min_samples: int = 4,
        time_fn: Callable[[], float] = time.monotonic,
    ):
        self.hold_time = hold_time
        self.min_confidence = min_confidence
        self.agreement = agreement
        self.cooldown = cooldown
        self.min_samples = min_samples
        self.time_fn = time_fn

        self._samples: Deque[Tuple[float, str, float]] = deque()
        self._observing_since: Optional[float] = None
        self._cooldown_until: float = 0.0

    # -- internals ----------------------------------------------------------
    def _clear(self) -> None:
        self._samples.clear()
        self._observing_since = None

    def _winner(self) -> Tuple[Optional[str], float, float]:
        """Return (label, agreement_fraction, mean_confidence) for the window."""
        if not self._samples:
            return None, 0.0, 0.0

        counts = Counter(label for _, label, _ in self._samples)
        label, count = counts.most_common(1)[0]
        fraction = count / len(self._samples)
        confs = [c for _, l, c in self._samples if l == label]
        return label, fraction, sum(confs) / len(confs)

    # -- public API ---------------------------------------------------------
    def reset(self) -> None:
        """Forget all state, including any active cooldown."""
        self._clear()
        self._cooldown_until = 0.0

    def update(
        self,
        label: str,
        confidence: Optional[float],
        now: Optional[float] = None,
    ) -> Optional[str]:
        """Feed one prediction.

        Returns:
            The committed label once the dwell completes, otherwise ``None``.
        """
        now = self.time_fn() if now is None else now
        confidence = 0.0 if confidence is None else float(confidence)

        # No hand / no sign: abandon the current dwell so the user can restart.
        if label in IDLE_LABELS:
            self._clear()
            return None

        if now < self._cooldown_until:
            return None

        if self._observing_since is None:
            self._observing_since = now
        self._samples.append((now, label, confidence))

        # Keep only the trailing hold_time window for the majority vote.
        cutoff = now - self.hold_time
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        if now - self._observing_since < self.hold_time:
            return None
        if len(self._samples) < self.min_samples:
            return None

        winner, fraction, mean_conf = self._winner()
        if winner is None:
            return None
        if fraction < self.agreement or mean_conf < self.min_confidence:
            return None

        self._clear()
        self._cooldown_until = now + self.cooldown
        return winner

    def status(self, now: Optional[float] = None) -> Tuple[Optional[str], float]:
        """Current candidate and dwell progress in ``[0, 1]``, for UI feedback.

        A dwell gate is confusing without a visible countdown, so the caller can
        render how close the current sign is to being accepted.
        """
        now = self.time_fn() if now is None else now

        if now < self._cooldown_until:
            return None, 0.0
        if self._observing_since is None or not self._samples:
            return None, 0.0

        winner, _, _ = self._winner()
        progress = (now - self._observing_since) / self.hold_time
        return winner, max(0.0, min(1.0, progress))
