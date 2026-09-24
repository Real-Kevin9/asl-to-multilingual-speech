from modules.recognition.stabilizer import SignStabilizer


class FakeClock:
    """Deterministic clock so dwell timing can be tested without sleeping."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t


def make(clock, **kwargs):
    params = dict(
        hold_time=0.9, min_confidence=0.6, agreement=0.65,
        cooldown=0.5, min_samples=4, time_fn=clock,
    )
    params.update(kwargs)
    return SignStabilizer(**params)


def feed(stab, clock, label, conf, duration, step=0.1):
    """Feed predictions for `duration` seconds; return every committed label."""
    committed = []
    elapsed = 0.0
    while elapsed < duration:
        out = stab.update(label, conf)
        if out is not None:
            committed.append(out)
        clock.advance(step)
        elapsed += step
    return committed


def test_single_prediction_does_not_commit():
    """The old behaviour: one frame instantly became a letter."""
    clock = FakeClock()
    stab = make(clock)
    assert stab.update("A", 0.99) is None


def test_brief_hold_below_dwell_does_not_commit():
    clock = FakeClock()
    stab = make(clock)
    # 0.5s of a confident, perfectly stable sign is still too short.
    assert feed(stab, clock, "A", 0.99, duration=0.5) == []


def test_sign_commits_after_hold_time():
    clock = FakeClock()
    stab = make(clock)
    committed = feed(stab, clock, "A", 0.99, duration=1.0)
    assert committed == ["A"]


def test_commit_happens_no_earlier_than_hold_time():
    clock = FakeClock()
    stab = make(clock, hold_time=0.9)
    start = clock.t
    committed_at = None
    for _ in range(40):
        if stab.update("B", 0.95) is not None:
            committed_at = clock.t
            break
        clock.advance(0.1)
    assert committed_at is not None
    assert committed_at - start >= 0.9


def test_low_confidence_never_commits():
    clock = FakeClock()
    stab = make(clock)
    assert feed(stab, clock, "A", 0.2, duration=3.0) == []


def test_flickering_predictions_do_not_commit():
    """Alternating labels are exactly what produced wrong letters before."""
    clock = FakeClock()
    stab = make(clock)
    committed = []
    for i in range(30):
        out = stab.update("A" if i % 2 else "B", 0.95)
        if out is not None:
            committed.append(out)
        clock.advance(0.1)
    assert committed == []


def test_occasional_misread_does_not_restart_the_dwell():
    """One bad frame in a steady hold should be tolerated."""
    clock = FakeClock()
    stab = make(clock)
    committed = []
    for i in range(12):
        label = "X" if i == 4 else "A"  # single outlier
        out = stab.update(label, 0.95)
        if out is not None:
            committed.append(out)
        clock.advance(0.1)
    assert committed == ["A"]


def test_no_hand_resets_the_dwell():
    clock = FakeClock()
    stab = make(clock)
    feed(stab, clock, "A", 0.99, duration=0.6)   # partial hold
    stab.update("nothing", 0.0)                  # hand lowered
    assert feed(stab, clock, "A", 0.99, duration=0.6) == []


def test_holding_yields_double_letter_not_a_stream():
    """Holding one sign for a long time must not spam that letter."""
    clock = FakeClock()
    stab = make(clock)
    committed = feed(stab, clock, "L", 0.99, duration=3.0)
    # hold_time 0.9 + cooldown 0.5 => roughly one commit per 1.4s, not per frame.
    assert 1 <= len(committed) <= 3
    assert set(committed) == {"L"}


def test_switching_sign_commits_the_new_one():
    clock = FakeClock()
    stab = make(clock)
    assert feed(stab, clock, "A", 0.99, duration=1.0) == ["A"]
    committed = feed(stab, clock, "B", 0.99, duration=1.6)
    assert "B" in committed


def test_status_reports_progress_for_ui():
    clock = FakeClock()
    stab = make(clock)
    label, progress = stab.status()
    assert label is None and progress == 0.0

    feed(stab, clock, "C", 0.95, duration=0.4)
    label, progress = stab.status()
    assert label == "C"
    assert 0.0 < progress < 1.0


def test_reset_clears_state():
    clock = FakeClock()
    stab = make(clock)
    feed(stab, clock, "A", 0.99, duration=0.6)
    stab.reset()
    assert stab.status()[0] is None
    assert feed(stab, clock, "A", 0.99, duration=0.6) == []
