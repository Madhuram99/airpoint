import math
from airpoint.filters import OneEuroFilter


def test_first_sample_passes_through():
    f = OneEuroFilter(fc_min=0.8, beta=0.01)
    assert f.filter(100.0, t=0.0) == 100.0


def test_steady_jitter_is_damped():
    f = OneEuroFilter(fc_min=0.8, beta=0.01)
    outs = [f.filter(200.0 + 14.0 * math.sin(2 * math.pi * 6 * (i / 30)), t=i / 30) for i in range(60)]
    settled = outs[30:]
    assert max(abs(v - 200.0) for v in settled) < 7.0  # raw peak is 14 px


def test_fast_move_is_followed_closely():
    f = OneEuroFilter(fc_min=0.8, beta=0.01)
    for i in range(30):
        f.filter(200.0, t=i / 30)
    out = None
    for i in range(30, 60):
        out = f.filter(800.0, t=i / 30)
    assert abs(out - 800.0) < 20.0


def test_custom_fc_min_lowers_cutoff_for_more_smoothing():
    slow, fast = OneEuroFilter(fc_min=0.8), OneEuroFilter(fc_min=0.8)
    for i in range(10):
        slow.filter(0.0, t=i / 30, custom_fc_min=0.3)
        fast.filter(0.0, t=i / 30)
    assert slow.filter(100.0, t=11 / 30, custom_fc_min=0.3) < fast.filter(100.0, t=11 / 30)
