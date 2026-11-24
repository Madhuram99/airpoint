import math

import numpy as np

from airpoint.filters import OneEuroFilter
from airpoint.pointer import BETA, FC_MIN, PointerPipeline
from airpoint.tremor import Reading, TremorCanceller
from tests.pointer_sim import dwell_rate, pointer_session, shake_left


def run(s, mode):
    p = PointerPipeline()
    p.mode = mode
    return np.array([p.update((float(x), float(y)), float(t)) for t, (x, y) in zip(s.t, s.raw)]), p


def test_without_a_tremor_full_is_exactly_smoothing_only():
    s = pointer_session(30, seed=11)
    full, p = run(s, "full")
    smooth, _ = run(s, "smooth")
    assert not p.reading.locked and np.array_equal(full, smooth)


def test_smoothing_only_is_the_plain_one_euro_filter():
    s = pointer_session(10, seed=12)
    smooth, _ = run(s, "smooth")
    fx, fy = OneEuroFilter(FC_MIN, BETA), OneEuroFilter(FC_MIN, BETA)
    plain = np.array([(fx.filter(x, t), fy.filter(y, t)) for t, (x, y) in zip(s.t, s.raw)])
    assert np.array_equal(smooth, plain)


def test_off_returns_the_raw_pointer_and_modes_cycle():
    s = pointer_session(5, seed=13, hz=6.0, amp=14.0)
    off, p = run(s, "off")
    assert np.array_equal(off, s.raw)
    seen = []
    for _ in range(3):
        p.cycle_mode()
        seen.append(p.mode)
    assert seen == ["full", "smooth", "off"]


def test_canceller_leaves_less_than_half_the_shake_of_smoothing_alone():
    for seed in (14, 15):
        s = pointer_session(40, seed=seed, hz=6.0, amp=14.0)
        full, p = run(s, "full")
        smooth, _ = run(s, "smooth")
        assert p.reading.locked and abs(p.reading.lock_hz - 6.0) < 0.8
        assert shake_left(s, full) < 0.5 * shake_left(s, smooth)
        assert p.left.amp < 0.5 * p.reading.amp        # the live readout agrees


def test_dwell_clicking_survives_a_severe_tremor_only_with_the_canceller():
    full_ok, smooth_ok = [], []
    for seed in (16, 17, 18):
        s = pointer_session(60, seed=seed, hz=5.0, amp=25.0)
        full_ok.append(dwell_rate(s, run(s, "full")[0]))
        smooth_ok.append(dwell_rate(s, run(s, "smooth")[0]))
    assert np.mean(full_ok) >= 0.95 and np.mean(smooth_ok) < 0.5


def test_canceller_still_helps_at_a_dim_room_frame_rate():
    s = pointer_session(40, seed=19, fs=16.5, hz=6.0, amp=14.0)
    assert shake_left(s, run(s, "full")[0]) < 0.6 * shake_left(s, run(s, "smooth")[0])


def test_a_gap_in_tracking_resets_the_canceller_so_the_pointer_glides_in_without_ringing():
    s = pointer_session(12, seed=20, hz=6.0, amp=14.0)
    p = PointerPipeline()
    for t, (x, y) in zip(s.t, s.raw):
        p.update((float(x), float(y)), float(t))
    assert p.reading.locked
    t0 = float(s.t[-1]) + 0.8                              # hand away for 0.8 s, back somewhere else
    outs = [p.update((100.0, 80.0), t0 + i / 30) for i in range(20)]
    assert p.reading.locked                                # a short gap keeps the lock
    d = [math.hypot(x - 100.0, y - 80.0) for x, y in outs]
    assert all(b <= a + 1e-9 for a, b in zip(d, d[1:]))    # closes in steadily, never swings back out
    assert d[-1] < 1.0


def test_canceller_reset_settles_it_on_the_new_position():
    c, fs = TremorCanceller(), 30.0
    locked = Reading(status="tremor", hz=6.0, amp=10.0, fs=fs, locked=True, lock_hz=6.0)
    for i in range(90):
        c.step((320.0 + 14.0 * math.sin(2 * math.pi * 6.0 * i / fs), 180.0), i / fs, locked)
    c.reset((100.0, 80.0))
    for i in range(10):
        x, y = c.step((100.0, 80.0), (90 + i) / fs, locked)
        assert abs(x - 100.0) < 1e-9 and abs(y - 80.0) < 1e-9


def test_canceller_passes_through_until_locked_then_removes_the_tone():
    c, fs = TremorCanceller(), 30.0
    tone = [(320.0 + 14.0 * math.sin(2 * math.pi * 6.0 * i / fs), 180.0) for i in range(150)]
    idle = Reading(status="steady", fs=fs)
    assert [c.step(xy, i / fs, idle) for i, xy in enumerate(tone[:30])] == tone[:30]
    locked = Reading(status="tremor", hz=6.0, amp=10.0, fs=fs, locked=True, lock_hz=6.0)
    out = [c.step(xy, (30 + i) / fs, locked) for i, xy in enumerate(tone[30:])]
    assert max(abs(x - 320.0) for x, _ in out[60:]) < 1.5   # 14 px of shake, under 1.5 px left
