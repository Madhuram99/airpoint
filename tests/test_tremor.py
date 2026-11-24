import numpy as np

from airpoint.features import normalize_63
from airpoint.tremor import TremorMeter, TremorSim, estimate
from tests.hands import BY_POSE
from airpoint.sim_pointer import pointer_session


def test_sim_moves_every_landmark_together_and_leaves_features_unchanged():
    sim, base, xs = TremorSim(hz=6.0, amp=0.2), BY_POSE["palm"], []
    for i in range(90):
        out = sim.apply(base, i / 30, 640, 360)
        assert np.allclose(normalize_63(out), normalize_63(base), atol=1e-9)
        xs.append(out[9, 0] * 640)
    assert base[9, 0] == 0.5                              # the input is not modified
    swing, palm_px = (max(xs) - min(xs)) / 2, 0.2 * 360
    assert 0.7 * 0.2 * palm_px < swing < 1.6 * 0.2 * palm_px


def test_sim_amplitude_scales_with_palm_length():
    near, far = TremorSim(), TremorSim()
    a = max(abs(near.offset(i / 30, 100.0)[0]) for i in range(60))
    b = max(abs(far.offset(i / 30, 50.0)[0]) for i in range(60))
    assert abs(a / b - 2.0) < 1e-9


def test_sim_nudges_clamp_to_their_ranges():
    sim = TremorSim()
    for _ in range(40):
        sim.nudge(d_hz=0.5, d_amp=0.05)
    assert (sim.hz, sim.amp) == (10.0, 0.60)
    for _ in range(40):
        sim.nudge(d_hz=-0.5, d_amp=-0.05)
    assert (sim.hz, sim.amp) == (3.5, 0.05)


def run_meter(s):
    """Feed a whole session through a TremorMeter. -> list of (t, Reading), one per sample."""
    m, out = TremorMeter(), []
    for t, xy in zip(s.t, s.raw):
        m.add(t, xy)
        out.append((t, m.read(t)))
    return out


def test_meter_waits_for_enough_samples():
    m = TremorMeter()
    for i in range(20):
        m.add(i / 30, (320.0, 180.0))
    assert m.read(0.7).status == "wait"


def test_meter_locks_fast_and_reads_frequency_and_size():
    s = pointer_session(20, seed=1, hz=6.0, amp=14.0)
    readings = run_meter(s)
    first_lock = next(t for t, r in readings if r.locked)
    assert first_lock <= 2.5
    for end in (8.0, 12.0, 16.0):
        r = next(r for t, r in readings if t >= end)
        win = (s.t > end - 3.0) & (s.t <= end)
        true_amp = float(np.hypot(s.trem[win, 0].std(), s.trem[win, 1].std()))
        assert r.locked and abs(r.lock_hz - s.freq[win].mean()) <= 0.4
        assert abs(r.amp / true_amp - 1.0) <= 0.25


def test_meter_never_locks_without_a_tremor():
    for seed in (2, 3):
        assert not any(r.locked for _, r in run_meter(pointer_session(40, seed=seed)))


def test_meter_lets_go_within_four_seconds_of_the_tremor_stopping():
    s = pointer_session(30, seed=4, hz=6.0, amp=14.0, stop=15.0)
    readings = run_meter(s)
    assert next(r for t, r in readings if t >= 14.0).locked
    released = next(t for t, r in readings if t >= 15.0 and not r.locked)
    assert released - 15.0 <= 4.0


def test_meter_at_a_dim_room_frame_rate_sees_only_slower_tremor():
    s = pointer_session(20, seed=5, fs=16.5, hz=5.0, amp=14.0)
    r = run_meter(s)[-1][1]
    assert r.locked and abs(r.lock_hz - 5.0) <= 0.6
    assert 6.5 <= r.max_hz <= 7.2                         # 0.42 x 16.5 fps


def test_meter_refuses_a_camera_that_is_too_slow():
    s = pointer_session(12, seed=6, fs=10.0, hz=4.0, amp=14.0)
    r = run_meter(s)[-1][1]
    assert r.status == "slow" and not r.locked


def test_estimate_reports_the_spectrum_it_judged():
    s = pointer_session(4, seed=7, hz=6.0, amp=14.0)
    r = estimate(s.t[-90:], s.raw[-90:])
    freqs, power = r.spectrum
    assert r.status == "tremor" and len(freqs) == len(power) == 91 and abs(freqs[int(np.argmax(power))] - r.hz) < 1e-9
