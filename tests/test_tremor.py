import numpy as np

from airpoint.features import normalize_63
from airpoint.tremor import TremorSim
from tests.hands import BY_POSE


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
