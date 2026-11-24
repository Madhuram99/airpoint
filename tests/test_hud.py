import numpy as np

from airpoint import hud


def test_mini_frame_is_320x240_with_a_label_strip():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    small = hud.mini_frame(frame, "MOVING | palm 0.98 | mouse ON")
    assert small.shape == (240, 320, 3)
    assert small[230:, :, :].any()


def test_dwell_ring_draws_only_with_progress():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hud.draw_dwell_ring(frame, (50, 50), 0.0)
    assert not frame.any()
    hud.draw_dwell_ring(frame, (50, 50), 0.5)
    assert frame.any()
