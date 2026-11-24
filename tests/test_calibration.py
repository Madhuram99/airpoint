import numpy as np

from airpoint.calibration import CalibSession, Trainer, comfort_box, fc_min_from_jitter, jitter_px


def test_fc_min_maps_jitter_linearly_and_clamps():
    assert fc_min_from_jitter(0.0) == 1.0
    assert fc_min_from_jitter(3.0) == 1.0
    assert fc_min_from_jitter(7.5) == 0.65
    assert fc_min_from_jitter(12.0) == 0.3
    assert fc_min_from_jitter(40.0) == 0.3


def test_jitter_is_spread_of_the_raw_cursor():
    assert jitter_px([(1, 1)]) == 0.0
    assert jitter_px([(0, 0), (0, 0)]) == 0.0
    assert abs(jitter_px([(0, 0), (6, 0), (0, 0), (6, 0)]) - 3.0) < 1e-9


def test_calib_session_phases():
    s = CalibSession(("palm", "pinch"), 0.0)
    assert s.update(0.0) == ("countdown", None, 3.0)
    assert s.update(3.0) == ("still", None, 5.0)
    s.still_points += [(100, 100), (104, 100), (100, 104), (104, 104)]
    assert s.update(7.9)[0] == "still"
    assert s.update(8.0) == ("reach", None, 6.0)
    assert s.fc_min == fc_min_from_jitter(jitter_px(s.still_points))
    assert s.update(13.9)[0] == "reach"
    assert s.update(14.0) == ("poses_countdown", "palm", 3.0)
    assert s.update(17.0) == ("capturing", "palm", 5.0)
    assert s.capturing and s.fresh
    s.fresh = False
    assert s.update(22.0) == ("poses_countdown", "pinch", 3.0)
    assert s.update(25.0)[0] == "capturing"
    assert s.update(30.0) == ("training", None, 0.0)
    assert not s.capturing


def test_comfort_box_is_the_reach_padded_and_kept_inside_the_frame():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0.30, 0.70, 400) * 640, rng.uniform(0.40, 0.80, 400) * 360])
    x0, y0, x1, y1 = comfort_box(pts, 640, 360)
    assert 0.24 < x0 < 0.31 and 0.69 < x1 < 0.76              # 5th-95th percentile plus 10%
    assert 0.34 < y0 < 0.41 and 0.79 < y1 < 0.86


def test_comfort_box_is_never_smaller_than_a_quarter_of_the_frame_and_needs_enough_samples():
    still = [(320.0, 300.0)] * 100                             # the hand did not move at all
    x0, y0, x1, y1 = comfort_box(still, 640, 360)
    assert abs((x1 - x0) - 0.25) < 1e-6 and abs((y1 - y0) - 0.25) < 1e-6
    assert abs((x0 + x1) / 2 - 0.5) < 1e-6
    corner = [(630.0, 355.0)] * 100                            # reach hugging the corner: shifted inside
    x0, y0, x1, y1 = comfort_box(corner, 640, 360)
    assert x1 == 0.98 and y1 == 0.98 and abs((x1 - x0) - 0.25) < 1e-6
    assert comfort_box(still[:10], 640, 360) is None


def test_trainer_reports_result_or_error_off_thread():
    ok = Trainer(lambda: {"acc": 1.0})
    ok.start()
    ok.join(2.0)
    assert ok.done and ok.result == {"acc": 1.0} and ok.error is None

    bad = Trainer(lambda: 1 / 0)
    bad.start()
    bad.join(2.0)
    assert bad.done and bad.result is None
    assert bad.error == "ZeroDivisionError: division by zero"


def test_trainer_is_not_done_before_it_runs():
    assert not Trainer(lambda: {}).done
