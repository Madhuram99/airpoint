from airpoint.calibration import CalibSession, fc_min_from_jitter, jitter_px


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
    assert s.update(8.0) == ("poses_countdown", "palm", 3.0)
    assert s.fc_min == fc_min_from_jitter(jitter_px(s.still_points))
    assert s.update(11.0) == ("capturing", "palm", 5.0)
    assert s.capturing and s.fresh
    s.fresh = False
    assert s.update(16.0) == ("poses_countdown", "pinch", 3.0)
    assert s.update(19.0)[0] == "capturing"
    assert s.update(24.0) == ("training", None, 0.0)
    assert not s.capturing
