from airpoint.calibration import (CalibSession, Trainer, fc_min_from_jitter, jitter_px,
                                  load_profile, save_profile)


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


def test_profile_round_trip(tmp_path):
    root = str(tmp_path)
    model = str(tmp_path / "models" / "personal_ana.joblib")
    labels = str(tmp_path / "models" / "personal_ana_labels.json")
    path = save_profile(root, "ana", 0.55, model, labels, 0.97)
    assert path == str(tmp_path / "profiles" / "ana.json")
    p = load_profile(root, "ana")
    assert p["user"] == "ana" and p["fc_min"] == 0.55 and p["acc_on_own_frames"] == 0.97
    assert p["model"] == "models/personal_ana.joblib"
    assert p["labels"] == "models/personal_ana_labels.json"
    assert load_profile(root, "nobody") is None
