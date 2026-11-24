import csv

from airpoint.recorder import HEADER, Recorder, RecordSession, looks_like
from tests.hands import BY_POSE, FIST, OPEN_PALM, PINCH, THUMB_UP, V_SIGN


def test_every_textbook_pose_looks_like_its_own_label():
    for pose, pts in BY_POSE.items():
        assert looks_like(pose, pts), pose


def test_gross_drift_is_rejected():
    assert not looks_like("palm", FIST)         # hand closed during a palm capture
    assert not looks_like("pinch", OPEN_PALM)   # tips apart during a pinch capture
    assert not looks_like("fist", OPEN_PALM)
    assert not looks_like("v_sign", FIST)
    assert not looks_like("thumb_up", OPEN_PALM)
    assert not looks_like("thumb_down", THUMB_UP)


def test_mild_variation_is_accepted():
    assert looks_like("pinch", FIST)            # relaxed-finger pinch: tips close, rest curled
    assert looks_like("palm", V_SIGN)           # two fingers up still counts as an open hand


def test_recorder_writes_header_once_and_rows_with_63_coords(tmp_path):
    path = tmp_path / "a.csv"
    r = Recorder(str(path), "a")
    r.write("palm", 1.5, "Right", OPEN_PALM)
    r.close()
    r2 = Recorder(str(path), "a")
    r2.write("pinch", 2.0, "Right", OPEN_PALM)
    r2.close()
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 2 and len(HEADER) == 4 + 63
    assert rows[0]["pose"] == "palm" and rows[1]["pose"] == "pinch"
    assert float(rows[0]["x0"]) == 0.5 and float(rows[0]["y0"]) == 0.8
    assert r2.count == 1


def test_session_counts_down_then_captures_then_advances_then_finishes():
    s = RecordSession(("palm", "pinch"))
    assert s.update(0.0) == ("idle", "palm", 0.0)
    s.start(0.0)
    assert s.update(0.0) == ("countdown", "palm", 3.0)
    assert s.update(2.9)[0] == "countdown"
    assert s.update(3.0) == ("capturing", "palm", 10.0)
    assert s.capturing
    assert s.update(12.9)[0] == "capturing"
    assert s.update(13.0) == ("countdown", "pinch", 3.0)
    assert s.update(16.0)[0] == "capturing"
    assert s.update(26.0) == ("done", None, 0.0)
    assert not s.capturing


def test_drop_removes_only_that_users_pose_rows(tmp_path):
    path = tmp_path / "a.csv"
    r = Recorder(str(path), "a")
    r.write("palm", 1.0, "Right", OPEN_PALM)
    r.write("palm", 1.1, "Right", OPEN_PALM)
    r.write("pinch", 1.2, "Right", OPEN_PALM)
    assert r.drop("palm") == 2
    r.write("palm", 2.0, "Right", OPEN_PALM)
    r.close()
    rows = list(csv.DictReader(open(path)))
    assert [(x["pose"], x["ts"]) for x in rows] == [("pinch", "1.200"), ("palm", "2.000")]


def test_capture_start_sets_fresh_once_per_pose():
    s = RecordSession(("palm", "pinch"))
    s.start(0.0)
    s.update(2.0)
    assert not s.fresh
    s.update(3.0)
    assert s.fresh
    s.fresh = False
    s.update(5.0)
    assert not s.fresh
    s.update(13.0)
    s.update(16.0)
    assert s.fresh and s.pose == "pinch"


def test_select_records_a_single_pose_then_finishes():
    s = RecordSession(("palm", "pinch", "fist"))
    s.select(0, 0.0)
    s.update(3.0)
    assert s.update(13.0) == ("done", "palm", 0.0)


def test_session_select_jumps_to_a_pose_and_redo_restarts_current():
    s = RecordSession(("palm", "pinch", "fist"))
    s.select(2, 5.0)
    assert s.update(5.0) == ("countdown", "fist", 3.0)
    s.update(8.0)
    s.redo(9.0)
    assert s.update(9.0) == ("countdown", "fist", 3.0)
    s.update(12.0)
    s.update(22.0)
    assert s.update(22.0)[0] == "done"
    s.redo(23.0)
    assert s.update(23.0) == ("countdown", "palm", 3.0)


def test_session_capture_length_is_configurable():
    s = RecordSession(("palm",), capture_s=5.0)
    s.start(0.0)
    assert s.update(3.0) == ("capturing", "palm", 5.0)
    assert s.update(7.9)[0] == "capturing"
    assert s.update(8.0)[0] == "done"
