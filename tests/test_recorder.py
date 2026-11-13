import csv

from airpoint.recorder import HEADER, Recorder, RecordSession
from tests.hands import OPEN_PALM


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
