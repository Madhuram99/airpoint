import math
import os

import numpy as np
import pytest

from airpoint import face as F
from airpoint.fsm import Event


def pose(yaw_deg, pitch_deg):
    """A head-pose matrix whose forward axis points yaw degrees to the side and pitch degrees up."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    m = np.eye(4)
    m[:3, :3] = ry @ rx
    return m


def obs(yaw=0.0, pitch=0.0, jaw=0.0, brow=0.0):
    return F.FaceObs(yaw, pitch, {"jawOpen": jaw, "browInnerUp": brow})


def test_head_angles_read_yaw_and_pitch_from_the_pose_matrix():
    for yaw, pitch in ((0, 0), (20, 0), (-15, 0), (0, 10), (0, -8)):
        got = F.head_angles(pose(yaw, pitch))
        assert got[0] == pytest.approx(yaw, abs=1e-6) and got[1] == pytest.approx(pitch, abs=1e-6)


def test_pointer_puts_the_resting_pose_at_the_centre_and_clamps_at_the_edges():
    p = F.FacePointer(dict(F.default_face(), yaw0=2.0, pitch0=-3.0))
    assert p.xy(2.0, -3.0, 640, 360) == (320.0, 180.0)
    assert p.xy(2.0 + 18.0, -3.0, 640, 360)[0] == 640.0                  # the comfortable limit is the screen edge
    assert p.xy(2.0 - 9.0, -3.0, 640, 360)[0] == 160.0
    assert p.xy(2.0, -3.0 + 6.0, 640, 360)[1] == 90.0                    # looking up moves the cursor up
    assert p.xy(90.0, -90.0, 640, 360) == (640.0, 360.0)                 # far beyond: clamped


def test_signs_switch_on_at_the_threshold_and_off_only_well_below_it():
    s = F.FaceSigns(F.default_face())
    assert s.update({"jawOpen": 0.30})[0] == "neutral"
    assert s.update({"jawOpen": 0.50})[0] == "mouth_open"
    assert s.update({"jawOpen": 0.30})[0] == "mouth_open"                # 67% of the threshold: still on
    assert s.update({"jawOpen": 0.20})[0] == "neutral"                   # 44%: off
    assert s.update({"browInnerUp": 0.60})[0] == "brows_up"
    assert s.update({"browInnerUp": 0.60, "jawOpen": 0.9})[0] == "brows_up"   # the first one on keeps the floor


def test_signs_report_probabilities_for_the_xray_and_how_near_a_click_is():
    s = F.FaceSigns(F.default_face())
    label, probs, near = s.update({"jawOpen": 0.18})
    assert label == "neutral" and near == pytest.approx(0.4)
    assert probs.sum() == pytest.approx(1.0) and probs[0] > probs[1] > probs[2] == 0.0
    label, probs, near = s.update({"jawOpen": 0.9})
    assert label == "mouth_open" and near == 1.0 and int(np.argmax(probs)) == 1


def test_a_sign_with_no_threshold_never_fires():
    s = F.FaceSigns(dict(F.default_face(), on={"mouth_open": 0.45}))
    assert s.update({"browInnerUp": 1.0})[0] == "neutral"


def record(c, rows, t0, dt=1 / 30):
    t = t0
    for o in rows:
        c.add(t, o)
        t += dt
    return t


def test_calibration_measures_rest_range_and_thresholds():
    rng = np.random.default_rng(0)
    c = F.FaceCalib(0.0)
    assert c.phase(0.0) == ("rest", 3.0) and c.phase(3.5)[0] == "reach" and c.phase(9.5)[0] == "mouth"
    assert c.phase(12.5)[0] == "brows" and c.phase(15.0) == ("done", 0.0)
    t = record(c, [obs(4 + rng.normal(0, 0.3), -6 + rng.normal(0, 0.3), 0.02, 0.03) for _ in range(90)], 0.0)
    sweep = [obs(4 + 22 * math.sin(i / 14), -6 + 12 * math.sin(i / 9), 0.03, 0.05) for i in range(180)]
    t = record(c, sweep, 3.0)
    t = record(c, [obs(4, -6, 0.8, 0.05) for _ in range(90)], 9.0)
    record(c, [obs(4, -6, 0.03, 0.7) for _ in range(90)], 12.0)
    r = c.finish()
    assert abs(r["yaw0"] - 4) < 0.3 and abs(r["pitch0"] + 6) < 0.3
    assert 18 < r["span_yaw"] < 23 and 9 < r["span_pitch"] < 12.5
    assert 0.35 < r["on"]["mouth_open"] < 0.48 and 0.3 < r["on"]["brows_up"] < 0.45      # halfway from rest to peak
    assert c.dropped == []


def test_calibration_drops_an_expression_it_never_saw_and_needs_a_face():
    c = F.FaceCalib(0.0)
    record(c, [obs(0, 0, 0.0, 0.0) for _ in range(90)], 0.0)
    record(c, [obs(10 * math.sin(i / 10), 5 * math.sin(i / 7)) for i in range(180)], 3.0)
    record(c, [obs(0, 0, 0.7, 0.0) for _ in range(90)], 9.0)
    record(c, [obs(0, 0, 0.0, 0.05) for _ in range(90)], 12.0)            # brows barely moved
    r = c.finish()
    assert c.dropped == ["brows_up"] and list(r["on"]) == ["mouth_open"]
    assert r["span_yaw"] >= 5.0 and r["span_pitch"] >= 4.0
    assert F.FaceCalib(0.0).finish() is None                              # no face seen at all


def test_roles_for_a_face():
    assert F.GRIP == frozenset({"mouth_open"}) and F.HOLD_EVENTS == {"brows_up": Event("right_click")}
    assert F.FACE_LABELS == ("neutral", "mouth_open", "brows_up")


@pytest.mark.skipif(not os.path.exists(F.MODEL_PATH), reason="face model not downloaded")
def test_tracker_sees_no_face_in_a_blank_frame():
    tracker = F.FaceTracker()
    assert tracker.detect(np.zeros((360, 640, 3), np.uint8)) is None
    tracker.close()


def test_tracker_explains_how_to_get_a_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="curl -L -o face_landmarker.task"):
        F.FaceTracker(str(tmp_path / "nope.task"))
