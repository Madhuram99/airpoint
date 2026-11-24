"""Whole flows through Session with synthetic hands, a FakeMouse and a blank image. No camera."""
import csv
import json
import time

import numpy as np

from airpoint.actions import ActionRunner, FakeMouse
from airpoint.classifier import RuleClassifier
from airpoint.session import Session
from tests.hands import BY_POSE
from tests.test_train import _write_user

DT = 1 / 30


def make(tmp_path, user="ana", mouse_on=True, **kw):
    mouse = FakeMouse()
    runner = ActionRunner(mouse, mouse.size())
    if mouse_on:
        runner.toggle()
    return Session(runner, RuleClassifier(), None, root=str(tmp_path), user=user, **kw), mouse


def blank(w=640, h=360):
    return np.zeros((h, w, 3), dtype=np.uint8)


def feed(s, pose, seconds, t, dt=DT):
    """Tick with one synthetic pose (None = no hand) for `seconds`. -> the time after the last tick."""
    for _ in range(round(seconds / dt)):
        s.tick(blank(), None if pose is None else BY_POSE[pose], "Right" if pose else None, t)
        t += dt
    return t


def kinds(mouse):
    return [c[0] for c in mouse.calls if c[0] != "move"]


def test_short_pinch_is_one_click_through_the_whole_pipeline(tmp_path):
    s, mouse = make(tmp_path)
    t = feed(s, "palm", 0.5, 0.0)
    t = feed(s, "pinch", 0.2, t)
    feed(s, "palm", 0.3, t)
    assert kinds(mouse) == ["click"]
    assert any(c[0] == "move" for c in mouse.calls)


def test_mouse_off_means_no_os_calls(tmp_path):
    s, mouse = make(tmp_path, mouse_on=False)
    t = feed(s, "palm", 0.5, 0.0)
    t = feed(s, "pinch", 0.2, t)
    feed(s, "palm", 0.3, t)
    assert mouse.calls == []


def test_held_pinch_drags_and_leaving_live_mode_releases_the_button(tmp_path):
    s, mouse = make(tmp_path)
    t = feed(s, "palm", 0.5, 0.0)
    t = feed(s, "pinch", 0.8, t)
    assert kinds(mouse) == ["down"]
    s.on_key(ord("r"), t)
    assert s.mode.name == "record" and kinds(mouse) == ["down", "up"]


def test_dwell_clicks_when_the_palm_stays_still(tmp_path):
    s, mouse = make(tmp_path, dwell_on=True)
    feed(s, "palm", 1.5, 0.0)
    assert kinds(mouse) == ["click"]
    assert s.last_event == "click"


def test_dwell_key_toggles(tmp_path):
    s, mouse = make(tmp_path)
    s.on_key(ord("d"), 0.0)
    feed(s, "palm", 1.5, 0.0)
    assert kinds(mouse) == ["click"]


def test_hand_lost_mid_drag_releases(tmp_path):
    s, mouse = make(tmp_path)
    t = feed(s, "palm", 0.5, 0.0)
    t = feed(s, "pinch", 0.8, t)
    feed(s, None, 0.3, t)
    assert kinds(mouse) == ["down", "up"]


def test_mini_view_is_320x240_in_live_mode_only(tmp_path):
    s, _ = make(tmp_path)
    s.on_key(ord("v"), 0.0)
    assert s.view_dirty
    assert s.tick(blank(), BY_POSE["palm"], "Right", 0.0).shape == (240, 320, 3)
    s.on_key(ord("r"), 0.1)
    assert s.tick(blank(), BY_POSE["palm"], "Right", 0.2).shape == (360, 640, 3)


class _Mlp:
    name, labels = "mlp", ["palm"]

    def predict(self, pts):
        return "palm", 1.0


def test_k_switches_classifier_and_clears_the_vote_buffer(tmp_path):
    mouse = FakeMouse()
    mlp = _Mlp()
    s = Session(ActionRunner(mouse, mouse.size()), RuleClassifier(), mlp, root=str(tmp_path))
    assert s.clf is mlp
    feed(s, "palm", 0.2, 0.0)
    s.on_key(ord("k"), 0.3)
    assert s.clf is s.rules and len(s.smoother.buf) == 0
    s.on_key(ord("k"), 0.4)
    assert s.clf is mlp


def test_record_and_calibrate_need_a_user(tmp_path, capsys):
    s, _ = make(tmp_path, user=None)
    s.on_key(ord("r"), 0.0)
    s.on_key(ord("c"), 0.0)
    assert s.mode.name == "live"
    assert "needs --user" in capsys.readouterr().out


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_record_mode_writes_rows_and_the_guard_rejects_the_wrong_sign(tmp_path):
    s, _ = make(tmp_path)
    s.on_key(ord("r"), 0.0)
    s.on_key(ord(" "), 0.0)
    t = 0.0
    while s.mode.session.phase != "done":
        s.tick(blank(), BY_POSE[s.mode.session.pose], "Right", t)
        t += DT
    path = tmp_path / "data" / "ana.csv"
    s.recorder.f.flush()                      # rows are buffered until the file is closed
    by_pose = {p: sum(r["pose"] == p for r in _rows(path)) for p in BY_POSE}
    assert all(295 <= n <= 302 for n in by_pose.values()), by_pose
    assert s.rejected == 0 and s.recorder.count == sum(by_pose.values())

    s.on_key(ord("1"), t)                     # redo palm, but show a fist the whole time
    while s.mode.session.phase != "done":
        s.tick(blank(), BY_POSE["fist"], "Right", t)
        t += DT
    assert s.rejected >= 290
    assert sum(r["pose"] == "palm" for r in _rows(path)) == 0   # old palm rows dropped, none accepted

    s.on_key(ord("r"), t)
    assert s.mode.name == "live"
    s.close()


def test_calibration_measures_trains_hot_swaps_and_saves_a_profile(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_user(data / "base.csv", "base", np.random.default_rng(0), 1.0, np.array([0.0, 0.0, 0.0]))
    s, _ = make(tmp_path, user="ana")
    s.on_key(ord("c"), 0.0)
    assert s.mode.name == "calibrate"
    t, dt = 0.0, 0.1
    while s.mode.calib.phase != "training":
        pose = s.mode.calib.pose if s.mode.calib.phase == "poses" else "palm"
        s.tick(blank(), BY_POSE[pose], "Right", t)
        t += dt
    s.on_key(ord("c"), t)                     # ignored while training
    assert s.mode.name == "calibrate"
    deadline = time.time() + 60
    while s.mode.calib.phase in ("training",) and time.time() < deadline:
        s.tick(blank(), BY_POSE["palm"], "Right", t)
        t += dt
        time.sleep(0.02)
    assert s.mode.calib.phase == "done", s.mode.calib.message
    assert s.clf.name == "mlp/ana" and s.clf is s.mlp
    assert s.pointer.fc_min == 1.0            # a perfectly still synthetic hand
    profile = json.loads((tmp_path / "profiles" / "ana.json").read_text())
    assert profile["fc_min"] == 1.0 and profile["model"] == "models/personal_ana.joblib"
    assert (tmp_path / "models" / "personal_ana.joblib").exists()
    s.on_key(ord("c"), t)
    assert s.mode.name == "live"
    s.close()


def test_failed_retrain_keeps_the_old_model(tmp_path):
    def boom(*a):
        raise ValueError("only 3 usable calibration frames; redo calibration")

    s, _ = make(tmp_path, user="ana", train_fn=boom)
    old = s.clf
    s.on_key(ord("c"), 0.0)
    t = 0.0
    while s.mode.calib.phase not in ("done", "failed") and t < 400:
        pose = s.mode.calib.pose if s.mode.calib.phase == "poses" else "palm"
        s.tick(blank(), BY_POSE[pose], "Right", t)
        t += 0.1
        if s.mode.calib.phase == "training":
            time.sleep(0.01)
    assert s.mode.calib.phase == "failed" and "only 3 usable" in s.mode.calib.message
    assert s.clf is old
    s.close()
