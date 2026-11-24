"""Whole flows through Session with synthetic hands, a FakeMouse and a blank image. No camera."""
import csv
import json
import time

import numpy as np

from airpoint.actions import ActionRunner, FakeMouse
from airpoint.classifier import RuleClassifier
from airpoint.effects import Effects
from airpoint.profile import Profile
from airpoint.modes.live import shake_line
from airpoint.tremor import Reading, TremorSim
from airpoint.session import Session
from airpoint.sounds import Sounds
from tests.hands import BY_POSE
from tests.test_train import _write_user

DT = 1 / 30


def make(tmp_path, user="ana", mouse_on=True, **kw):
    mouse = FakeMouse()
    runner = ActionRunner(mouse, mouse.size())
    if mouse_on:
        runner.toggle()
    played = []                                     # sound files the session asked for; nothing is played
    done = []                                       # commands the session asked to run; none is executed
    kw.setdefault("effects", Effects(run=lambda cmd: done.append(cmd)))
    s = Session(runner, RuleClassifier(), None, root=str(tmp_path), user=user,
                sounds=Sounds(player=lambda path: played.append(path.rsplit("/", 1)[1])), **kw)
    s.played, s.done = played, done
    return s, mouse


def blank(w=640, h=360):
    return np.full((h, w, 3), 120, dtype=np.uint8)      # a lit room; a black frame would raise the "too dark" hint


def dark(w=640, h=360):
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
    assert s.tick(blank(), BY_POSE["palm"], "Right", 0.2).shape == (540, 960, 3)   # full view at zoom 1.5


def test_z_cycles_the_zoom_of_the_full_view(tmp_path):
    s, _ = make(tmp_path)
    shapes = []
    for i in range(3):
        shapes.append(s.tick(blank(), None, None, i * DT).shape[:2])
        s.view_dirty = False
        s.on_key(ord("z"), i * DT)
        assert s.view_dirty
    assert shapes == [(540, 960), (720, 1280), (360, 640)]


def test_the_pointer_does_not_depend_on_camera_resolution(tmp_path):
    def run(w, h):
        s, mouse = make(tmp_path)
        t = 0.0
        for i in range(60):
            pts = BY_POSE["palm"] + np.array([0.002 * i, -0.001 * i, 0.0])   # hand drifting right and up
            s.tick(blank(w, h), pts, "Right", t)
            t += DT
        return mouse.calls

    assert run(640, 360) == run(1920, 1080) == run(1280, 720)


def test_the_pointer_keeps_sub_pixel_precision(tmp_path):
    s, _ = make(tmp_path)
    pts = BY_POSE["palm"] + np.array([0.00037, 0.0, 0.0])
    for i in range(45):
        s.tick(blank(), pts, "Right", i * DT)
    assert abs(s.pointer.out[0] - (0.50037 * 640)) < 0.01      # 320.2368, not 320


class _Mlp:
    name, labels = "mlp", ["palm"]

    def proba(self, pts):
        return np.array([1.0])


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
    assert profile["region"] == [0.375, 0.475, 0.625, 0.725]      # the hand never moved: the minimum box around it
    assert s.runner.active_region == (0.375, 0.475, 0.625, 0.725)
    assert abs(s.pointer.gain - 0.70 / 0.25) < 1e-9               # default box is 0.70 wide
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


def test_t_shakes_the_pointer_but_not_the_sign(tmp_path):
    s, mouse = make(tmp_path)
    s.on_key(ord("t"), 0.0)
    xs, t = [], 0.0
    for _ in range(90):
        s.tick(blank(), BY_POSE["palm"], "Right", t)
        xs.append(s.obs.raw[0])
        t += DT
    assert s.sim.on and np.std(xs) > 5.0
    assert s.fsm.state == "MOVING" and kinds(mouse) == []
    s.on_key(ord("]"), t)
    s.on_key(ord("}"), t)
    assert (s.sim.amp, s.sim.hz) == (0.25, 6.5)


def test_simulated_tremor_cannot_follow_you_into_record_or_calibrate(tmp_path):
    s, _ = make(tmp_path)
    s.on_key(ord("t"), 0.0)
    s.on_key(ord("r"), 0.0)
    assert s.mode.name == "record" and not s.sim.on
    s.on_key(ord("t"), 0.0)
    assert not s.sim.on                       # t only works in live mode
    s.on_key(ord("r"), 0.0)
    s.on_key(ord("t"), 0.0)
    s.on_key(ord("c"), 0.0)
    assert s.mode.name == "calibrate" and not s.sim.on
    s.close()


def test_simulated_tremor_is_locked_and_cancelled_in_the_live_loop(tmp_path):
    s, _ = make(tmp_path)
    s.on_key(ord("t"), 0.0)
    t = feed(s, "palm", 6.0, 0.0)
    r = s.pointer.reading
    assert r.locked and abs(r.lock_hz - 6.0) < 0.8 and r.amp > 5.0
    assert s.pointer.left.amp < 0.5 * r.amp
    s.on_key(ord("b"), t)
    assert s.pointer.mode == "smooth"
    s.on_key(ord("b"), t)
    assert s.pointer.mode == "off"
    feed(s, "palm", 0.1, t)
    assert s.pointer.out == s.obs.raw                        # off = the raw pointer


def test_shake_line_says_what_is_measured_and_what_is_left():
    sim, idle = TremorSim(), Reading(status="steady")
    assert shake_line(sim, idle, idle) is None
    sim.on = True
    assert shake_line(sim, idle, idle) == "SIMULATED TREMOR 6.0 Hz | measuring..."
    locked = Reading(status="tremor", amp=12.4, locked=True, lock_hz=6.1)
    left = Reading(status="steady", amp=0.83)
    assert shake_line(sim, locked, left) == "SIMULATED TREMOR 6.0 Hz | LOCKED 6.1 Hz: shake 12 px -> 0.8 px"
    sim.on = False
    assert shake_line(sim, locked, left) == "LOCKED 6.1 Hz: shake 12 px -> 0.8 px"


def test_mini_label_carries_the_simulator_and_lock_tags(tmp_path):
    s, _ = make(tmp_path)
    s.on_key(ord("t"), 0.0)
    feed(s, "palm", 6.0, 0.0)
    assert " | SIM" in s.mini_label and " | LOCK " in s.mini_label


def test_x_adds_the_xray_panel_in_live_mode_only(tmp_path):
    s, _ = make(tmp_path)
    s.zoom = 1.0
    s.on_key(ord("x"), 0.0)
    assert s.view_dirty
    t = feed(s, "palm", 0.5, 0.0)
    assert s.tick(blank(), BY_POSE["palm"], "Right", t).shape == (360, 640 + 400, 3)
    assert s.tick(blank(), None, None, t + DT).shape == (360, 1040, 3)          # no hand: still draws
    s.on_key(ord("v"), t)
    assert s.tick(blank(), BY_POSE["palm"], "Right", t + 2 * DT).shape == (240, 320, 3)   # mini wins
    s.on_key(ord("v"), t)
    s.on_key(ord("r"), t)
    assert s.tick(blank(), BY_POSE["palm"], "Right", t + 3 * DT).shape == (360, 640, 3)   # not while recording
    s.close()


def test_xray_taps_report_the_live_pipeline(tmp_path):
    s, _ = make(tmp_path)
    t = feed(s, "palm", 0.5, 0.0)
    feed(s, "pinch", 0.2, t)
    x = s.taps()
    assert x["labels"][int(np.argmax(x["probs"]))] == "pinch" and x["label"] == "pinch"
    assert x["state"] == "PRESSING" and 0.0 < x["state_s"] < 0.2
    assert x["votes"][-3:] == ["pinch"] * 3 and x["gate"] == 0.75
    assert len(x["trail"]) == 21 and x["mode"] == "full" and not x["sim"]


def test_live_state_file_tells_the_lab_what_is_driving_the_cursor(tmp_path):
    path = tmp_path / "results" / "live.json"
    s, _ = make(tmp_path, dwell_on=True)
    t = feed(s, "palm", 0.5, 0.0)
    live = json.loads(path.read_text())
    assert live["mouse"] and live["dwell"] and live["filter"] == "full" and live["mode"] == "live"
    assert live["sim"] == {"on": False, "hz": 6.0, "amp": 0.2} and live["user"] == "ana"
    assert live["classifier"] == "rules" and abs(live["ts"] - time.time()) < 5
    assert live["locked"] is False and live["gain"] == 1.0
    s.on_key(ord("t"), t)
    s.on_key(ord("b"), t)
    t = feed(s, "palm", 0.3, t)                      # under a second since the last write
    assert json.loads(path.read_text())["filter"] == "full"
    feed(s, "palm", 0.5, t)
    live = json.loads(path.read_text())
    assert live["filter"] == "smooth" and live["sim"]["on"]
    s.close()
    assert not path.exists()                          # a closed AirPoint is "not running" at once


def test_p_pauses_everything_and_a_still_open_palm_resumes(tmp_path):
    s, mouse = make(tmp_path)
    t = feed(s, "palm", 0.5, 0.0)
    s.on_key(ord("p"), t)
    assert s.runner.locked and s.played == ["Submarine.aiff"]
    before = len(mouse.calls)
    t = feed(s, "pinch", 0.2, t)                     # a click attempt while paused
    t = feed(s, "palm", 1.5, t)                      # not yet two seconds of stillness
    assert s.runner.locked and len(mouse.calls) == before and s.last_event == "none"
    t = feed(s, "palm", 0.8, t)
    assert not s.runner.locked and s.played[-1] == "Glass.aiff"
    t = feed(s, "pinch", 0.2, t)
    feed(s, "palm", 0.3, t)
    assert kinds(mouse) == ["click"]


def test_sounds_follow_real_actions_only(tmp_path):
    s, _ = make(tmp_path)
    t = feed(s, "palm", 0.5, 0.0)
    t = feed(s, "pinch", 0.2, t)
    t = feed(s, "palm", 0.3, t)
    assert s.played == ["Tink.aiff"]
    s.on_key(ord("s"), t)                            # sounds off
    t = feed(s, "pinch", 0.2, t)
    t = feed(s, "palm", 0.3, t)
    assert s.played == ["Tink.aiff"]
    quiet, _ = make(tmp_path, mouse_on=False)        # mouse off: the click never reached the system
    t = feed(quiet, "palm", 0.5, 0.0)
    t = feed(quiet, "pinch", 0.2, t)
    feed(quiet, "palm", 0.3, t)
    assert quiet.played == []


def test_hints_appear_after_a_while_and_h_turns_them_off(tmp_path):
    s, _ = make(tmp_path)
    t = 0.0
    for _ in range(100):                             # 3.3 s with no hand, in a lit room
        s.tick(blank(), None, None, t)
        t += DT
    assert s.hint == "show your hand to the camera" and s.mini_label.startswith("! show your hand")
    for _ in range(100):                             # a black frame: the room is dark
        s.tick(dark(), None, None, t)
        t += DT
    assert s.hint == "too dark: put more light on your hand"
    s.on_key(ord("h"), t)
    s.tick(dark(), None, None, t + DT)
    assert s.hint is None


def test_calibration_notices_what_the_hand_cannot_do_and_falls_back_to_dwell(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_user(data / "base.csv", "base", np.random.default_rng(0), 1.0, np.array([0.0, 0.0, 0.0]))
    s, _ = make(tmp_path, user="ana")
    cannot = {"pinch", "fist", "thumb_down"}         # this user shows an open palm instead of these
    s.on_key(ord("c"), 0.0)
    t, dt = 0.0, 0.1
    while s.mode.calib.phase != "training":
        asked = s.mode.calib.pose if s.mode.calib.phase == "poses" else "palm"
        s.tick(blank(), BY_POSE["palm" if asked in cannot else asked], "Right", t)
        t += dt
    deadline = time.time() + 60
    while s.mode.calib.phase == "training" and time.time() < deadline:
        s.tick(blank(), BY_POSE["palm"], "Right", t)
        t += dt
        time.sleep(0.02)
    assert s.mode.calib.phase == "done", s.mode.calib.message
    profile = json.loads((tmp_path / "profiles" / "ana.json").read_text())
    assert sorted(profile["skipped"]) == ["fist", "pinch", "thumb_down"]
    assert profile["roles"] == {"grip": [], "right_click": ["v_sign"], "scroll_up": ["thumb_up"], "scroll_down": []}
    assert profile["dwell"] is True and s.dwell_on                # no grip sign left: dwell click took over
    assert "skipped" in s.mode.calib.message and "dwell" in s.mode.calib.message
    assert s.fsm.grip == frozenset() and s.fsm.scroll == {"thumb_up": 1}
    s.close()


def _speaking_profile():
    p = Profile(user="ana")
    p.roles["right_click"] = []                      # free the V-sign so it can stand in for a taught sign
    p.signs = {"v_sign": {"action": "say", "arg": "I need water", "hold_s": 0.6}}
    return p


def test_a_bound_sign_speaks_once_per_hold_even_with_the_mouse_off(tmp_path):
    s, mouse = make(tmp_path, mouse_on=False, profile=_speaking_profile())
    t = feed(s, "palm", 0.5, 0.0)
    t = feed(s, "v_sign", 0.5, t)                    # under the 0.6 s hold (plus the vote delay)
    assert s.done == []
    t = feed(s, "v_sign", 1.0, t)
    assert s.done == [["say", "I need water"]] and mouse.calls == []
    t = feed(s, "palm", 0.3, t)
    feed(s, "v_sign", 1.0, t)
    assert s.done == [["say", "I need water"]] * 2


def test_nothing_is_spoken_while_paused_and_a_pause_sign_pauses(tmp_path):
    s, _ = make(tmp_path, profile=_speaking_profile())
    s.on_key(ord("p"), 0.0)
    t = feed(s, "palm", 0.2, 0.0)
    feed(s, "v_sign", 1.2, t)
    assert s.done == []

    p = Profile(user="ana")
    p.roles["right_click"] = []
    p.signs = {"v_sign": {"action": "pause", "arg": None, "hold_s": 0.6}}
    s, _ = make(tmp_path, profile=p)
    t = feed(s, "palm", 0.5, 0.0)
    feed(s, "v_sign", 1.0, t)
    assert s.runner.locked and s.played == ["Submarine.aiff"]
