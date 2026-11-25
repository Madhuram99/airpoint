"""Whole flows through Session with synthetic hands, a FakeMouse and a blank image. No camera."""
import csv
import json
import time

import numpy as np

from airpoint.actions import ActionRunner, FakeMouse
from airpoint.classifier import RuleClassifier
from airpoint.effects import Effects
from airpoint.face import FaceObs
from airpoint.helper import Helper
from airpoint.profile import Profile
from airpoint.modes.live import helper_line, shake_line, talk_line
from airpoint.tremor import Reading, TremorSim
from airpoint.session import Session
from airpoint.sounds import Sounds
from airpoint.speech import Talker
from tests.hands import BY_POSE
from tests.test_helper import Script, Shot, step
from tests.test_speech import Ears, Mic
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
    kw.setdefault("helper", Helper(Script(), Shot, mouse.size(), sleep=lambda s: None))   # never the real screen or model
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
    assert live["locked"] is False and live["gain"] == 1.0 and live["wheel"] is False
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


def feed_at(s, dx, dy, seconds, t):
    """Open palm shifted by (dx, dy) view pixels from where the fixtures hold it."""
    pts = BY_POSE["palm"] + np.array([dx / 640.0, dy / 360.0, 0.0])
    for _ in range(round(seconds / DT)):
        s.tick(blank(), pts, "Right", t)
        t += DT
    return t


def open_wheel(s, dx, dy, t):
    """Hold the palm at (dx, dy) until the wheel opens there. -> time."""
    pts = BY_POSE["palm"] + np.array([dx / 640.0, dy / 360.0, 0.0])
    for _ in range(90):
        s.tick(blank(), pts, "Right", t)
        t += DT
        if s.wheel.mode == "open":
            return t
    raise AssertionError("the wheel did not open")


def test_w_turns_dwell_into_a_wheel_and_staying_centred_clicks(tmp_path):
    s, mouse = make(tmp_path)
    s.on_key(ord("w"), 0.0)
    t = open_wheel(s, 0, 0, 0.0)
    assert kinds(mouse) == []
    moves = sum(c[0] == "move" for c in mouse.calls)
    t = feed_at(s, 3, 2, 0.5, t)                         # the hand drifts a little: the system cursor must not
    assert sum(c[0] == "move" for c in mouse.calls) == moves and kinds(mouse) == []
    t = feed_at(s, 3, 2, 0.5, t)
    assert kinds(mouse) == ["click"] and s.wheel.mode == "idle" and s.played == ["Tink.aiff"]


def test_wheel_directions_reach_right_click_and_pause(tmp_path):
    s, mouse = make(tmp_path)
    s.on_key(ord("w"), 0.0)
    t = open_wheel(s, 0, 0, 0.0)
    t = feed_at(s, 26, -15, 1.0, t)                      # up and to the right: the second item, right click
    assert kinds(mouse) == ["right_click"]
    t = open_wheel(s, 120, 0, t)                         # move away, stop: it opens again there
    t = feed_at(s, 120 - 26, -15, 1.0, t)                # up and to the left: the sixth item, pause
    assert s.runner.locked and s.wheel.mode == "idle"


def test_wheel_drag_is_dropped_when_wheel_mode_or_live_mode_ends(tmp_path):
    for leave in (ord("w"), ord("r")):
        s, mouse = make(tmp_path)
        s.on_key(ord("w"), 0.0)
        t = open_wheel(s, 0, 0, 0.0)
        t = feed_at(s, 26, 15, 1.0, t)                   # down and to the right: the third item, drag
        assert kinds(mouse) == ["down"] and s.wheel.mode == "drag"
        s.on_key(leave, t)
        assert kinds(mouse) == ["down", "up"] and s.wheel.mode == "idle"
        s.close()


class FakeOverlay:
    alive = True

    def __init__(self):
        self.states, self.closed = [], False

    def send(self, state):
        self.states.append(state)
        return True

    def close(self):
        self.closed = True


def test_overlay_gets_the_cursor_position_ring_and_pulses(tmp_path):
    ov = FakeOverlay()
    s, mouse = make(tmp_path, overlay=ov, dwell_on=True)
    t = feed(s, "palm", 0.6, 0.0)
    last = ov.states[-1]
    move = [c for c in mouse.calls if c[0] == "move"][-1]
    assert last["show"] and abs(last["x"] - move[1]) <= 1 and abs(last["y"] - move[2]) <= 1
    assert 0.4 < last["ring"] < 0.7 and last["tag"] is None and "wheel" not in last
    t = feed(s, "palm", 0.6, t)
    assert [st["pulse"] for st in ov.states if "pulse" in st] == ["click"]      # the dwell click, reported once
    s.on_key(ord("p"), t)
    t = feed(s, "palm", 0.2, t)
    assert ov.states[-1]["tag"] == "PAUSED"


def test_overlay_is_hidden_when_the_mouse_is_off_outside_live_mode_and_after_o(tmp_path):
    ov = FakeOverlay()
    s, _ = make(tmp_path, overlay=ov, mouse_on=False)
    t = feed(s, "palm", 0.3, 0.0)
    assert ov.states[-1] == {"show": False}
    s.on_key(ord("m"), t)
    t = feed(s, "palm", 0.3, t)
    assert ov.states[-1]["show"]
    s.on_key(ord("r"), t)                                # record mode: no glow
    assert ov.states[-1] == {"show": False}
    s.on_key(ord("r"), t)
    t = feed(s, "palm", 0.2, t)
    assert ov.states[-1]["show"]
    s.on_key(ord("o"), t)                                # overlay off
    n = len(ov.states)
    assert ov.states[-1] == {"show": False}
    feed(s, "palm", 0.3, t)
    assert len(ov.states) == n
    s.close()
    assert ov.closed


def test_overlay_shows_the_wheel_at_its_anchor_and_scroll_and_simulated_tags(tmp_path):
    ov = FakeOverlay()
    s, _ = make(tmp_path, overlay=ov)
    s.on_key(ord("w"), 0.0)
    t = open_wheel(s, 0, 0, 0.0)
    anchor = (ov.states[-1]["x"], ov.states[-1]["y"])
    t = feed_at(s, 26, -15, 0.3, t)
    st = ov.states[-1]
    assert (st["x"], st["y"]) == anchor                  # the glow stays where the cursor waits
    assert st["wheel"]["pick"] == 1 and 0.0 < st["wheel"]["progress"] < 1.0 and len(st["wheel"]["items"]) == 6
    s.on_key(ord("w"), t)
    s.on_key(ord("t"), t)
    t = feed(s, "palm", 0.2, t)
    assert ov.states[-1]["tag"] == "SIMULATED" and "wheel" not in ov.states[-1]
    s.on_key(ord("t"), t)
    t = feed(s, "thumb_up", 0.5, t)
    assert ov.states[-1]["tag"] == "SCROLL"


def face_feed(s, seconds, t, yaw=0.0, pitch=0.0, jaw=0.0, brow=0.0, seen=True):
    """Tick in face mode with one synthetic face observation for `seconds`. -> time after."""
    for _ in range(round(seconds / DT)):
        s.tick(blank(), None, None, t, face=FaceObs(yaw, pitch, {"jawOpen": jaw, "browInnerUp": brow}) if seen else None)
        t += DT
    return t


def last_move(mouse):
    return [c for c in mouse.calls if c[0] == "move"][-1][1:]


def test_f_switches_to_the_face_and_the_head_moves_the_cursor(tmp_path):
    s, mouse = make(tmp_path)
    s.on_key(ord("f"), 0.0)
    assert s.source == "face" and s.wheel_on and s.runner.active_region == (0.0, 0.0, 1.0, 1.0)
    t = face_feed(s, 1.0, 0.0)
    cx, cy = last_move(mouse)
    assert abs(cx - 500) <= 2 and abs(cy - 250) <= 2            # resting pose = the middle of the screen
    t = face_feed(s, 1.5, t, yaw=9.0, pitch=6.0)                # half the comfortable range to the right and up
    x, y = last_move(mouse)
    assert 730 < x < 755 and 115 < y < 140                      # three quarters across, a quarter down
    assert s.taps()["labels"] == ["neutral", "mouth_open", "brows_up"] and s.state()["source"] == "face"


def test_mouth_open_clicks_or_drags_and_raised_brows_right_click(tmp_path):
    s, mouse = make(tmp_path)
    s.on_key(ord("f"), 0.0)
    s.on_key(ord("w"), 0.0)                                      # wheel off: test the expressions alone
    t = face_feed(s, 0.5, 0.0)
    t = face_feed(s, 0.2, t, jaw=0.8)
    t = face_feed(s, 0.4, t)
    assert kinds(mouse) == ["click"]
    t = face_feed(s, 0.9, t, jaw=0.8)                            # held open: drag
    assert kinds(mouse) == ["click", "down"]
    t = face_feed(s, 0.4, t)
    assert kinds(mouse) == ["click", "down", "up"]
    t = face_feed(s, 0.7, t, brow=0.8)
    assert kinds(mouse) == ["click", "down", "up", "right_click"]
    face_feed(s, 0.5, t, seen=False)                             # face lost: resting, nothing more
    assert s.fsm.state == "RESTING"


def test_face_mode_refuses_hand_only_things_and_f_restores_the_hand(tmp_path, capsys):
    s, _ = make(tmp_path)
    s.on_key(ord("f"), 0.0)
    for key in "rnt":
        s.on_key(ord(key), 0.0)
    assert s.mode.name == "live" and not s.sim.on
    assert capsys.readouterr().out.count("press f first") == 3
    s.on_key(ord("f"), 0.0)
    assert s.source == "hand" and not s.wheel_on
    assert s.fsm.grip == frozenset({"pinch", "fist"}) and s.runner.active_region == (0.15, 0.15, 0.85, 0.80)
    t = feed(s, "palm", 0.5, 0.0)
    assert s.taps()["labels"] == s.clf.labels


def test_face_hints_talk_about_the_face(tmp_path):
    s, _ = make(tmp_path)
    s.on_key(ord("f"), 0.0)
    face_feed(s, 3.4, 0.0, seen=False)
    assert s.hint == "show your face to the camera"


def test_face_calibration_sets_range_and_thresholds_and_saves_them(tmp_path):
    import math
    s, mouse = make(tmp_path)
    s.on_key(ord("f"), 0.0)
    s.on_key(ord("c"), 0.0)
    assert s.mode.name == "face_calibrate"
    t = face_feed(s, 3.0, 0.0, yaw=3.0, pitch=-4.0, jaw=0.02, brow=0.02)
    for i in range(180):                                         # 6 s of looking around
        s.tick(blank(), None, None, t, face=FaceObs(3.0 + 20 * math.sin(i / 18), -4.0 + 10 * math.sin(i / 12), {"jawOpen": 0.03, "browInnerUp": 0.03}))
        t += DT
    t = face_feed(s, 3.0, t, yaw=3.0, pitch=-4.0, jaw=0.9)
    t = face_feed(s, 3.0, t, yaw=3.0, pitch=-4.0, brow=0.04)     # this face barely moves its brows
    t = face_feed(s, 0.2, t, yaw=3.0, pitch=-4.0)
    assert "not seen: brows_up" in s.mode.message
    saved = json.loads((tmp_path / "profiles" / "ana.json").read_text())["face"]
    assert abs(saved["yaw0"] - 3.0) < 0.5 and 14 < saved["span_yaw"] <= 20 and list(saved["on"]) == ["mouth_open"]
    s.on_key(ord(" "), t)
    assert s.mode.name == "live" and s.source == "face" and s.fsm.hold_events == {}     # no brows: no right-click sign
    t = face_feed(s, 0.6, t, yaw=3.0, pitch=-4.0)
    cx, cy = last_move(mouse)
    assert abs(cx - 500) <= 3 and abs(cy - 250) <= 3            # the calibrated resting pose is now the centre
    s.close()


def test_an_uncalibrated_face_is_centred_on_wherever_it_first_looks(tmp_path):
    s, mouse = make(tmp_path)
    s.on_key(ord("f"), 0.0)
    t = face_feed(s, 1.2, 0.0, yaw=-31.0, pitch=-8.0)            # sitting side-on to the camera
    cx, cy = last_move(mouse)
    assert abs(cx - 500) <= 3 and abs(cy - 250) <= 3            # that pose is the middle of the screen
    assert abs(s.face_cfg["yaw0"] + 31.0) < 0.01 and s.face_centring is None
    face_feed(s, 1.5, t, yaw=-31.0 + 9.0, pitch=-8.0)
    assert 730 < last_move(mouse)[0] < 755
    assert s.profile.face is None                                # a quick centre is not a calibration: not saved


def test_a_calibrated_face_keeps_its_own_centre(tmp_path):
    p = Profile(user="ana")
    p.face = {"yaw0": 5.0, "pitch0": 0.0, "span_yaw": 10.0, "span_pitch": 8.0, "on": {"mouth_open": 0.4}}
    s, mouse = make(tmp_path, profile=p)
    s.on_key(ord("f"), 0.0)
    assert s.face_centring is None
    face_feed(s, 1.0, 0.0, yaw=10.0)
    assert 740 < last_move(mouse)[0] < 755                       # 5 degrees past a centre of 5, in a 10 degree span


def hotkeys(mouse):
    return [c[1:] for c in mouse.calls if c[0] == "hotkey"]


def run_until_idle(s, t):
    """Tick with an open palm until the talker has its answer. -> time."""
    deadline = time.time() + 5
    while s.talker.state != "idle" and time.time() < deadline:
        s.tick(blank(), BY_POSE["palm"], "Right", t)
        t += DT
        time.sleep(0.005)
    assert s.talker.state == "idle"
    return t


def test_l_starts_and_ends_system_dictation_and_needs_the_mouse_switch(tmp_path):
    s, mouse = make(tmp_path)
    assert not s.talker.local and s.state()["talk"] == "idle"
    t = feed(s, "palm", 0.3, 0.0)
    s.on_key(ord("l"), t)
    assert s.talker.state == "listening" and hotkeys(mouse) == [("control", "option", "d")]
    assert s.state()["talk"] == "listening" and s.last_event == "talk: listening" and s.played == []
    t = feed(s, "palm", 0.3, t)
    s.on_key(ord("l"), t)
    assert s.talker.state == "idle" and len(hotkeys(mouse)) == 2 and s.last_event == "talk: stopped"
    s.on_key(ord("l"), t)
    s.on_key(ord("m"), t)                            # mouse off while listening: dictation is ended first
    assert s.talker.state == "idle" and len(hotkeys(mouse)) == 4 and not s.runner.enabled
    s.on_key(ord("l"), t)                            # and it does not start with the mouse off
    assert s.talker.state == "idle" and len(hotkeys(mouse)) == 4 and "mouse ON" in s.last_event


def test_a_pause_leaving_the_live_view_or_quitting_ends_listening(tmp_path):
    s, mouse = make(tmp_path)
    t = feed(s, "palm", 0.3, 0.0)
    s.on_key(ord("l"), t)
    s.on_key(ord("p"), t)
    assert s.talker.state == "idle" and len(hotkeys(mouse)) == 2 and s.runner.locked
    s.on_key(ord("l"), t)                            # paused: talk does not start
    assert s.talker.state == "idle" and len(hotkeys(mouse)) == 2 and "paused" in s.last_event
    s.on_key(ord("p"), t)
    s.on_key(ord("l"), t)
    s.on_key(ord("r"), t)                            # into record mode
    assert s.mode.name == "record" and s.talker.state == "idle" and len(hotkeys(mouse)) == 4
    s.on_key(ord("l"), t)                            # l belongs to the live view only
    assert s.talker.state == "idle"
    s.on_key(ord("r"), t)
    s.on_key(ord("l"), t)
    s.close()
    assert len(hotkeys(mouse)) == 6


def test_system_dictation_is_ended_after_twenty_seconds(tmp_path):
    s, mouse = make(tmp_path)
    t = feed(s, "palm", 0.3, 0.0)
    s.on_key(ord("l"), t)
    t = feed(s, "palm", 19.0, t, dt=0.1)
    assert s.talker.state == "listening" and len(hotkeys(mouse)) == 1
    feed(s, "palm", 1.5, t, dt=0.1)
    assert s.talker.state == "idle" and len(hotkeys(mouse)) == 2


def _talking_profile():
    p = Profile(user="ana")
    p.roles["right_click"] = []                      # free the V-sign so it can stand in for a taught sign
    p.signs = {"v_sign": {"action": "talk", "arg": None, "hold_s": 0.6}}
    return p


def test_a_sign_bound_to_talk_starts_listening_and_the_same_sign_ends_it(tmp_path):
    s, mouse = make(tmp_path, profile=_talking_profile())
    t = feed(s, "palm", 0.5, 0.0)
    t = feed(s, "v_sign", 1.0, t)
    assert s.talker.state == "listening" and len(hotkeys(mouse)) == 1 and s.last_event == "talk: listening"
    t = feed(s, "v_sign", 1.0, t)                    # still held: one tap, not two
    assert s.talker.state == "listening"
    t = feed(s, "palm", 0.4, t)
    feed(s, "v_sign", 1.0, t)
    assert s.talker.state == "idle" and len(hotkeys(mouse)) == 2


def test_local_engine_types_what_was_said_and_carries_out_spoken_commands(tmp_path):
    ears = Ears("I will be late.")
    s, mouse = make(tmp_path, talker=Talker(ears, Mic()))
    t = feed(s, "palm", 0.3, 0.0)
    s.on_key(ord("l"), t)
    assert s.talker.state == "listening" and hotkeys(mouse) == [] and s.played == ["Morse.aiff"]
    t = feed(s, "palm", 0.5, t)
    s.on_key(ord("l"), t)
    assert s.talker.state == "thinking" and s.played == ["Morse.aiff", "Frog.aiff"]
    t = run_until_idle(s, t)
    assert ("write", "I will be late. ") in mouse.calls and s.last_event == "type I will be late. "

    ears.text = "Scroll down."
    s.on_key(ord("l"), t)
    s.on_key(ord("l"), t + 0.5)
    t = run_until_idle(s, t + 0.5)
    assert ("scroll", -5) in mouse.calls and s.last_event == "scroll -5"

    ears.text = "Pause"
    s.on_key(ord("l"), t)
    s.on_key(ord("l"), t + 0.5)
    run_until_idle(s, t + 0.5)
    assert s.runner.locked


def test_what_was_said_is_dropped_when_the_mouse_went_off_meanwhile(tmp_path):
    s, mouse = make(tmp_path, talker=Talker(Ears("I will be late."), Mic()))
    t = feed(s, "palm", 0.3, 0.0)
    s.on_key(ord("l"), t)
    s.on_key(ord("m"), t + 1.0)                      # off: the recording stops and is still worked out
    assert s.talker.state == "thinking"
    run_until_idle(s, t + 1.0)
    assert not any(c[0] == "write" for c in mouse.calls)


def test_in_face_mode_a_speaking_mouth_is_not_a_click(tmp_path):
    s, mouse = make(tmp_path)
    s.on_key(ord("f"), 0.0)
    s.on_key(ord("w"), 0.0)                          # wheel off: the mouth alone
    t = face_feed(s, 0.5, 0.0)
    s.on_key(ord("l"), t)
    t = face_feed(s, 0.3, t, jaw=0.8)
    t = face_feed(s, 0.3, t)
    t = face_feed(s, 0.9, t, jaw=0.8, brow=0.8)
    assert kinds(mouse) == ["hotkey"]                # only the dictation shortcut: no click, no drag
    s.on_key(ord("l"), t)
    t = face_feed(s, 0.5, t, jaw=0.8)                # the last word is still being finished
    t = face_feed(s, 0.8, t)
    assert kinds(mouse) == ["hotkey", "hotkey"]
    t = face_feed(s, 0.2, t, jaw=0.8)                # after that the mouth is a sign again
    face_feed(s, 0.4, t)
    assert kinds(mouse) == ["hotkey", "hotkey", "click"]


def test_the_top_of_the_wheel_starts_and_ends_talk(tmp_path):
    s, mouse = make(tmp_path)
    s.on_key(ord("w"), 0.0)
    t = open_wheel(s, 0, 0, 0.0)
    t = feed_at(s, 0, -30, 1.0, t)                   # straight up: talk
    assert s.talker.state == "listening" and kinds(mouse) == ["hotkey"]
    t = open_wheel(s, 120, 0, t)
    feed_at(s, 120, -30, 1.0, t)
    assert s.talker.state == "idle" and kinds(mouse) == ["hotkey", "hotkey"]


def test_overlay_and_banner_say_listening_then_what_was_heard(tmp_path):
    ov = FakeOverlay()
    s, mouse = make(tmp_path, overlay=ov, talker=Talker(Ears("Scroll down."), Mic()))
    t = feed(s, "palm", 0.5, 0.0)
    assert talk_line(s.talker, t) is None
    s.on_key(ord("l"), t)
    t = feed(s, "palm", 0.2, t)
    assert ov.states[-1]["tag"] == "LISTENING" and "LISTENING" in s.mini_label
    line = talk_line(s.talker, t)
    assert line.startswith("LISTENING [l]: say a command or a sentence") and line.endswith("20 s")
    s.on_key(ord("l"), t)
    assert talk_line(s.talker, t) == "THINKING: working out what you said"
    s.tick(blank(), BY_POSE["palm"], "Right", t)
    assert ov.states[-1]["tag"] in ("THINKING", None)            # the answer may already be in
    t = run_until_idle(s, t + DT)
    assert talk_line(s.talker, t) == 'heard "Scroll down."' and talk_line(s.talker, t + 5.0) is None
    quiet = Talker(Ears(""), Mic())
    quiet.toggle(0.0)
    quiet.toggle(1.0)
    while quiet.state != "idle":
        quiet.poll(1.5)
    assert talk_line(quiet, 2.0) == "talk: heard nothing"
    system = Talker()
    system.toggle(0.0)
    assert "macOS dictation is typing" in talk_line(system, 8.0) and talk_line(system, 8.0).endswith("12 s")


# ---- the AI helper in the live loop

def helped(tmp_path, *steps, armed=True, **kw):
    """A session whose helper answers from a script. -> (session, mouse, script)."""
    script = Script(*steps)
    h = Helper(script, Shot, (1000, 500), sleep=lambda s: None)
    h.armed = armed
    s, mouse = make(tmp_path, helper=h, **kw)
    return s, mouse, script


def until(s, t, pose="palm", leave="thinking"):
    """Tick until the helper has left a state. -> time."""
    deadline = time.time() + 5
    while s.helper.state == leave and time.time() < deadline:
        s.tick(blank(), BY_POSE[pose], "Right", t)
        t += DT
        time.sleep(0.003)
    assert s.helper.state != leave
    return t


def type_keys(s, text, t=0.0):
    for ch in text:
        s.on_key(ord(ch), t)


def test_a_switches_the_helper_on_under_a_consent_line_and_an_errand_runs_shown_step_by_step(tmp_path):
    s, mouse, script = helped(tmp_path, step(x=300, y=400, target="Note field"), step("type", text="milk"),
                              step("done", text="I wrote milk in your note."), armed=False)
    t = feed(s, "palm", 0.3, 0.0)
    s.on_key(ord("a"), t)
    assert s.mode.name == "ask" and s.typing and s.mode.first and not s.helper.armed
    s.tick(blank(), BY_POSE["palm"], "Right", t)                     # the prompt draws
    type_keys(s, "write milk in my notex", t)
    s.on_key(127, t)
    s.on_key(13, t)
    assert s.mode.name == "live" and s.helper.armed and s.helper.state == "thinking"
    assert s.last_event == "asked the AI: write milk in my note" and s.state()["helper"] == "thinking"
    before = len(mouse.calls)
    t = until(s, t)
    assert mouse.calls[before:] == [("move", 300, 200)]              # the cursor goes to the target and waits there
    assert s.helper.state == "showing" and 'AI 1/12: click "Note field" in 1.' in helper_line(s, t)
    t = feed_at(s, 40, 0, 0.5, t)                                    # the hand drifts: it does not move the cursor
    t = feed(s, "pinch", 0.3, t)                                     # and a pinch is not a click
    assert mouse.calls[before:] == [("move", 300, 200)]
    t = feed(s, "palm", 1.0, t)
    assert mouse.calls[before:] == [("move", 300, 200), ("move", 300, 200), ("click",)] and s.played == ["Tink.aiff"]
    t = until(s, feed(s, "palm", 1.3, t))
    t = feed(s, "palm", 1.6, t)
    assert mouse.calls[-1] == ("write", "milk")
    t = until(s, feed(s, "palm", 1.3, t))
    assert not s.helper.active and s.last_event == "AI done: I wrote milk in your note."
    assert s.done == [["say", "I wrote milk in your note."]] and s.played[-1] == "Hero.aiff"
    assert helper_line(s, t) == "AI done: I wrote milk in your note." and helper_line(s, t + 9.0) is None
    n = len(mouse.calls)
    feed_at(s, 60, 0, 0.3, t)                                        # the hand has the cursor again
    assert len(mouse.calls) > n and mouse.calls[-1][0] == "move"
    assert script.calls[1][1] == ['click "Note field"']


def test_enter_on_an_empty_prompt_only_switches_it_on_and_esc_does_nothing(tmp_path):
    s, _, script = helped(tmp_path, armed=False)
    s.on_key(ord("a"), 0.0)
    s.on_key(27, 0.0)
    assert s.mode.name == "live" and not s.helper.armed
    s.on_key(ord("a"), 0.0)
    s.on_key(13, 0.0)
    assert s.mode.name == "live" and s.helper.armed and not s.helper.active and script.calls == []
    s.on_key(ord("a"), 0.0)
    assert s.mode.name == "ask" and not s.mode.first                 # now it is just the request prompt


def test_the_helper_will_not_switch_on_without_what_it_needs(tmp_path, capsys):
    h = Helper(Script(), Shot, (1000, 500), check=lambda: "no API key: set OPENAI_API_KEY")
    s, _ = make(tmp_path, helper=h)
    s.on_key(ord("a"), 0.0)
    assert s.mode.name == "live" and not h.armed and s.last_event == "AI helper: no API key: set OPENAI_API_KEY"
    assert "AI helper: no API key" in capsys.readouterr().out


def _asking_profile():
    p = Profile(user="ana")
    p.roles["right_click"] = []
    p.signs = {"v_sign": {"action": "ask", "arg": "read my newest mail", "hold_s": 0.6}}
    return p


def test_a_sign_can_carry_an_errand_but_only_once_the_helper_is_on_and_the_mouse_too(tmp_path):
    s, mouse, script = helped(tmp_path, step("done", text="No new mail."), armed=False, profile=_asking_profile())
    t = feed(s, "palm", 0.5, 0.0)
    t = feed(s, "v_sign", 1.0, t)
    assert not s.helper.active and s.last_event == "AI helper is off: a switches it on"
    s.helper.armed = True
    s.on_key(ord("m"), t)
    t = feed(s, "palm", 0.4, t)
    t = feed(s, "v_sign", 1.0, t)
    assert not s.helper.active and "mouse ON" in s.last_event
    s.on_key(ord("m"), t)
    t = feed(s, "palm", 0.4, t)
    t = feed(s, "v_sign", 1.0, t)
    assert s.helper.task == "read my newest mail"
    until(s, t, pose="v_sign")
    assert s.last_event == "AI done: No new mail." and script.calls[0][0] == "read my newest mail"


def test_a_spoken_request_starting_with_computer_goes_to_the_helper(tmp_path):
    script = Script(step("done", text="It is open."))
    h = Helper(script, Shot, (1000, 500), sleep=lambda s: None)
    h.armed = True
    s, mouse = make(tmp_path, helper=h, talker=Talker(Ears("Computer, open the calculator."), Mic()))
    t = feed(s, "palm", 0.3, 0.0)
    s.on_key(ord("l"), t)
    s.on_key(ord("l"), t + 0.5)
    t = run_until_idle(s, t + 0.5)
    assert s.helper.task == "open the calculator." and not any(c[0] == "write" for c in mouse.calls)
    until(s, t)
    assert script.calls[0][0] == "open the calculator."


def test_a_fist_a_quick_move_or_a_key_stops_the_helper_and_slow_drift_does_not(tmp_path):
    s, mouse, _ = helped(tmp_path, step(), step(), step(), step())
    t = feed(s, "palm", 0.3, 0.0)
    s.ask_helper("do something", t)
    t = until(s, t)
    t = feed(s, "fist", 0.5, t)
    assert not s.helper.active and s.last_event == "AI: you made the stop sign: the helper stopped"
    assert s.played[-1] == "Basso.aiff"
    t = feed(s, "fist", 0.2, t)                                      # the fist is still up, then opens:
    t = feed(s, "palm", 0.5, t)                                      # it stopped the helper, it is not a click
    assert kinds(mouse) == [] and not s.hands_off
    t = feed(s, "pinch", 0.2, t)                                     # after an open palm, signs count again
    t = feed(s, "palm", 0.3, t)
    assert kinds(mouse) == ["click"]

    t = feed(s, "palm", 0.5, t)
    s.ask_helper("do something", t)
    t = until(s, t)
    for i in range(20):                                              # 60 px in two thirds of a second: a slow drift
        t = feed_at(s, 3 * i, 0, DT, t)
    assert s.helper.active
    for i in range(6):                                               # then 150 px in a fifth of a second
        t = feed_at(s, 60 + 30 * i, 0, DT, t)
    assert not s.helper.active and s.last_event == "AI: you moved: the helper stopped"

    t = feed(s, "palm", 1.2, t)
    s.ask_helper("do something", t)
    t = until(s, t)
    s.on_key(ord("x"), t)
    assert not s.helper.active and "a key was pressed" in s.last_event and s.mode.name == "live"


def test_a_hand_that_leaves_and_comes_back_elsewhere_has_not_taken_over(tmp_path):
    s, _, _ = helped(tmp_path, step(), step())
    t = feed(s, "palm", 0.3, 0.0)
    s.ask_helper("do something", t)
    t = until(s, t)
    t = feed(s, None, 0.3, t)
    feed_at(s, 200, 60, 0.3, t)
    assert s.helper.active


def test_pause_mouse_off_leaving_the_view_and_quitting_all_stop_it(tmp_path):
    for stopper, why in ((ord("p"), "paused"), (ord("m"), "a key was pressed")):
        s, _, _ = helped(tmp_path, step(), step())
        t = feed(s, "palm", 0.3, 0.0)
        s.ask_helper("do something", t)
        s.on_key(stopper, t)
        assert not s.helper.active
    s, _, _ = helped(tmp_path, step(), step())
    s.ask_helper("do something", feed(s, "palm", 0.3, 0.0))
    s.set_locked(True)
    assert not s.helper.active and "paused" in s.last_event
    s, _, _ = helped(tmp_path, step(), step())
    s.ask_helper("do something", feed(s, "palm", 0.3, 0.0))
    s.close()
    assert not s.helper.active


def risky(tmp_path, **kw):
    s, mouse, script = helped(tmp_path, step(x=700, y=600, target="Send button"), step("done", text="Sent."), **kw)
    t = feed(s, "palm", 0.3, 0.0)
    s.ask_helper("send it", t)
    t = until(s, t)
    assert s.helper.state == "asking" and ("click",) not in mouse.calls
    return s, mouse, t


def test_a_risky_step_waits_and_a_thumb_up_says_yes(tmp_path):
    s, mouse, t = risky(tmp_path)
    line = helper_line(s, t)
    assert line.startswith('AI 1/12: click "Send button"? thumb up = yes, thumb down = no, or point up / down | ')
    t = feed(s, "palm", 3.0, t)                                      # nothing runs on its own
    assert s.helper.state == "asking" and ("click",) not in mouse.calls
    t = feed(s, "thumb_up", 0.9, t)
    assert mouse.calls[-2:] == [("move", 700, 300), ("click",)] and s.helper.state != "asking"
    until(s, feed(s, "palm", 1.3, t))
    assert s.last_event == "AI done: Sent."


def test_thumb_down_n_or_silence_say_no_and_y_says_yes(tmp_path):
    s, mouse, t = risky(tmp_path)
    t = feed(s, "thumb_down", 1.5, t)                                # held on after the answer: it must not scroll
    assert not s.helper.active and "you said no" in s.last_event and kinds(mouse) == []
    t = feed(s, "palm", 0.4, t)
    feed(s, "thumb_down", 0.5, t)
    assert kinds(mouse)[-1] == "scroll"                              # a fresh thumb down scrolls as always
    s, mouse, t = risky(tmp_path)
    s.on_key(ord("n"), t)
    assert not s.helper.active and "you said no" in s.last_event and ("click",) not in mouse.calls
    s, mouse, t = risky(tmp_path)
    s.on_key(ord("y"), t)
    assert ("click",) in mouse.calls and s.helper.state == "settling"
    s, mouse, t = risky(tmp_path)
    feed(s, "palm", 31.0, t, dt=0.1)
    assert not s.helper.active and "no answer" in s.last_event and ("click",) not in mouse.calls


def test_a_hand_that_only_points_answers_by_pointing_up_or_down(tmp_path):
    s, mouse, t = risky(tmp_path)
    t = feed(s, "palm", 0.3, t)
    assert s.chooser.mode == "open" and s.chooser.view["items"] == ["yes", "", "", "no", "", ""]
    t = feed_at(s, 0, -30, 1.0, t)                                   # up
    assert ("click",) in mouse.calls and s.chooser.mode == "idle" and s.helper.state != "asking"
    s, mouse, t = risky(tmp_path)
    t = feed(s, "palm", 0.3, t)
    feed_at(s, 0, 30, 1.0, t)                                        # down
    assert not s.helper.active and "you said no" in s.last_event and ("click",) not in mouse.calls
    s, mouse, t = risky(tmp_path)
    t = feed(s, "palm", 0.3, t)
    feed_at(s, 30, 0, 2.0, t)                                        # sideways is not an answer
    assert s.helper.state == "asking" and ("click",) not in mouse.calls


def test_in_face_mode_the_mouth_says_yes_and_the_eyebrows_stop_or_say_no(tmp_path):
    s, mouse, _ = helped(tmp_path, step(x=700, y=600, target="Send button"), step(), step())
    s.on_key(ord("f"), 0.0)
    t = face_feed(s, 0.5, 0.0)
    s.ask_helper("send it", t)
    while s.helper.state == "thinking":
        t = face_feed(s, DT, t)
        time.sleep(0.003)
    assert s.helper.state == "asking" and "open mouth = yes, eyebrows = no" in helper_line(s, t)
    t = face_feed(s, 0.9, t, jaw=0.8)
    assert ("click",) in mouse.calls and s.helper.state == "settling"
    t = face_feed(s, 1.3, t)
    while s.helper.state == "thinking":
        t = face_feed(s, DT, t)
        time.sleep(0.003)
    assert s.helper.state == "showing"
    t = face_feed(s, 0.6, t, brow=0.8)
    assert not s.helper.active and "stop sign" in s.last_event
    face_feed(s, 1.0, t, brow=0.8)                                   # eyebrows still up: not a right click
    assert "right_click" not in kinds(mouse)


def test_the_glow_follows_the_helper_and_hides_while_the_picture_is_taken(tmp_path):
    import threading
    taking, taken = threading.Event(), threading.Event()

    def grab():
        taking.set()
        taken.wait(5)
        return Shot()

    ov = FakeOverlay()
    h = Helper(Script(step(x=300, y=400, target="Note field"), step(x=700, y=600, target="Send button")), grab, (1000, 500),
               sleep=lambda s: None)
    h.armed = True
    s, mouse = make(tmp_path, helper=h, overlay=ov)
    t = feed(s, "palm", 0.5, 0.0)
    s.ask_helper("write and send", t)
    assert taking.wait(5)
    t = feed(s, "palm", 0.1, t)
    assert ov.states[-1] == {"show": False}                          # nothing of ours is in the picture
    taken.set()
    t = until(s, t)
    t = feed(s, "palm", 0.75, t)
    last = ov.states[-1]
    assert (last["x"], last["y"], last["tag"], last["hint"]) == (300.0, 200.0, "AI 1/12", 'click "Note field"')
    assert 0.4 < last["ring"] < 0.6 and "AI 1/12 | click" in s.mini_label
    t = feed(s, "palm", 1.0, t)
    assert [st["pulse"] for st in ov.states if "pulse" in st] == ["click"]
    t = until(s, feed(s, "palm", 1.3, t))
    feed(s, "palm", 0.2, t)
    assert ov.states[-1]["tag"] == "AI 2/12: YES OR NO?" and ov.states[-1]["hint"] == 'click "Send button"?'
