"""Teaching a sign end to end, with synthetic hands and no camera."""
import csv
import json
import time

import numpy as np

from tests.hands import BY_POSE, ROCK
from tests.test_session import blank, feed, make
from tests.test_train import _write_user

DT = 0.1


def keys(s, text, t=0.0):
    for ch in text:
        s.on_key(ord(ch), t)


def base(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_user(data / "base.csv", "base", np.random.default_rng(0), 1.0, np.array([0.0, 0.0, 0.0]))


def hold(s, pts, t, until):
    """Tick with one hand until the teach screen reaches `until`. -> time."""
    while s.mode.phase != until and t < 400:
        s.tick(blank(), pts, None if pts is None else "Right", t)
        t += DT
    return t


def wait_trained(s, t):
    deadline = time.time() + 60
    while s.mode.phase == "training" and time.time() < deadline:
        s.tick(blank(), BY_POSE["palm"], "Right", t)
        t += DT
        time.sleep(0.02)
    return t


def teach_rock(s, action_key="1", arg="hello there"):
    s.on_key(ord("n"), 0.0)
    keys(s, "rock")
    s.on_key(13, 0.0)
    t = hold(s, ROCK, 0.0, "action")
    s.on_key(ord(action_key), t)
    if arg is not None:
        keys(s, arg, t)
        s.on_key(13, t)
    return wait_trained(s, t)


def test_teach_a_sign_that_speaks_then_use_it(tmp_path):
    base(tmp_path)
    s, mouse = make(tmp_path, mouse_on=False)
    assert not s.typing
    s.on_key(ord("n"), 0.0)
    assert s.mode.name == "teach" and s.typing
    keys(s, "rock")
    s.on_key(13, 0.0)
    assert s.mode.phase == "capture" and not s.typing
    t = hold(s, ROCK, 0.0, "action")
    assert s.mode.frames >= 75                                # 8 s at 10 fps
    s.on_key(ord("1"), t)
    assert s.mode.phase == "arg" and s.typing
    keys(s, "hello there", t)
    s.on_key(13, t)
    assert s.mode.phase == "training"
    s.on_key(27, t)                                           # Esc is ignored while training
    assert s.mode.name == "teach"
    t = wait_trained(s, t)
    assert s.mode.phase == "done", s.mode.message
    assert "rock: 100% of its own frames recognised" in s.mode.message
    profile = json.loads((tmp_path / "profiles" / "ana.json").read_text())
    assert profile["signs"] == {"rock": {"action": "say", "arg": "hello there", "hold_s": 0.6}}
    assert profile["model"] == "models/personal_ana.joblib" and "rock" in s.clf.labels

    s.on_key(ord(" "), t)                                     # any key: back to live
    assert s.mode.name == "live"
    t = feed(s, "palm", 0.5, t)
    for _ in range(12):                                       # hold the new sign for 1.2 s
        s.tick(blank(), ROCK, "Right", t)
        t += DT
    assert s.done == [["say", "hello there"]] and mouse.calls == []
    s.close()


def test_a_built_in_name_is_refused_and_esc_cancels(tmp_path):
    s, _ = make(tmp_path)
    s.on_key(ord("n"), 0.0)
    keys(s, "Fist")
    s.on_key(13, 0.0)
    assert s.mode.phase == "name" and "built-in" in s.mode.error
    s.on_key(127, 0.0)                                        # backspace
    assert s.mode.text == "Fis"
    s.on_key(27, 0.0)
    assert s.mode.name == "live"


def test_an_empty_name_gets_a_number_and_no_hand_means_no_sign(tmp_path):
    base(tmp_path)
    s, _ = make(tmp_path)
    s.on_key(ord("n"), 0.0)
    s.on_key(13, 0.0)
    assert s.mode.sign == "sign1"
    hold(s, None, 0.0, "failed")
    assert "only 0 frames" in s.mode.message
    assert not any(r["pose"] == "sign1" for r in csv.DictReader(open(tmp_path / "data" / "ana_signs.csv")))
    s.on_key(ord("x"), 0.0)
    assert s.mode.name == "live" and not s.xray               # the key closed the screen; it was not a live key


def test_key_combo_signs_are_validated_and_stored_as_lists(tmp_path):
    base(tmp_path)
    s, mouse = make(tmp_path)
    s.on_key(ord("n"), 0.0)
    keys(s, "rock")
    s.on_key(13, 0.0)
    t = hold(s, ROCK, 0.0, "action")
    s.on_key(ord("2"), t)
    keys(s, "ctrl+alt+del", t)
    s.on_key(13, t)
    assert s.mode.phase == "arg" and "not a key combo" in s.mode.error
    for _ in range(20):
        s.on_key(127, t)
    keys(s, "cmd+c", t)
    s.on_key(13, t)
    t = wait_trained(s, t)
    assert s.profile.signs["rock"] == {"action": "hotkey", "arg": ["command", "c"], "hold_s": 0.6}
    s.on_key(ord(" "), t)
    t = feed(s, "palm", 0.5, t)
    for _ in range(12):
        s.tick(blank(), ROCK, "Right", t)
        t += DT
    assert ("hotkey", "command", "c") in mouse.calls
    s.close()


def test_forgetting_the_only_sign_brings_the_shared_model_back(tmp_path):
    base(tmp_path)
    s, _ = make(tmp_path)
    t = teach_rock(s)
    s.on_key(ord(" "), t)
    assert "rock" in s.clf.labels
    s.on_key(ord("n"), t)
    s.on_key(ord("1"), t)                                     # 1 forgets rock
    t = wait_trained(s, t)
    assert s.mode.phase == "done" and "forgot rock" in s.mode.message
    assert s.profile.signs == {} and s.clf is s.rules         # tests start with the rules as the shared model
    assert json.loads((tmp_path / "profiles" / "ana.json").read_text())["model"] is None
    assert not any(r["pose"] == "rock" for r in csv.DictReader(open(tmp_path / "data" / "ana_signs.csv")))
    s.close()
