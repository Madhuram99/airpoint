"""What the side panel and the message line say, for a session in different states. Nothing is drawn."""
from airpoint import panel
from airpoint.fsm import Event
from airpoint.profile import Profile
from tests.hands import BY_POSE
from tests.test_session import DT, blank, face_feed, feed, make


def test_the_word_follows_what_the_hand_is_doing_in_plain_language(tmp_path):
    s, _ = make(tmp_path)
    s.tick(blank(), None, None, 0.0)
    assert panel.status(s) == ("No hand", "Show your hand to the camera.", "idle")
    t = feed(s, "palm", 0.3, DT)
    assert panel.status(s) == ("Moving", "Open palm moves the cursor.", "track")
    t = feed(s, "pinch", 0.2, t)
    assert panel.status(s) == ("Pressing", "Let go to click. Keep holding to drag.", "act")
    t = feed(s, "pinch", 0.6, t)
    assert panel.status(s) == ("Dragging", "Move, then let go to drop.", "act")
    t = feed(s, "palm", 0.3, t)
    t = feed(s, "thumb_down", 0.3, t)
    assert panel.status(s) == ("Scrolling down", "Let go of the sign to stop.", "act")
    t = feed(s, "palm", 0.3, t)
    t = feed(s, "v_sign", 0.2, t)
    assert panel.status(s) == ("V sign", "Hold it: right click.", "act")
    s.on_key(ord("p"), t)
    assert panel.status(s)[0] == "Paused" and panel.status(s)[2] == "warn"
    s.on_key(ord("p"), t)
    s.on_key(ord("r"), t)
    assert panel.status(s) == ("Recording", "Follow the card over the picture. R goes back.", "idle")


def test_the_detail_line_says_what_holding_still_does_when_that_is_how_you_click(tmp_path):
    s, _ = make(tmp_path)
    s.on_key(ord("d"), 0.0)
    feed(s, "palm", 0.3, 0.0)
    assert panel.status(s)[1] == "Hold still for 1 s to click."
    s.on_key(ord("w"), 0.5)
    assert panel.status(s)[1] == "Hold still for 1 s to open the wheel."


def test_the_sign_rows_follow_this_users_roles_and_light_the_sign_in_view(tmp_path):
    prof = Profile(user="ana", roles={"grip": ["fist"], "right_click": ["pinch"], "scroll_up": ["thumb_up"], "scroll_down": []},
                   skipped=["thumb_down"], signs={"rock": {"action": "say", "arg": "hello", "hold_s": 0.6}})
    s, _ = make(tmp_path, profile=prof)
    feed(s, "fist", 0.3, 0.0)
    rows = {r["name"]: r for r in panel.sign_rows(s, s.mode.label)}
    assert list(rows) == ["Open palm", "Pinch", "Fist", "V sign", "Thumb up", "Thumb down", "Rock"]
    assert rows["Fist"] == dict(glyph="fist", name="Fist", does="click, hold to drag", tone="act", lit=True, off=False)
    assert rows["Pinch"]["does"] == "right click" and rows["V sign"]["does"] == "move the cursor"   # no role: it only moves
    assert rows["V sign"]["tone"] == "track" and not rows["Pinch"]["lit"]
    assert rows["Thumb down"]["off"] and rows["Thumb down"]["does"] == "not used"
    assert rows["Rock"] == dict(glyph=None, name="Rock", does='says "hello"', tone="act", lit=False, off=False)


def test_what_a_taught_sign_does_is_said_in_a_few_words():
    assert panel.does(Event("hotkey", ("command", "c"))) == "presses command+c"
    assert panel.does(Event("type", "on my way")) == 'types "on my way"' and panel.does(Event("open", "Notes")) == "opens Notes"
    assert panel.does(Event("talk")) == "starts and stops talk" and panel.does(Event("ask", "x")) == "asks the AI helper"
    assert panel.does(Event("pause")) == "pauses AirPoint"


def test_face_mode_lists_the_faces_signs(tmp_path):
    s, _ = make(tmp_path)
    s.on_key(ord("f"), 0.0)
    t = face_feed(s, 0.5, 0.0)
    rows = panel.sign_rows(s, s.mode.label)
    assert [r["name"] for r in rows] == ["Turn head", "Open mouth", "Raise brows", "Hold still"]
    assert rows[0]["lit"] and rows[3]["does"] == "open the wheel" and not rows[3]["off"]      # face mode switches the wheel on
    face_feed(s, 0.3, t, jaw=0.8)
    assert panel.sign_rows(s, s.mode.label)[1]["lit"] and panel.status(s)[0] in ("Pressing", "Dragging")
    s.tick(blank(), None, None, t + 5.0, face=None)
    assert panel.status(s)[:2] == ("No face", "Show your face to the camera.")


def test_the_switch_rows_show_each_key_and_its_state(tmp_path):
    s, _ = make(tmp_path, mouse_on=False)
    rows = {r["key"]: r for r in panel.switch_rows(s)}
    assert list(rows) == ["M", "D", "W", "O", "B", "F", "L", "A", "P", "?"]
    assert rows["M"] == dict(key="M", name="Mouse", value="off", tone=None) and rows["O"]["value"] == "none"   # no glow here
    assert rows["B"] == dict(key="B", name="Filter", value="full", tone="track") and rows["L"]["value"] == ""
    for key in "mdwf":
        s.on_key(ord(key), 0.0)
    s.on_key(ord("b"), 0.0)
    s.on_key(ord("b"), 0.0)
    s.on_key(ord("p"), 0.0)
    rows = {r["key"]: r for r in panel.switch_rows(s)}
    assert [rows[k]["value"] for k in "MDWF"] == ["on", "on", "on", "on"] and rows["M"]["tone"] == "track"
    assert rows["B"]["value"] == "off" and rows["B"]["tone"] is None
    assert rows["P"] == dict(key="P", name="Pause", value="paused", tone="act")


def test_a_key_that_changes_something_says_what_it_did(tmp_path):
    s, _ = make(tmp_path, mouse_on=False)
    said = {}
    for key in "mdwkvzxobshft":
        s.toast = None
        s.on_key(ord(key), 1.0)
        said[key] = s.toast[0] if s.toast else None
    assert said["m"].startswith("Mouse control is on.") and said["d"].startswith("Dwell click is on.")
    assert said["w"].startswith("The wheel is on. Hold still for 1 s") and said["x"].startswith("X-ray is on")
    assert said["z"] == "Zoom 1.5x." and said["b"].startswith("Filter: smoothing only.")
    assert said["o"].startswith("No glow here") and said["s"] == "Sounds are off." and said["h"] == "Hints are off."
    assert said["f"].startswith("Face mode. Turn your head to move")
    assert said["k"] is None                                   # no trained network in this session: nothing to switch
    assert all(text.endswith(".") for text in said.values() if text)
    s.toast = None
    s.on_key(ord("t"), 1.0)                                     # face mode: the simulated tremor is for the hand
    assert s.toast[:2] == ("Record, teach and the simulated tremor are for the hand. Press F first.", "warn")
    nameless, _ = make(tmp_path, user=None)
    for key, what in (("c", "Calibration"), ("n", "Teaching a sign"), ("r", "Recording")):
        nameless.on_key(ord(key), 1.0)
        assert nameless.toast[:2] == (f"{what} needs your name. Start AirPoint with --user <name>.", "warn")


def test_talk_that_cannot_start_says_why(tmp_path):
    s, _ = make(tmp_path, mouse_on=False)
    s.on_key(ord("l"), 1.0)
    assert s.toast[:2] == ("Talk needs mouse control. Press M first.", "warn") and s.talker.state == "idle"
    s.on_key(ord("m"), 1.0)
    s.on_key(ord("p"), 1.0)
    s.on_key(ord("l"), 1.0)
    assert s.toast[0] == "Talk is off while paused. Press P first."


def test_the_message_line_has_one_thing_to_say_the_most_pressing_first(tmp_path):
    s, _ = make(tmp_path, mouse_on=False)
    t = feed(s, "palm", 0.3, 0.0)
    assert panel.bar_model(s, t, s.lines) == dict(text=panel.MOUSE_OFF, tone="idle")
    s.on_key(ord("m"), t)
    assert panel.bar_model(s, t, s.lines)["text"].startswith("Mouse control is on.")       # what the key just did
    t = feed(s, "palm", panel.TOAST_S + 0.2, t)
    assert panel.bar_model(s, t, s.lines) == dict(text="", tone="idle")                    # nothing to say: say nothing
    t = feed(s, "pinch", 0.2, t)
    t = feed(s, "palm", 0.1, t)
    assert panel.bar_model(s, t, s.lines) == dict(text="Clicked.", tone="act")
    t = feed(s, "palm", panel.DID_S + 0.2, t)
    assert panel.bar_model(s, t, s.lines)["text"] == ""
    assert panel.bar_model(s, t, (("", "act"), (None, "warn"), ("Move closer to the camera.", "warn"), ("later", "track"))) == \
        dict(text="Move closer to the camera.", tone="warn")
    s.say("Hints are off.", t)
    assert panel.bar_model(s, t, (("Move closer to the camera.", "warn"),))["text"] == "Hints are off."


def test_what_just_happened_is_said_as_a_sentence():
    assert panel.did([Event("click")]) == "Clicked." and panel.did([Event("scroll", -2)]) == "Scrolling down."
    assert panel.did([Event("hotkey", ("command", "a"))]) == "Pressed command+a."
    assert panel.did([Event("settle", 0.35), Event("type", "I will be late. ")]) == 'Typed "I will be late.".'
    assert panel.did([Event("say", "hello")]) == 'Said "hello".' and panel.did([Event("talk")]) == ""


def test_question_mark_shows_and_hides_the_list_of_keys(tmp_path):
    s, _ = make(tmp_path)
    plain = s.tick(blank(), BY_POSE["palm"], "Right", 0.0).copy()
    s.on_key(ord("?"), 0.0)
    assert s.help and (s.tick(blank(), BY_POSE["palm"], "Right", DT)[:400, :700] != plain[:400, :700]).mean() > 0.3
    s.on_key(ord("/"), 0.1)                                     # the same key without shift
    assert not s.help
