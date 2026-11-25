import numpy as np
import pytest

from airpoint import hud, ui
from airpoint.tremor import Reading
from tests.hands import BY_POSE


def room(w=640, h=360, v=90):
    return np.full((h, w, 3), v, np.uint8)


def count(img, color):
    return int((img == np.array(color, np.uint8)).all(axis=2).sum())


def near(img, color, tol=60):
    """Pixels close to a colour: thin anti-aliased lines rarely hold the exact one."""
    return int((np.abs(img.astype(int) - np.array(color, int)).sum(axis=2) < tol).sum())


def test_the_stage_is_the_camera_picture_at_the_size_it_is_shown():
    full = np.zeros((1080, 1920, 3), np.uint8)
    assert hud.canon_size(full) == (640, 360) and hud.canon_size(np.zeros((480, 640, 3), np.uint8)) == (640, 480)
    assert hud.stage(full, 1.0).shape == (360, 640, 3) and hud.stage(full, 2.5).shape == (900, 1600, 3)
    assert hud.canon_view(np.zeros((720, 1280, 3), np.uint8)).shape == (360, 640, 3)
    same = np.zeros((360, 640, 3), np.uint8)
    assert hud.stage(same, 1.0) is same and hud.stage(same, 2.0).shape == (720, 1280, 3)
    assert hud.brightness(room(1920, 1080, 120)) == 120.0 and hud.brightness(np.zeros((360, 640, 3), np.uint8)) == 0.0


def test_marks_on_the_picture_keep_their_place_and_size_at_any_scale():
    for k in (1.0, 2.0, 2.5):
        img = np.zeros((int(360 * k), int(640 * k), 3), np.uint8)
        hud.draw_cursor(img, (320.0, 180.0), (320.0, 180.0))
        ys, xs = np.nonzero(img.any(axis=2))
        assert abs(xs.mean() / k - 320) < 1 and abs(ys.mean() / k - 180) < 1
        assert 14 < (xs.max() - xs.min()) / k < 22


def test_dwell_ring_draws_only_with_progress():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    hud.draw_dwell_ring(frame, (50, 50), 0.0)
    assert not frame.any()
    hud.draw_dwell_ring(frame, (50, 50), 0.5)
    assert count(frame, ui.CAM_YELLOW) > 20 and not frame[:, 80:].any()


def test_the_hand_is_blue_while_it_moves_the_cursor_and_yellow_while_a_sign_acts():
    tracking, acting = room(), room()
    hud.draw_skeleton(tracking, BY_POSE["palm"])
    hud.draw_skeleton(acting, BY_POSE["pinch"], lit=True)
    assert count(tracking, ui.CAM_BLUE) > 200 and count(tracking, ui.CAM_YELLOW) == 0
    assert count(acting, ui.CAM_YELLOW) > 200 and count(acting, ui.CAM_BLUE) == 0


def test_the_comfort_box_is_four_corners_and_says_what_it_is():
    on, off = room(), room()
    hud.draw_region(on, (0.15, 0.15, 0.85, 0.80), True)
    hud.draw_region(off, (0.15, 0.15, 0.85, 0.80), False)
    assert count(on, ui.CAM_BLUE) > 100 and count(off, ui.CAM_BLUE) == 0
    assert (on[150:200, 250:400] == 90).all()                       # the middle of the picture stays clear


def test_the_prompt_card_stays_at_the_top_of_the_picture():
    frame = np.zeros((360, 640, 3), np.uint8)
    hud.draw_prompt(frame, "HOLD palm   7s", "tilt | kept 12", "SIGN LOST")
    assert frame[:90].any() and not frame[100:].any()               # never over a hand held in view
    assert count(frame, ui.CAM_RED) > 10
    plain = np.zeros((360, 640, 3), np.uint8)
    hud.draw_prompt(plain, "Learning...", "a few seconds")
    assert count(plain, ui.CAM_RED) == 0


def _wheel(**over):
    view = dict(mode="open", anchor=(320.0, 180.0), pick=1, progress=0.5, centre="click", dot=(0.3, -0.2),
                items=["talk", "right", "drag", "scroll", "double", "pause"])
    view.update(over)
    return view


def test_the_wheel_is_drawn_around_its_anchor_with_the_pointed_slice_lit():
    frame = room()
    hud.draw_wheel(frame, _wheel())
    ys, xs = np.nonzero((frame == np.array(ui.CAM_YELLOW, np.uint8)).all(axis=2))
    assert len(xs) > 300 and xs.mean() > 330 and ys.mean() < 180      # slice 1: up and to the right
    assert (frame[5:60, 5:200] == 90).all() and (frame[:, :230] == 90).all()
    idle = room()
    hud.draw_wheel(idle, _wheel(mode="idle"))
    assert (idle == 90).all()
    hub = room()
    hud.draw_wheel(hub, _wheel(pick=-1))
    ys, xs = np.nonzero((hub == np.array(ui.CAM_YELLOW, np.uint8)).all(axis=2))
    assert abs(xs.mean() - 320) < 12 and abs(ys.mean() - 180) < 16     # staying put: the centre is what is lit


def _model(**over):
    names = (("palm", "Open palm", "move the cursor", "track"), ("pinch", "Pinch", "click, hold to drag", "act"),
             ("fist", "Fist", "click, hold to drag", "act"), ("v_sign", "V sign", "right click", "act"),
             ("thumb_up", "Thumb up", "scroll up", "act"), ("thumb_down", "Thumb down", "scroll down", "act"))
    model = dict(word="Pressing", tone="act", detail="Let go to click. Keep holding to drag.",
                 signs=[dict(glyph=g, name=n, does=d, tone=t, lit=g == "pinch", off=False) for g, n, d, t in names],
                 switches=[dict(key=k, name=n, value=v, tone=t) for k, n, v, t in (
                     ("M", "Mouse", "on", "track"), ("D", "Dwell", "off", None), ("W", "Wheel", "off", None),
                     ("O", "Glow", "on", "track"), ("B", "Filter", "full", "track"), ("F", "Face", "off", None),
                     ("L", "Talk", "listening", "act"), ("A", "AI helper", "off", None), ("P", "Pause", "off", None),
                     ("?", "All keys", "", None))])
    model.update(over)
    return model


@pytest.mark.parametrize("theme", [ui.LIGHT, ui.DARK])
def test_the_side_panel_shows_the_word_the_lit_sign_and_the_switches(theme):
    img = hud.draw_panel(_model(), theme, 1.0, 360)
    assert img.shape == (360, hud.PANEL_W, 3) and count(img, theme.ground) > 0.5 * 360 * hud.PANEL_W
    yellow = (img == np.array(theme.yellow, np.uint8)).all(axis=2)
    assert yellow[14:52].sum() > 1500                                  # the word, on the action colour
    lit = yellow[103:245].sum(axis=1)
    assert lit[26:44].min() > 120 and lit[:22].max() == 0 and lit[50:].max() == 0     # the second row only
    assert yellow[255:].sum() > 100                                    # talk is busy: its state is marked
    assert count(img[255:], theme.blue) > 20                           # "on" and "full"
    big = hud.draw_panel(_model(), theme, 2.0, 360)
    assert big.shape == (720, 2 * hud.PANEL_W, 3)
    by, bx = np.nonzero((big == np.array(theme.yellow, np.uint8)).all(axis=2)[:104])
    sy, sx = np.nonzero(yellow[:52])
    assert abs(bx.mean() / 2 - sx.mean()) < 2 and abs(by.mean() / 2 - sy.mean()) < 2   # the same layout, twice as fine


def test_the_panel_is_quiet_when_nothing_acts_and_fits_more_signs_than_six():
    resting = hud.draw_panel(_model(word="No hand", tone="idle", detail="Show your hand to the camera.",
                                    signs=[dict(s, lit=False) for s in _model()["signs"]],
                                    switches=[dict(s, tone=None if s["tone"] == "act" else s["tone"]) for s in _model()["switches"]]),
                             ui.LIGHT, 1.0, 360)
    assert count(resting, ui.LIGHT.yellow) == 0
    many = [dict(s, lit=False) for s in _model()["signs"]] + [dict(glyph=None, name="Rock", does='says "hello"', tone="act", lit=True, off=False),
                                dict(glyph=None, name="Call me", does="opens Notes", tone="act", lit=False, off=True)]
    img = hud.draw_panel(_model(signs=many), ui.LIGHT, 1.0, 360)
    lit = (img == np.array(ui.LIGHT.yellow, np.uint8)).all(axis=2)[103:245].sum(axis=1)
    assert lit[:100].max() < 60 and lit[108:120].min() > 150           # row seven of eight, inside the same space


def test_where_the_system_has_no_picture_of_a_sign_a_drawing_stands_in(monkeypatch):
    monkeypatch.setattr(ui, "emoji", lambda char, px: None)
    for glyph in ("palm", "face:mouth", "face:brows", "face:turn"):
        img = np.zeros((40, 40, 3), np.uint8)
        hud.draw_glyph(ui.Canvas(img, 2.0), glyph, 0, 0, 20, (255, 255, 255))
        assert img.any(), glyph
    blank = np.zeros((40, 40, 3), np.uint8)
    hud.draw_glyph(ui.Canvas(blank, 2.0), None, 0, 0, 20, (255, 255, 255))
    assert not blank.any()                                             # a taught sign has no picture


def test_the_message_line_is_one_sentence_with_a_mark_for_its_kind():
    said = hud.draw_bar(dict(text="Dwell click is on.", tone="track"), ui.LIGHT, 1.0, 920)
    assert said.shape == (hud.BAR_H, 920, 3) and count(said[:, :40], ui.LIGHT.blue) > 20
    warn = hud.draw_bar(dict(text="Talk needs mouse control. Press M first.", tone="warn"), ui.LIGHT, 1.0, 920)
    assert count(warn[:, :40], ui.LIGHT.red) > 20
    empty = hud.draw_bar(dict(text="", tone="idle"), ui.LIGHT, 1.0, 920)
    assert (empty[4:, :700] == np.array(ui.LIGHT.ground, np.uint8)).all() and (empty[:, 800:] != np.array(ui.LIGHT.ground, np.uint8)).any()
    long = hud.draw_bar(dict(text="word " * 80, tone="track"), ui.LIGHT, 1.0, 920)
    assert (long[4:, 842:858] == np.array(ui.LIGHT.ground, np.uint8)).all()        # cut before the hint, not drawn over it
    assert hud.draw_bar(dict(text="x", tone="act"), ui.DARK, 2.5, 920).shape == (90, 2300, 3)


def test_picture_panel_and_line_make_one_window():
    picture, side = room(1600, 900), hud.draw_panel(_model(), ui.DARK, 2.5, 360)
    bar = hud.draw_bar(dict(text="hello", tone="track"), ui.DARK, 2.5, 920)
    assert hud.compose(picture, side, bar, ui.DARK).shape == (990, 2300, 3)


def test_the_small_view_is_the_picture_the_word_and_one_line():
    small = hud.mini_frame(room(800, 450), dict(word="V sign", tone="act", line="Hold it: right click."), ui.LIGHT, 1.0)
    assert small.shape == (240, 320, 3) and (small[:180] == 90).all()
    assert count(small[180:], ui.LIGHT.yellow) > 300
    assert hud.mini_frame(room(800, 450), dict(word="Moving", tone="track", line=""), ui.DARK, 2.0).shape == (480, 640, 3)
    assert hud.mini_frame(room(640, 480), dict(word="Moving", tone="track", line=""), ui.DARK, 1.0).shape == (300, 320, 3)


def test_help_lists_a_line_for_every_key_the_app_takes():
    frame = room()
    hud.draw_help(frame)
    assert (frame != 90).mean() > 0.5
    assert {key for key, _ in hud.HELP} >= set("MDWOBFLAPCNRXVZTKSHQ") and all(does for _, does in hud.HELP)


def _taps(**over):
    x = dict(clf="mlp/ana", fps=30.0, ms_tracker=5.0, ms_classifier=0.2,
             labels=["palm", "pinch", "fist", "v_sign", "thumb_up", "thumb_down"],
             probs=np.array([0.02, 0.9, 0.03, 0.02, 0.02, 0.01]), gate=0.75,
             votes=["palm", "pinch", "pinch", "pinch", "pinch"], label="pinch",
             state="PRESSING", state_s=0.21, last_event="click", trail=[],
             reading=Reading(), left=Reading(), mode="full", sim=False)
    x.update(over)
    return x


def test_xray_panel_has_a_fixed_width_at_any_height_and_scale():
    for h in (360, 480):
        assert hud.draw_xray(h, _taps()).shape == (h, hud.XRAY_W, 3)
    assert hud.draw_xray(360, _taps(), 2.5, ui.LIGHT).shape == (900, 1000, 3)
    empty = hud.draw_xray(360, _taps(probs=None, votes=[], label="none", state="RESTING"))
    assert empty.shape == (360, 400, 3)


def test_xray_panel_shows_the_winning_bar_the_active_state_and_the_curves():
    t = np.arange(120) / 30.0
    trail = [(ti, (300.0 + 12.0 * np.sin(2 * np.pi * 6.0 * ti), 200.0), (300.0, 200.0)) for ti in t]
    f = np.linspace(3.0, 12.0, 40)
    locked = Reading(status="tremor", amp=12.0, hz=6.0, locked=True, lock_hz=6.0, max_hz=12.0,
                     spectrum=(f, np.exp(-(f - 6.0) ** 2)))
    left = Reading(status="steady", amp=0.8, max_hz=12.0, spectrum=(f, 0.01 * np.exp(-(f - 6.0) ** 2)))
    theme = ui.DARK
    panel = hud.draw_xray(360, _taps(trail=trail, reading=locked, left=left, sim=True))
    plain = hud.draw_xray(360, _taps())
    assert count(plain[23:119], theme.yellow) > 800                    # the winning bar
    assert count(plain[141:180], theme.yellow) > 1000                  # the active state
    assert near(panel[217:255], theme.red) > 30 and near(panel[217:255], theme.blue) > 30         # raw and output
    assert near(panel[277:313], theme.red) > 30 and near(plain[217:313], theme.red) == 0
    assert near(panel[330:], theme.red) > 20 and near(plain[330:], theme.red) == 0                # "simulated" is said in red


def test_xray_shows_the_six_likeliest_classes_when_signs_were_taught():
    labels = ["palm", "pinch", "fist", "v_sign", "thumb_up", "thumb_down", "rock", "call_me"]
    probs = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.9, 0.04])
    panel = hud.draw_xray(360, _taps(labels=labels, probs=probs, label="rock"))
    assert panel.shape == (360, 400, 3)
    assert count(panel[23:119], ui.DARK.yellow) > 800                  # the taught sign's bar is among the six shown
    assert hud.draw_xray(360, _taps(labels=labels, probs=None, label="none")).shape == (360, 400, 3)
