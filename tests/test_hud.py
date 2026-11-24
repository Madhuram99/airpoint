import numpy as np

from airpoint import hud


def test_mini_frame_is_320x240_with_a_label_strip():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    small = hud.mini_frame(frame, "MOVING | palm 0.98 | mouse ON")
    assert small.shape == (240, 320, 3)
    assert small[230:, :, :].any()


def test_dwell_ring_draws_only_with_progress():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hud.draw_dwell_ring(frame, (50, 50), 0.0)
    assert not frame.any()
    hud.draw_dwell_ring(frame, (50, 50), 0.5)
    assert frame.any()


def test_canon_view_is_640_wide_and_keeps_the_aspect():
    assert hud.canon_view(np.zeros((1080, 1920, 3), np.uint8)).shape == (360, 640, 3)
    assert hud.canon_view(np.zeros((720, 1280, 3), np.uint8)).shape == (360, 640, 3)
    assert hud.canon_view(np.zeros((480, 640, 3), np.uint8)).shape == (480, 640, 3)


def test_zoomed_scales_the_shown_image():
    img = np.zeros((360, 640, 3), np.uint8)
    assert hud.zoomed(img, 1.0) is img
    assert hud.zoomed(img, 1.5).shape == (540, 960, 3)


def _width(text, scale, thickness=1):
    import cv2
    return cv2.getTextSize(text, hud.FONT, scale, thickness)[0][0]


def test_fit_scale_shrinks_only_when_needed():
    assert hud.fit_scale("short", 0.5, 600) == 0.5
    long = "x" * 200
    scale = hud.fit_scale(long, 0.5, 600)
    assert scale < 0.5 and _width(long, scale) <= 601


def test_worst_case_hud_lines_stay_legible_on_the_640_canvas():
    line1 = "SCROLLING | pose thumb_down 0.99 | ratio 1.23 | mlp/madhuramrathi | 30 fps"
    line2 = "last: drag_end, click | mouse [m] OFF | dwell [d] OFF | filter [b] smooth only | " + hud.KEY_HINT
    title = "HOLD thumb_down   10s"
    sub = "tilt, move closer, move away | kept 1508 | rejected 1508"
    assert hud.fit_scale(line1, 0.50, 620) >= 0.42
    assert hud.fit_scale(line2, 0.45, 620) >= 0.40
    assert hud.fit_scale(title, 0.95, 610, 2) == 0.95
    assert hud.fit_scale(sub, 0.45, 610) >= 0.40


def test_banner_and_prompt_draw_inside_the_frame():
    frame = np.zeros((360, 640, 3), np.uint8)
    hud.draw_banner(frame, "MOVING | pose palm 0.98", "last: none", True, line3=("shake 12 px", hud.WHITE))
    assert frame[:68].any() and not frame[70:].any()
    frame = np.zeros((360, 640, 3), np.uint8)
    hud.draw_prompt(frame, "HOLD palm   7s", "tilt | kept 12", "SIGN LOST")
    assert frame[:86].any() and not frame[88:].any()      # top strip: never over the hand


def _taps(**over):
    from airpoint.tremor import Reading
    x = dict(clf="mlp/ana", fps=30.0, ms_tracker=5.0, ms_classifier=0.2,
             labels=["palm", "pinch", "fist", "v_sign", "thumb_up", "thumb_down"],
             probs=np.array([0.02, 0.9, 0.03, 0.02, 0.02, 0.01]), gate=0.75,
             votes=["palm", "pinch", "pinch", "pinch", "pinch"], label="pinch",
             state="PRESSING", state_s=0.21, last_event="click", trail=[],
             reading=Reading(), left=Reading(), mode="full", sim=False)
    x.update(over)
    return x


def test_xray_panel_draws_at_both_common_heights_and_with_nothing_to_show():
    for h in (360, 480):
        assert hud.draw_xray(h, _taps()).shape == (h, hud.XRAY_W, 3)
    empty = hud.draw_xray(360, _taps(probs=None, votes=[], label="none", state="RESTING"))
    assert empty.shape == (360, 400, 3)


def test_xray_panel_shows_the_winning_bar_the_active_state_and_the_curves():
    from airpoint.tremor import Reading
    freqs = np.arange(3.0, 12.01, 0.1)
    power = np.exp(-((freqs - 6.0) ** 2) / 0.1)
    trail = [(i / 30, (320 + 14 * np.sin(2 * np.pi * 6 * i / 30), 180.0), (320.0, 180.0)) for i in range(120)]
    locked = Reading(status="tremor", hz=6.0, amp=12.0, fs=30.0, max_hz=12.0, locked=True, lock_hz=6.0,
                     spectrum=(freqs, power))
    left = Reading(status="steady", amp=0.8, fs=30.0, max_hz=12.0, spectrum=(freqs, power * 0.01))
    panel = hud.draw_xray(360, _taps(trail=trail, reading=locked, left=left, sim=True))
    plain = hud.draw_xray(360, _taps())
    accent = np.array(hud.ACCENT_ON, np.uint8)
    assert (panel[44:52, 100:260] == accent).all(axis=2).all()            # the pinch bar (gate tick is at x 265)
    assert (panel[143:159, 270:390] == accent).all(axis=2).mean() > 0.5   # the PRESSING box is filled
    assert (panel[217:263] != plain[217:263]).any()                        # the scope has curves
    assert (panel[283:325] != plain[283:325]).any()                        # the spectrum has curves
